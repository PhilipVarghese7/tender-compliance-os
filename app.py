import streamlit as st
import os, hashlib, json, logging, pandas as pd
import fitz
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

load_dotenv()
st.set_page_config(page_title="Tender Compliance Validator", layout="wide")


def determine_priority(text):
    text = text.lower()
    if any(word in text for word in ["must", "shall", "mandatory", "required", "penalty"]):
        return "Critical"
    if any(word in text for word in ["should", "expected", "preferred"]):
        return "Medium"
    return "Low"


def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def display_pdf(file_path, page_num):
    try:
        doc = fitz.open(file_path)
        # PyMuPDF is 0-indexed (Page 1 is index 0)
        page_index = max(0, int(page_num) - 1)
        
        if page_index >= len(doc):
            st.error(f"Page {page_num} not found in document.")
            return

        # Render page to an image (Matrix(2,2) gives 2x zoom for crisp text)
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        
        # Display the image natively in Streamlit
        st.image(pix.tobytes(), caption=f"RFP Source Evidence - Page {page_num}", use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering PDF: {e}")


def run_rag_extraction(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    search_queries = {
        "ELIGIBILITY": "Minimum turnover, years in business, net worth, certifications",
        "DEADLINES": "Submission dates, project timeline, delivery milestones",
        "PENALTIES": "Liquidated damages, fines, service level penalties",
        "DOCUMENTS": "Required annexures, certificates, financial statements",
        "TECHNICAL": "Technical specifications, scope of work, functional requirements",
        "LEGAL": "Payment terms, bank guarantee, indemnity, arbitration"
    }
    
    raw_requirements = []
    prog_bar = st.progress(0, text="Hunting for requirements...")
    
    for i, (cat, query) in enumerate(search_queries.items()):
        prog_bar.progress((i+1)/len(search_queries))
        docs = vectorstore.similarity_search(query, k=5)
        context = "\n".join([f"[Page {d.metadata['page']+1}]: {d.page_content}" for d in docs])
        
        prompt = f"Extract mandatory requirements for {cat} from: {context}. Return ONLY a JSON list: [{{'req': '...', 'page_no': 1}}]"
        
        try:
            response = llm.invoke(prompt)
            clean = response.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            for item in data:
                item['cat'] = cat
                item['pri'] = determine_priority(item['req']) 
                raw_requirements.append(item)
        except Exception as e:
            st.error(f"Error in {cat}: {e}")

    # --- DEDUPLICATION LOGIC ---
    unique = { (r['req'].lower()[:50], r['page_no']): r for r in raw_requirements }
    return list(unique.values())


def main():
    st.title("⚖️ Tender Compliance Validator")

    if "view_page" not in st.session_state: 
        st.session_state.view_page = 1

    uploaded_file = st.file_uploader("Upload Master RFP PDF", type="pdf")

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        doc_id = get_file_hash(file_bytes)
        storage_path = f"store_{doc_id}"
        temp_pdf = "active_tender.pdf"
        
        # Load/Create Vectorstore
        if os.path.exists(storage_path):
            vectorstore = Chroma(persist_directory=storage_path, embedding_function=OpenAIEmbeddings())
            with open(temp_pdf, "wb") as f: 
                f.write(file_bytes)
        else:
            with st.status("Building Semantic Index..."):
                with open(temp_pdf, "wb") as f: 
                    f.write(file_bytes)
                loader = PyMuPDFLoader(temp_pdf)
                chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
                vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=storage_path)
                vectorstore.persist()


        if "master_data" not in st.session_state:
            if st.button("🚀 Run Full Compliance Extraction"):
                st.session_state.master_data = run_rag_extraction(vectorstore)
                st.rerun()

        if "master_data" in st.session_state:
            st.divider()
            col_left, col_right = st.columns([0.4, 0.6])

            with col_left:
                st.header("📋 Master Checklist")
                df = pd.DataFrame(st.session_state.master_data)
                
                # Selection & Sync
                if not df.empty:
                    selection = st.selectbox("Verify Requirement Source:", range(len(df)), 
                        format_func=lambda x: f"P{df.iloc[x]['page_no']} | {df.iloc[x]['cat']} | {df.iloc[x]['req'][:45]}...")
                    
                    st.session_state.view_page = int(df.iloc[selection]['page_no'])

                    # Editable Table
                    st.data_editor(df, num_rows="dynamic", use_container_width=True)
                    
                    if st.button("✔️ Lock Checklist"): 
                        st.success("Checklist Finalized!")
                else:
                    st.warning("No requirements found.")

            with col_right:
                st.header("📄 PDF Evidence")
                display_pdf(temp_pdf, st.session_state.view_page)

if __name__ == "__main__":
    main()