# ui/app.py
import streamlit as st
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG + GraphRAG Demo", layout="wide")
st.title("📄 Document Q&A with RAG & GraphRAG")

# --- SIDEBAR ---
st.sidebar.header("Connection")
api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")

st.sidebar.header("Query Mode")
query_mode = st.sidebar.selectbox("Mode", options=["hybrid", "flat", "graph"], index=0)

# --- SESSION STATE FOR CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_ingested" not in st.session_state:
    st.session_state.file_ingested = False

# --- UPLOAD SECTION ---
st.subheader("Step 1: Upload a Document")
uploaded_file = st.file_uploader("Choose a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    if st.button("🚀 Ingest Document"):
        with st.spinner("Running ingestion pipeline (Chunking, Embedding, Neo4j Graph)... This may take a minute."):
            try:
                # Send the file to the FastAPI /upload endpoint
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{api_url}/upload", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Ingested {data['chunks_processed']} chunks! You can now chat.")
                    st.session_state.file_ingested = True
                    st.session_state.messages = [] # Clear history for new doc
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

st.divider()

# --- CHAT SECTION ---
st.subheader("Step 2: Ask Questions")
if not st.session_state.file_ingested:
    st.info("👆 Please upload a document first.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about your document..."):
        # 1. Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Call the FastAPI /query endpoint
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {"query": prompt, "mode": query_mode}
                    response = requests.post(f"{api_url}/query", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data['answer']
                        sources = data.get('sources', [])
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Display sources in an expander
                        if sources:
                            with st.expander("📚 View Sources"):
                                for i, src in enumerate(sources):
                                    st.write(f"**Source {i+1}:** {src}")
                        
                        # Save to session state
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error(f"API Error: {response.text}")
                        
                except Exception as e:
                    st.error(f"Connection Error: {e}")