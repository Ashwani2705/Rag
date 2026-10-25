# ui/app.py
import streamlit as st
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG + GraphRAG Demo", layout="wide", page_icon="📄")
st.title("📄 Document Q&A with RAG & GraphRAG")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")

st.sidebar.header("🔍 Query Settings")
query_mode = st.sidebar.selectbox("Retrieval Mode", options=["hybrid", "flat", "graph"], index=0)

st.sidebar.header("⚡ Semantic Cache")
bypass_cache = st.sidebar.checkbox("Bypass Cache (Force fresh LLM run)", value=False)

st.sidebar.header("📊 Live Evaluation")
enable_ragas = st.sidebar.checkbox("Compute RAGAS Metrics on every answer", value=True)

# Fetch Cache Stats
try:
    health_resp = requests.get(f"{api_url}/health", timeout=2)
    if health_resp.status_code == 200:
        c_stats = health_resp.json().get("cache", {})
        if c_stats.get("cached_queries", 0) > 0:
            st.sidebar.caption(
                f"⚡ **Cache Status**: {c_stats.get('cached_queries')} queries stored | "
                f"Hit Ratio: {c_stats.get('hit_ratio_pct')}%"
            )
except Exception:
    pass

# --- SESSION STATE FOR CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_ingested" not in st.session_state:
    st.session_state.file_ingested = True  # Ready to chat with existing database

# --- UPLOAD SECTION ---
st.subheader("Step 1: Upload a Document (Optional if already ingested)")
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
                    st.success(f"✅ Ingested {data['chunks_processed']} chunks! Cache refreshed.")
                    st.session_state.file_ingested = True
                    st.session_state.messages = []  # Clear history for new doc
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

st.divider()

# --- HELPER FUNCTIONS ---
def render_metrics(metrics: dict, latency_ms: float = 0, confidence: str | None = None, graph_path: str | None = None, chunk_count: int = 0):
    if not metrics:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    f_score = metrics.get("faithfulness", 0.0)
    r_score = metrics.get("answer_relevancy", 0.0)
    c_score = metrics.get("context_precision", 0.0)
    
    col1.metric(
        "🛡️ Faithfulness",
        f"{f_score * 100:.0f}%",
        help=metrics.get("faithfulness_reason", "Measures factual grounding in source chunks")
    )
    col2.metric(
        "🎯 Answer Relevancy",
        f"{r_score * 100:.0f}%",
        help=metrics.get("relevancy_reason", "Measures how directly the answer addresses the query")
    )
    col3.metric(
        "📊 Context Precision",
        f"{c_score * 100:.0f}%",
        help="Measures whether retrieved chunks were relevant to query"
    )
    col4.metric(
        "⚡ Latency",
        f"{latency_ms:.0f} ms"
    )

    with st.expander("🔬 View RAGAS Evaluation Details"):
        st.markdown(f"**Overall Quality Score:** `{metrics.get('overall_score', 0.0) * 100:.1f} / 100`")
        if chunk_count > 0:
            st.markdown(f"- **Evaluated Chunks:** `{chunk_count}` exact chunk(s) retrieved from DB by ID")
        if metrics.get("faithfulness_reason"):
            st.markdown(f"- **Faithfulness Analysis:** {metrics['faithfulness_reason']}")
        if metrics.get("relevancy_reason"):
            st.markdown(f"- **Relevancy Analysis:** {metrics['relevancy_reason']}")
        if confidence:
            st.markdown(f"- **Retrieval Confidence:** `{confidence.upper()}`")
        if graph_path and graph_path.lower() != "none":
            st.markdown(f"- **Graph Traversal Path:** `{graph_path}`")


def render_sources_and_chunks(sources: list[str], chunks: list[dict]):
    if not sources and not chunks:
        return
    
    with st.expander("📚 View Cited Sources & Exact Chunks from DB"):
        if chunks:
            for i, c in enumerate(chunks, 1):
                st.markdown(f"**Chunk #{i} — ID:** `{c.get('chunk_id')}`")
                if c.get("source"):
                    st.caption(f"📁 Document: `{c.get('source')}`")
                st.info(c.get("text", ""))
        elif sources:
            for i, src in enumerate(sources, 1):
                st.write(f"**Chunk ID {i}:** `{src}`")


# --- CHAT SECTION ---
st.subheader("Step 2: Ask Questions")
if not st.session_state.file_ingested:
    st.info("👆 Please upload a document first.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("is_cached"):
                match_type = message.get("cache_match_type", "semantic").upper()
                sim = message.get("cache_similarity")
                sim_text = f" | Similarity: {sim * 100:.1f}%" if sim and match_type == "SEMANTIC" else ""
                orig = f" (Original: *\"{message.get('matched_query')}\"*)" if message.get("matched_query") and message.get("matched_query") != message.get("query") else ""
                st.success(f"⚡ **Served from Semantic Cache** [{match_type}{sim_text}]{orig} in `{message.get('latency_ms', 0):.0f} ms`")

            st.markdown(message["content"])
            if message.get("metrics"):
                render_metrics(
                    message["metrics"],
                    latency_ms=message.get("latency_ms", 0),
                    confidence=message.get("confidence"),
                    graph_path=message.get("graph_path"),
                    chunk_count=len(message.get("chunks", [])),
                )
            render_sources_and_chunks(
                message.get("sources", []),
                message.get("chunks", [])
            )

    # Chat Input
    if prompt := st.chat_input("Ask a question about your document..."):
        # 1. Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Call the FastAPI /query endpoint
        with st.chat_message("assistant"):
            spinner_text = "Checking Semantic Cache & Processing..."
            with st.spinner(spinner_text):
                try:
                    payload = {
                        "query": prompt,
                        "mode": query_mode,
                        "evaluate": enable_ragas,
                        "bypass_cache": bypass_cache,
                    }
                    response = requests.post(f"{api_url}/query", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])
                        chunks = data.get("chunks", [])
                        metrics = data.get("metrics")
                        latency_ms = data.get("latency_ms", 0)
                        confidence = data.get("confidence")
                        graph_path = data.get("graph_path")
                        is_cached = data.get("is_cached", False)
                        match_type = data.get("cache_match_type")
                        similarity = data.get("cache_similarity")
                        matched_query = data.get("matched_query")
                        
                        # Display Cache Indicator
                        if is_cached:
                            m_type = (match_type or "semantic").upper()
                            sim_str = f" | Similarity: {similarity * 100:.1f}%" if similarity and m_type == "SEMANTIC" else ""
                            orig_str = f" (Original: *\"{matched_query}\"*)" if matched_query and matched_query != prompt else ""
                            st.success(f"⚡ **Served from Semantic Cache** [{m_type}{sim_str}]{orig_str} in `{latency_ms:.0f} ms`")

                        # Display answer
                        st.markdown(answer)
                        
                        # Display RAGAS Metrics
                        if metrics:
                            render_metrics(
                                metrics,
                                latency_ms=latency_ms,
                                confidence=confidence,
                                graph_path=graph_path,
                                chunk_count=len(chunks),
                            )
                        
                        # Display sources and exact chunks
                        render_sources_and_chunks(sources, chunks)
                        
                        # Save to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "metrics": metrics,
                            "sources": sources,
                            "chunks": chunks,
                            "latency_ms": latency_ms,
                            "confidence": confidence,
                            "graph_path": graph_path,
                            "is_cached": is_cached,
                            "cache_match_type": match_type,
                            "cache_similarity": similarity,
                            "matched_query": matched_query,
                            "query": prompt,
                        })
                    else:
                        st.error(f"API Error: {response.text}")
                        
                except Exception as e:
                    st.error(f"Connection Error: {e}")
