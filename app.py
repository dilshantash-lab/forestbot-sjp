import os
import streamlit as st
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# 0. API Keys Setup
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
INDEX_NAME = "forest-bot-index"

# 1. Page Configuration
st.set_page_config(page_title="ForestBot.SJP", page_icon="🌲", layout="wide")

# 2. Sidebar - Chat History & Info
with st.sidebar:
    st.title("🌲 ForestBot.SJP")
    st.markdown("**AI-Powered Forestry Law & Environmental Science Assistant**")
    st.markdown("---")
    
    # New Chat Button
    if st.button("➕ Start New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.subheader("📜 Chat History")
    
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        user_queries = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
        for idx, q in enumerate(reversed(user_queries), 1):
            short_q = q[:30] + "..." if len(q) > 30 else q
            st.caption(f"{idx}. {short_q}")
    else:
        st.info("No chat history in this session yet.")
        
    st.markdown("---")
    st.caption("University of Sri Jayewardenepura | Academic Research Project")

# 3. Main Title
st.title("🌲 ForestBot.SJP")
st.subheader("Interactive Knowledge Assistant for Sri Lankan Forestry & Environmental Sciences")
st.markdown("---")

# 4. RAG Initialization Function
@st.cache_resource
def initialize_rag():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.1)
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    qa_template = PromptTemplate(
        "You are ForestBot.SJP, an expert AI assistant for Sri Lankan Forestry & Environmental Sciences.\n"
        "Context information from university documents and forestry laws is provided below:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "INSTRUCTIONS:\n"
        "1. Base your answer strictly on the provided context.\n"
        "2. Provide exact legal sections, fines (LKR), and academic details if available.\n"
        "3. Answer thoroughly and concisely.\n\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    
    query_engine = index.as_query_engine(
        llm=llm,
        text_qa_template=qa_template,
        similarity_top_k=8
    )
    
    return query_engine

with st.spinner("Connecting Pinecone Cloud Vector Store and Groq LLM..."):
    query_engine = initialize_rag()

st.caption("Connected to Cloud Pinecone Index: `forest-bot-index` | Groq Llama 3 Active")

# 5. Chat History Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 6. Display Previous Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.markdown("---")
            st.markdown("##### 📚 **Sources & References Cited:**")
            st.markdown(message["sources"])

# 7. User Input Processing
if prompt := st.chat_input("Ask ForestBot.SJP any forestry or environmental law question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching relevant forestry laws and calculating response..."):
            response = query_engine.query(prompt)
            
            sources_text = ""
            seen_sources = set()
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    meta = node.node.metadata
                    raw_file_name = meta.get('file_name', 'Unknown Document')
                    page_num = meta.get('page_label', meta.get('page_number', '1'))
                    clean_title = raw_file_name.replace('+', ' ').replace('_', ' ').replace('.pdf', '')
                    
                    source_id = f"{raw_file_name}_{page_num}"
                    if source_id not in seen_sources and len(seen_sources) < 2:
                        seen_sources.add(source_id)
                        sources_text += (
                            f"• **Research Paper / Document:** `{raw_file_name}`  \n"
                            f"  📌 **Extracted Title:** {clean_title}  \n"
                            f"  📄 **Cited Page:** {page_num}\n\n"
                        )
            
            st.markdown(response.response)
            
            if sources_text:
                st.markdown("---")
                st.markdown("##### 📚 **Sources & Exact Citations:**")
                st.markdown(sources_text)
                    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response.response,
        "sources": sources_text
    })
    st.rerun()
