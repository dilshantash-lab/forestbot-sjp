import os
import streamlit as st
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# 0. API Keys Setup
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
        if "chat_engine" in st.session_state:
            del st.session_state["chat_engine"]
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
    llm = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.1)
    
    # Global Settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    qa_prompt_tmpl_str = (
        "You are ForestBot.SJP, an expert AI assistant for Sri Lankan Forestry & Environmental Sciences.\n"
        "Context information from retrieved documents is provided below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "STRICT SYSTEM INSTRUCTIONS:\n"
        "1. STRICT DOCUMENT ACCURACY: Never say 'I do not have access' or 'I cannot provide the paper' if document metadata or abstracts exist in the context above. Summarize using whatever text or metadata is available.\n"
        "2. CONTEXT ALIGNMENT: Base your answer STRICTLY on the retrieved context that matches the query/chat history. Ignore context chunks that belong to completely unrelated topics, regions, or papers.\n"
        "3. EXACT AUTHOR ATTRIBUTION: Always search the context for explicit Author Names (e.g., 'Ariyarathna, T.D.S., Disanayaka, R.M.S.H.'). NEVER say 'Various researchers' or 'Unknown authors' if author names are present.\n"
        "4. CONVERSATIONAL FOLLOW-UPS: Maintain absolute focus on the specific document or paper being discussed in recent chat history.\n"
        "5. LEGAL & PENALTY QUERIES: Provide exact Fine Amounts (LKR), Imprisonment Terms, and Section numbers ONLY IF explicitly requested.\n"
        "6. Keep responses academic, rigorous, concise, and accurate.\n\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    
    # llm එක සෘජුවම chat_engine වෙත ලබා දීම මඟින් OpenAI fallback වීම වළක්වයි
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        llm=llm,
        context_prompt=qa_prompt_tmpl_str,
        similarity_top_k=8
    )
    
    return chat_engine

with st.spinner("Connecting Pinecone Cloud Vector Store and Groq LLM..."):
    chat_engine = initialize_rag()

st.caption("Connected to Cloud Pinecone Index: `forest-bot-index` | Groq Llama 3 Inference Active")

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
            response = chat_engine.chat(prompt)
            
            sources_text = ""
            seen_sources = set()
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    meta = node.node.metadata
                    raw_file_name = meta.get('file_name', 'Unknown Document')
                    page_num = meta.get('page_label', meta.get('page_number', '1'))
                    
                    clean_title = raw_file_name.replace('+', ' ').replace('_', ' ').replace('.pdf', '')
                    
                    prompt_words = prompt.lower().split()
                    ans_lower = response.response.lower()
                    title_words = [w.lower() for w in clean_title.split() if len(w) > 3]
                    
                    is_relevant = any(w in ans_lower for w in title_words[:3]) or any(w in prompt.lower() for w in title_words[:3])
                    
                    source_id = f"{raw_file_name}_{page_num}"
                    if source_id not in seen_sources and (is_relevant or len(seen_sources) < 2):
                        seen_sources.add(source_id)
                        sources_text += (
                            f"• **Research Paper / Document:** `{raw_file_name}`  \n"
                            f"  📌 **Extracted Title:** {clean_title}  \n"
                            f"  📄 **Cited Page:** {page_num}\n\n"
                        )
            
            st.markdown(response.response)
            
            if sources_text:
                st.markdown("---")
                st.markdown("##### 📚 **Sources & Exact PDF Citations:**")
                st.markdown(sources_text)
                    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response.response,
        "sources": sources_text
    })
    st.rerun()
