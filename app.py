import os
import streamlit as st
from pinecone import Pinecone
from groq import Groq
from sentence_transformers import SentenceTransformer

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

# 4. Direct Pinecone & Groq Setup
@st.cache_resource
def setup_engines():
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    groq_client = Groq(api_key=GROQ_API_KEY)
    return embed_model, index, groq_client

with st.spinner("Connecting Pinecone Cloud Vector Store and Groq Engine..."):
    embed_model, pinecone_index, groq_client = setup_engines()

st.caption("Connected to Cloud Pinecone Index: `forest-bot-index` | Groq Llama-3 Active")

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
            # Query Vector
            query_vector = embed_model.encode(prompt).tolist()
            
            # Pinecone Vector Search
            query_results = pinecone_index.query(
                vector=query_vector,
                top_k=8,
                include_metadata=True
            )
            
            context_text = ""
            sources_text = ""
            seen_sources = set()
            
            if "matches" in query_results and query_results["matches"]:
                for match in query_results["matches"]:
                    meta = match.get("metadata", {})
                    text_content = meta.get("text", meta.get("_node_content", ""))
                    context_text += f"{text_content}\n\n"
                    
                    raw_file_name = meta.get("file_name", "Unknown Document")
                    page_num = meta.get("page_label", meta.get("page_number", "1"))
                    clean_title = raw_file_name.replace("+", " ").replace("_", " ").replace(".pdf", "")
                    
                    source_id = f"{raw_file_name}_{page_num}"
                    if source_id not in seen_sources and len(seen_sources) < 2:
                        seen_sources.add(source_id)
                        sources_text += (
                            f"• **Research Paper / Document:** `{raw_file_name}`  \n"
                            f"  📌 **Extracted Title:** {clean_title}  \n"
                            f"  📄 **Cited Page:** {page_num}\n\n"
                        )
            
            # Groq Llama-3 Call
            system_instruction = (
                "You are ForestBot.SJP, an expert AI assistant for Sri Lankan Forestry & Environmental Sciences. "
                "Base your answer strictly on the provided internal context. "
                "Provide exact legal sections, fines (LKR), and academic details if available."
            )
            
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Context Information:\n{context_text}\n\nUser Question: {prompt}\nAnswer:"}
                ],
                model="openai/gpt-oss-120b",
                temperature=0.1
            )
            
            answer = chat_completion.choices[0].message.content
            st.markdown(answer)
            
            if sources_text:
                st.markdown("---")
                st.markdown("##### 📚 **Sources & Exact Citations:**")
                st.markdown(sources_text)
                    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": sources_text
    })
    st.rerun()
