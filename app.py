import json
from urllib.parse import unquote_plus

import streamlit as st
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------
# 0. Settings (change these here if needed)
# ---------------------------------------------------------------
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
INDEX_NAME = "forest-bot-index"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # must match the model used to build the index
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 8                    # how many chunks to fetch from Pinecone
MAX_CONTEXT_CHARS = 9000     # keeps the request small enough for Groq's free limits
MAX_SOURCES_SHOWN = 3
DEFAULT_MIN_SCORE = 0.0      # your scores are around 0.7 for good matches

NO_INFO_MESSAGE = (
    "ඔබේ ප්‍රශ්නයට අදාළ තොරතුරු දැනට ForestBot.SJP හි ලේඛන ගබඩාවේ සොයාගත නොහැකි විය.\n\n"
    "*The indexed documents do not contain enough information to answer this question.*"
)
EMPTY_REPLY_MESSAGE = (
    "⚠️ The language model returned an empty reply. Please ask again or rephrase the question."
)

SYSTEM_INSTRUCTION = (
    "You are ForestBot.SJP, an academic assistant for Sri Lankan forestry, "
    "environmental law and environmental science.\n"
    "Rules:\n"
    "1. Use ONLY the facts in the numbered context sources. Do not add outside "
    "knowledge, and never guess section numbers, fines or figures.\n"
    "2. When you state a fact, cite its source number like [1].\n"
    "3. If the context only partly answers the question, answer the supported part "
    "and clearly say what is not covered.\n"
    "4. If the question is only a topic or keywords (for example 'life cycle assessment'), "
    "treat it as 'What do the sources say about this topic?' and summarise what the "
    "context says about it.\n"
    "5. Reply with exactly NO_INFO only if the context contains nothing at all related "
    "to the question.\n"
    "6. Reply in the same language as the question. Keep act names, section numbers "
    "and amounts exactly as written in the context."
)

# ---------------------------------------------------------------
# 1. Page configuration + session state
# ---------------------------------------------------------------
st.set_page_config(page_title="ForestBot.SJP", page_icon="🌲", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------
# 2. Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.title("🌲 ForestBot.SJP")
    st.markdown("**AI-Powered Forestry Law & Environmental Science Assistant**")
    st.markdown("---")

    if st.button("➕ Start New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("📜 Chat History")
    user_queries = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if user_queries:
        for idx, q in enumerate(reversed(user_queries), 1):
            short_q = q[:30] + "..." if len(q) > 30 else q
            st.caption(f"{idx}. {short_q}")
    else:
        st.info("No chat history in this session yet.")

    st.markdown("---")
    with st.expander("⚙️ Advanced (for testing)"):
        show_debug = st.checkbox("Show retrieval details", value=True)
        min_score = st.slider(
            "Minimum relevance score", 0.0, 0.9, DEFAULT_MIN_SCORE, 0.05,
            help="Chunks scoring below this are ignored. Keep it at 0 unless "
                 "off-topic chunks are being used.",
        )
    st.caption("University of Sri Jayewardenepura | Academic Research Project")

# ---------------------------------------------------------------
# 3. Title
# ---------------------------------------------------------------
st.title("🌲 ForestBot.SJP")
st.subheader("Interactive Knowledge Assistant for Sri Lankan Forestry & Environmental Sciences")
st.markdown("---")


# ---------------------------------------------------------------
# 4. Engines (loaded once)
# ---------------------------------------------------------------
@st.cache_resource
def setup_engines():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    groq_client = Groq(api_key=GROQ_API_KEY)
    return embed_model, index, groq_client


with st.spinner("Connecting Pinecone Cloud Vector Store and Groq Engine..."):
    embed_model, pinecone_index, groq_client = setup_engines()

st.caption(f"Connected to Pinecone index `{INDEX_NAME}` | Groq model: `{GROQ_MODEL}`")


# ---------------------------------------------------------------
# 5. Helper functions
# ---------------------------------------------------------------
def get_chunk_text(meta):
    """Get the real text of a chunk from Pinecone metadata (LlamaIndex stores it in _node_content)."""
    text = meta.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    raw = meta.get("_node_content")
    if raw:
        try:
            return json.loads(raw).get("text", "").strip()
        except Exception:
            return ""
    return ""


def clean_name(file_name):
    name = unquote_plus(str(file_name))
    for ext in (".pdf", ".docx", ".txt"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
    return name.replace("_", " ").strip()


def build_llm_history(history, max_pairs=2):
    """Earlier question/answer pairs, for follow-up questions.
    Only exchanges that got a REAL answer are kept. 'No information' replies are
    never sent back, because the model would copy them again and again."""
    pairs = []
    for i in range(len(history) - 1):
        a, b = history[i], history[i + 1]
        if a["role"] == "user" and b["role"] == "assistant" and b.get("ok"):
            pairs.append((a["content"], b["content"]))
    msgs = []
    for question, answer in pairs[-max_pairs:]:
        msgs.append({"role": "user", "content": question})
        msgs.append({"role": "assistant", "content": answer})
    return msgs


# ---------------------------------------------------------------
# 6. Show previous messages
# ---------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.markdown("---")
            st.markdown("##### 📚 **Sources & References Cited:**")
            st.markdown(message["sources"])
        if show_debug and message.get("debug"):
            with st.expander("🔎 Retrieval details"):
                st.table(message["debug"])
                if message.get("debug_note"):
                    st.text(message["debug_note"])

# ---------------------------------------------------------------
# 7. Handle a new question
# ---------------------------------------------------------------
if prompt := st.chat_input("Ask ForestBot.SJP any forestry or environmental law question..."):
    history_before = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and preparing the answer..."):
            # --- Search Pinecone ---
            query_vector = embed_model.encode(prompt).tolist()
            results = pinecone_index.query(
                vector=query_vector, top_k=TOP_K, include_metadata=True
            )

            context_blocks = []
            used_sources = []        # (file, page) in score order, no repeats
            debug_rows = []
            seen_texts = set()
            total_chars = 0

            for match in results.get("matches", []):
                meta = match.get("metadata", {}) or {}
                score = float(match.get("score", 0))
                file_name = clean_name(meta.get("file_name", "Unknown document"))
                page = meta.get("page_label", meta.get("page_number", "-"))
                text = get_chunk_text(meta)

                kept = False
                if text and score >= min_score and text not in seen_texts:
                    if total_chars + len(text) <= MAX_CONTEXT_CHARS:
                        seen_texts.add(text)
                        total_chars += len(text)
                        context_blocks.append(
                            f"[Source {len(context_blocks) + 1}] ({file_name}, page {page})\n{text}"
                        )
                        if (file_name, page) not in used_sources:
                            used_sources.append((file_name, page))
                        kept = True

                debug_rows.append({
                    "score": round(score, 3),
                    "document": file_name[:60],
                    "page": str(page),
                    "used": "yes" if kept else "no",
                })

            # --- Ask Groq (only if we found something) ---
            answer = NO_INFO_MESSAGE
            sources_text = ""
            answer_ok = False
            notes = [f"chunks sent to the model: {len(context_blocks)} ({total_chars} characters)"]

            if context_blocks:
                notes.append("first chunk starts: " + context_blocks[0][:300].replace("\n", " "))
                context = "\n\n---\n\n".join(context_blocks)
                messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
                messages += build_llm_history(history_before)
                messages.append({
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:",
                })
                try:
                    completion = groq_client.chat.completions.create(
                        messages=messages,
                        model=GROQ_MODEL,
                        temperature=0.1,
                    )
                    choice = completion.choices[0]
                    reply = (choice.message.content or "").strip()
                    notes.append(f"model finish reason: {choice.finish_reason}")
                    notes.append("model reply starts: " + (reply[:80] or "(empty)"))
                    if not reply:
                        answer = EMPTY_REPLY_MESSAGE
                    elif not reply.startswith("NO_INFO"):
                        answer = reply
                        answer_ok = True
                        for file_name, page in used_sources[:MAX_SOURCES_SHOWN]:
                            sources_text += f"• **{file_name}**  \n"
                            if str(page) not in ("-", ""):
                                sources_text += f"  📄 Page: {page}  \n"
                            sources_text += "\n"
                except Exception as e:
                    answer = (
                        "⚠️ The language model could not answer right now "
                        "(possibly a rate limit). Please wait a minute and try again.\n\n"
                        f"`{type(e).__name__}: {str(e)[:200]}`"
                    )
                    notes.append(f"error: {type(e).__name__}")
            else:
                notes.append("no chunk passed the filter, so the model was not asked")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources_text,
        "debug": debug_rows,
        "debug_note": "\n".join(notes),
        "ok": answer_ok,
    })
    st.rerun()
