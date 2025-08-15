import os
import time
import json
import streamlit as st

from rag_pipeline import EduRAG
from sentiment import analyze_sentiment_and_mood

st.set_page_config(page_title="Edu Support RAG + Sentiment", page_icon="🎓", layout="wide")

st.title("🎓 Customer Support RAG (Education) with Sentiment & Escalation")
st.caption("Retrieval-Augmented Generation • Real-time Sentiment • Empathetic Responses • Escalation Prediction")

# Initialize RAG
@st.cache_resource(show_spinner=True)
def _load_rag():
    return EduRAG(
        persist_dir=os.getenv("CHROMA_DIR", "./chroma_db"),
        collection_name="edu_kb",
        kb_path=os.getenv("KB_PATH", "data/knowledge_base.json"),
    )

rag = _load_rag()

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"role": "user"/"assistant", "text": str, "sentiment": {...}}
if "neg_streak" not in st.session_state:
    st.session_state.neg_streak = 0

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K documents", min_value=1, max_value=6, value=3)
    temperature = st.slider("Temperature", 0.0, 1.2, 0.3, 0.1)
    st.divider()
    if st.button("➕ Rebuild Index"):
        rag.rebuild()
        st.success("Index rebuilt from knowledge_base.json")
    st.divider()
    st.caption("📌 Tip: Adjust Top-K and Temperature to tune response quality & creativity.")

st.subheader("Ask a question")
default_q = "How can I prepare for my semester exams while balancing part-time work?"
query = st.text_input("Your question (education domain):", value=default_q)

col1, col2 = st.columns([1, 1])
with col1:
    user_name = st.text_input("Student name (optional):", value="Awal")
with col2:
    priority = st.selectbox("Ticket priority (optional):", ["Low", "Normal", "High"], index=1)

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    start = time.time()
    # 1) Sentiment & mood
    sent = analyze_sentiment_and_mood(query)
    compound = sent["compound"]
    label = sent["label"]

    # Update negative streak
    if label == "negative":
        st.session_state.neg_streak += 1
    else:
        st.session_state.neg_streak = 0

    should_escalate = (
        st.session_state.neg_streak >= 2
        or compound <= -0.6
        or (priority == "High" and label == "negative")
    )

    # 2) Retrieve
    contexts = rag.retrieve(query, k=top_k)

    # 3) Generate (no model arg — uses local flan-t5-base)
    answer, usage, scores = rag.generate_empathetic_answer(
        user_query=query,
        contexts=contexts,
        sentiment=sent,
        temperature=temperature,
        student_name=user_name,
        domain="education",
        escalate=should_escalate,
    )
    latency = time.time() - start

    # Save history
    st.session_state.history.append({"role": "user", "text": query, "sentiment": sent})
    st.session_state.history.append({
        "role": "assistant",
        "text": answer,
        "meta": {"latency": latency, "usage": usage, "scores": scores, "escalate": should_escalate}
    })

# --- Conversation display ---
st.divider()
st.subheader("Conversation")
for turn in st.session_state.history:
    if turn["role"] == "user":
        st.markdown(f"**🧑 Student:** {turn['text']}")
        s = turn.get("sentiment", {})
        if s:
            st.caption(f"Sentiment: {s.get('label','?')} | Compound: {s.get('compound',0):.2f} | Mood: {s.get('mood','-')}")
    else:
        st.markdown(f"**🤖 Assistant:** {turn['text']}")
        meta = turn.get("meta", {})
        if meta:
            with st.expander("Details"):
                st.write({
                    "latency_s": round(meta.get("latency", 0), 2),
                    **meta.get("usage", {}),
                    **meta.get("scores", {}),
                    "escalate": meta.get("escalate", False)
                })

# --- Diagnostic panel ---
st.divider()
st.subheader("Diagnostics")
colA, colB, colC = st.columns(3)
with colA:
    st.metric("Negative Streak", st.session_state.neg_streak)
with colB:
    st.metric("KB Documents", rag.count())
with colC:
    st.metric("Index Path", os.getenv("CHROMA_DIR", "./chroma_db"))
st.caption("Escalation triggers: 2× consecutive negatives, very negative tone (≤ −0.6), or High priority + negative.")
