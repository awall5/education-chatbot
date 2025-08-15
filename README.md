# 🎓 Customer Support RAG (Education) with Sentiment & Escalation

A Render-deployable **RAG** app for the **education** domain with:
- Retrieval over a small knowledge base using **ChromaDB**
- **OpenAI** embeddings & generation (RAG with context grounding)
- Real-time **sentiment analysis** (VADER) + simple **mood detection**
- **Escalation prediction** (consecutive negatives / strong negativity / high-priority + negative)
- **Streamlit** UI

---

## 🧱 Architecture

```
User → Sentiment/VADER → ChromaDB (OpenAI embeddings) → Context
   → OpenAI Chat Completion (empathetic, context-grounded) → Response + Citation tags
   → Escalation check + Metrics
```

- Vector DB: **Chroma** (persistent at `./chroma_db`)
- Embeddings: `text-embedding-3-small`
- LLM: `gpt-4o-mini` (default)
- UI: **Streamlit**

---

## 🚀 Quickstart (Local)

1) Python 3.10+ recommended.
2) Install:
```bash
pip install -r requirements.txt
```
3) Set your API key:
```bash
export OPENAI_API_KEY=sk-...
```
4) Run:
```bash
streamlit run app.py
```

---

## ☁️ Deploy to Render

1. Create a **new Web Service** on Render
2. Repository Root: this project
3. **Runtime**: Python 3.10+
4. Build Command: 
   ```
   pip install -r requirements.txt
   ```
5. Start Command (auto from `Procfile`) or:
   ```
   streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
6. Add **Environment Variable**:
   - `OPENAI_API_KEY` = your key
   - (optional) `CHROMA_DIR` = `./chroma_db`
   - (optional) `KB_PATH` = `data/knowledge_base.json`

7. Deploy. First boot will auto-index the KB.

---

## 🧪 Features & Evaluation

- **Top-K** retrieval: adjustable
- Shows latency, token usage, average retrieval distance
- Basic evaluation hooks can be added to compute retrieval accuracy against a labeled set (e.g., RAGAS).

---

## 📁 Project Structure

```
.
├── app.py                 # Streamlit UI + orchestration
├── rag_pipeline.py        # Chroma retrieval + OpenAI generation
├── sentiment.py           # VADER sentiment + mood tagging
├── data/
│   └── knowledge_base.json
├── requirements.txt
├── Procfile
└── README.md
```

---

## 🔐 Notes

- This app uses **OpenAI** for embeddings and generation. Costs are minimal for demo scale.
- If you prefer **fully offline** embeddings: swap in `sentence-transformers` and pass a custom embedding function into Chroma (keep in mind build size on Render).

---

## 🧭 Roadmap (Optional Extensions)

- Multi-turn memory grounding per user
- RAGAS evaluation report page
- Richer emotion detection (Ekman categories) with a lightweight classifier
- Admin console to upload new KB docs and reindex live
