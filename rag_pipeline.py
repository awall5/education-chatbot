import os
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from transformers import pipeline
from sentence_transformers import SentenceTransformer

@dataclass
class RAGDoc:
    id: str
    title: str
    text: str
    url: str = ""

class EduRAG:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "edu_kb", kb_path: str = "data/knowledge_base.json"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.kb_path = kb_path

        # Local embeddings
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Local text generation model (Q&A style)
        self.qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

        # Chroma setup
        self.client = chromadb.PersistentClient(path=self.persist_dir, settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embed_text
        )
        self._ensure_index_built()

    def _embed_text(self, texts: List[str]):
        if isinstance(texts, str):
            texts = [texts]
        return self.embed_model.encode(texts).tolist()

    def _ensure_index_built(self):
        if self.count() == 0:
            self._load_kb()

    def rebuild(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embed_text
        )
        self._load_kb()

    def count(self) -> int:
        try:
            return len(self.collection.get()["ids"])
        except Exception:
            return 0

    def _load_kb(self):
        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids, docs, metas = [], [], []
        for i, d in enumerate(data):
            ids.append(str(i))
            docs.append(d["text"])
            metas.append({"title": d["title"], "url": d.get("url", "")})
        self.collection.add(ids=ids, documents=docs, metadatas=metas)

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        result = self.collection.query(query_texts=[query], n_results=k)
        out = []
        for i in range(len(result["ids"][0])):
            out.append({
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "title": result["metadatas"][0][i].get("title", f"Doc {i+1}"),
                "url": result["metadatas"][0][i].get("url", ""),
                "score": result.get("distances", [[None]*k])[0][i] if "distances" in result else None
            })
        return out

    def generate_empathetic_answer(
        self,
        user_query: str,
        contexts: List[Dict],
        sentiment: Dict,
        temperature: float = 0.3,
        student_name: str = "",
        domain: str = "education",
        escalate: bool = False,
    ) -> Tuple[str, Dict, Dict]:
        system_prompt = (
            "You are an empathetic, helpful customer support assistant for the education domain. "
            "Always base answers on the provided context. If context is missing, say so briefly."
        )
        context_block = "\n\n".join([f"[{i+1}] {c['title']}\n{c['text']}" for i, c in enumerate(contexts)]) or "No relevant documents found."

        style = f"Sentiment: {sentiment.get('label')} | Mood: {sentiment.get('mood')}."
        escalation_note = "This query may need escalation to a specialist. " if escalate else ""
        name_line = f"Student name: {student_name}." if student_name else ""

        final_prompt = (
            f"{system_prompt}\n{name_line}\n{style}\n{escalation_note}\n"
            f"Question: {user_query}\n\nContext:\n{context_block}"
        )

        result = self.qa_pipeline(final_prompt, max_new_tokens=256, temperature=temperature)
        answer = result[0]['generated_text']

        usage = {"model": "flan-t5-base (local)"}
        scores = {
            "retrieved_k": len(contexts),
            "avg_distance": (sum([c["score"] for c in contexts if isinstance(c.get("score"), (int,float))]) /
                             max(1, sum([1 for c in contexts if isinstance(c.get("score"), (int,float))]))) if contexts else None
        }
        return answer, usage, scores
