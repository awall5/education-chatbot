import os
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from openai import OpenAI

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
        self.client = chromadb.PersistentClient(path=self.persist_dir, settings=Settings(allow_reset=True))
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_fn
            )
        except Exception:
            # If embedding function can't be bound (missing key), still create collection without EF.
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        self._ensure_index_built()

        self.oai = OpenAI()

    def _ensure_index_built(self):
        # If collection is empty, load KB
        count = self.count()
        if count == 0:
            self._load_kb()

    def rebuild(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn
        )
        self._load_kb()

    def count(self) -> int:
        try:
            # Chroma doesn't expose a direct count; run a dummy query
            res = self.collection.get(limit=1)
            total = len(self.collection.get()["ids"])
            return total
        except Exception:
            return 0

    def _load_kb(self):
        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids, docs, metas = [], [], []
        for i, d in enumerate(data):
            ids.append(str(i))
            docs.append(d["text"])
            metas.append({"title": d["title"], "url": d.get("url","")})
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

    def _build_system_prompt(self, domain: str = "education") -> str:
        return (
            "You are an empathetic, factual, and helpful customer support assistant for the education domain "
            "(universities, online learning, enrollment, exams, scholarships, student life). "
            "You must ground each answer in the provided context snippets. "
            "If context is missing, say so briefly and offer general guidance. "
            "Use simple, supportive language and, when appropriate, step-by-step suggestions."
        )

    def generate_empathetic_answer(
        self,
        user_query: str,
        contexts: List[Dict],
        sentiment: Dict,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        student_name: str = "",
        domain: str = "education",
        escalate: bool = False,
    ) -> Tuple[str, Dict, Dict]:
        system_prompt = self._build_system_prompt(domain)
        context_block = "\n\n".join([f"[{i+1}] {c['title']}\n{c['text']}" for i, c in enumerate(contexts)]) or "No relevant documents found."

        style = f"Sentiment: {sentiment.get('label')} (compound={sentiment.get('compound'):.2f}); Mood: {sentiment.get('mood')}."
        tone = "Use an empathetic and reassuring tone. Acknowledge feelings briefly, then give clear, actionable steps."
        escalation_note = "If escalation is suggested, include a short note: 'I've flagged this for a specialist to follow up.' " if escalate else ""

        name_line = f"Student name: {student_name}." if student_name else ""

        messages = [
            {"role":"system","content": system_prompt},
            {"role":"user","content": (
                f"{name_line}\n"
                f"{style}\n"
                f"{tone}\n"
                f"{escalation_note}\n"
                "User question:\n"
                f"{user_query}\n\n"
                "Use ONLY the context if possible. If something isn't in context, say so. "
                "Cite context snippets like [1], [2] when used.\n\n"
                f"Context Snippets:\n{context_block}"
            )}
        ]

        resp = self.oai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        text = resp.choices[0].message.content
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if hasattr(resp, "usage") else None,
            "completion_tokens": resp.usage.completion_tokens if hasattr(resp, "usage") else None,
            "total_tokens": resp.usage.total_tokens if hasattr(resp, "usage") else None,
            "model": model
        }
        scores = {
            "retrieved_k": len(contexts),
            "avg_distance": (sum([c["score"] for c in contexts if isinstance(c.get("score"), (int,float))]) / max(1, sum([1 for c in contexts if isinstance(c.get('score'), (int,float))]))) if contexts else None
        }
        return text, usage, scores
