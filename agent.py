import json
from typing import Dict, List, Optional

from config import (
    OPENAI_API_KEY,
    MODEL_NAME,
    build_openai_client,
)


class AIIntentParser:
    """LLM-based intent parser for research queries."""

    def __init__(self):
        self.model = MODEL_NAME
        self.api_key = OPENAI_API_KEY
        self.client = build_openai_client()

    def parse(self, query: str) -> Dict[str, str]:
        """Parse user query into structured info using an LLM."""
        fallback = {"keyword": query.strip(), "explanation": "fallback"}
        if not query:
            return fallback

        if not self.client:
            return fallback

        prompt = (
            "You are a search co-pilot for academic papers. "
            "Rewrite the user query into a concise search keyword or phrase. "
            "Return JSON with keys: keyword, explanation. "
            "Do not choose or suggest a sorting mode; sorting is controlled by the user interface. "
            "The explanation should briefly state how the keyword preserves the user's research intent."
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
            )
            content = completion.choices[0].message.content or ""
            parsed = json.loads(content)
            keyword = str(parsed.get("keyword", "")).strip() or query.strip()
            explanation = parsed.get("explanation", "")
            return {"keyword": keyword, "explanation": explanation}
        except Exception:
            return fallback


class RAGResearchAssistant:
    """RAG assistant that summarizes search results plus KG expansions."""

    def __init__(self):
        self.model = MODEL_NAME
        self.api_key = OPENAI_API_KEY
        self.client = build_openai_client()

        self.system_prompt = (
            "You are a research co-pilot that fuses knowledge-graph expansions and bibliographic search results. "
            "Provide concise, evidence-grounded answers for researchers. "
            "Cite titles when using them, and prefer actionable next steps."
        )

    @staticmethod
    def _build_context(
        user_query: str,
        keyword: str,
        expansion_terms: List[str],
        relevant: List[Dict],
        others: List[Dict],
        top_n: int = 6,
    ) -> str:
        lines = [f"User query: {user_query}", f"Parsed keyword: {keyword}"]
        if expansion_terms:
            lines.append("KG expansion terms: " + ", ".join(expansion_terms))

        def _format_paper(paper: Dict, idx: int) -> str:
            title = paper.get("title", "Untitled")
            cites = paper.get("cited_by_count", 0) or 0
            year = paper.get("publication_year", "N/A")
            link = paper.get("link", "")
            overlap = paper.get("overlapping_keywords", [])
            overlap_str = ", ".join(overlap) if overlap else ""
            return (
                f"{idx}. {title} (cites: {cites}, year: {year})\n"
                f"link: {link}\n"
                f"overlap: {overlap_str}\n"
            )

        if relevant:
            lines.append("Top graph-overlap papers:")
            for i, paper in enumerate(relevant[:top_n], 1):
                lines.append(_format_paper(paper, i))

        remaining_slots = max(0, top_n - len(relevant[:top_n]))
        if others and remaining_slots > 0:
            lines.append("Additional papers:")
            for j, paper in enumerate(others[:remaining_slots], 1):
                lines.append(_format_paper(paper, j))

        return "\n".join(lines)

    def answer(
        self,
        user_query: str,
        keyword: str,
        expansion_terms: List[str],
        relevant: List[Dict],
        others: List[Dict],
        language: Optional[str] = None,
    ) -> str:
        if not user_query:
            return "请先输入问题。"

        if not self.client:
            return "未检测到 OPENAI_API_KEY，已跳过生成。"

        context = self._build_context(user_query, keyword, expansion_terms, relevant, others)

        # Heuristic: detect language from user input first char (very rough)
        lang_hint = "中文" if language and ord(language[0]) > 127 else "English"

        user_prompt = (
            f"Reply in {lang_hint}. "
            "Use the provided context (from KG expansions + search results) to answer the user's research question. "
            "If evidence is thin, clearly say so and suggest 2-3 next steps (e.g., try other keywords, inspect overlaps). "
            "Keep it concise (<= 200 words). "
            "Prefer bullet points."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}\n{user_prompt}"},
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                max_tokens=500,
                messages=messages,
            )
            return completion.choices[0].message.content or "(无输出)"
        except Exception as exc:
            return f"生成回答出错: {exc}"
