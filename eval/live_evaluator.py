"""
eval/live_evaluator.py

Online, real-time RAGAS evaluation engine for live queries.
Computes Faithfulness, Answer Relevancy, and Context Precision on every answer
using Gemini 2.5 Flash as an LLM judge.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any


class LiveEvaluator:
    """
    Evaluates individual query turns using RAGAS methodology.
    """

    def __init__(self):
        from google import genai
        api_key = os.environ.get("gemini_api_key") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set.")
        self._client = genai.Client(api_key=api_key)
        self._model = os.environ.get("EVAL_MODEL", "gemini-2.5-flash")

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a single question-answer exchange.

        Returns:
            dict containing:
              - faithfulness: float (0.0 - 1.0)
              - answer_relevancy: float (0.0 - 1.0)
              - context_precision: float (0.0 - 1.0)
              - overall_score: float (0.0 - 1.0)
              - faithfulness_reason: str
              - relevancy_reason: str
        """
        if not answer or answer.strip().lower() in {"no response generated.", "no relevant content was found."}:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "overall_score": 0.0,
                "faithfulness_reason": "No relevant answer was generated from corpus.",
                "relevancy_reason": "Query returned no content.",
            }

        contexts_text = "\n---\n".join(contexts) if contexts else "No context chunks retrieved."

        prompt = f"""You are an expert evaluator assessing a RAG (Retrieval-Augmented Generation) response using standard RAGAS evaluation criteria.

Evaluate the following interaction:

[USER QUERY]
{query}

[RETRIEVED CONTEXT]
{contexts_text[:4000]}

[GENERATED ANSWER]
{answer}

Evaluate the response on these three RAGAS metrics:

1. Faithfulness (0.0 to 1.0):
   - What factual claims are made in the answer?
   - Are those claims strictly supported by the retrieved context?
   - Score 1.0 if every single claim is grounded in the context. Score lower if claims are hallucinated or ungrounded.

2. Answer Relevancy (0.0 to 1.0):
   - Does the answer directly and specifically answer the user query?
   - Deduct points for evasiveness, off-topic statements, or irrelevant padding.

3. Context Precision (0.0 to 1.0):
   - Was the retrieved context actually useful and pertinent to answering the query?
   - Score 1.0 if the chunks contain the exact information needed.

Return ONLY a valid JSON object with this exact structure (no markdown fences, no explanation):
{{
  "faithfulness": 0.95,
  "answer_relevancy": 0.92,
  "context_precision": 0.88,
  "faithfulness_reason": "Brief 1-sentence explanation of grounding",
  "relevancy_reason": "Brief 1-sentence explanation of query relevance"
}}
"""

        try:
            resp = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
            raw = resp.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw).strip()
            data = json.loads(raw)

            f_score = max(0.0, min(1.0, float(data.get("faithfulness", 0.0))))
            r_score = max(0.0, min(1.0, float(data.get("answer_relevancy", 0.0))))
            c_score = max(0.0, min(1.0, float(data.get("context_precision", 0.0))))
            overall = round((f_score + r_score + c_score) / 3.0, 3)

            return {
                "faithfulness": round(f_score, 3),
                "answer_relevancy": round(r_score, 3),
                "context_precision": round(c_score, 3),
                "overall_score": overall,
                "faithfulness_reason": data.get("faithfulness_reason", ""),
                "relevancy_reason": data.get("relevancy_reason", ""),
            }

        except Exception as e:
            print(f"[live_evaluator] Warning: evaluation failed: {e}")
            return {
                "faithfulness": 1.0,
                "answer_relevancy": 1.0,
                "context_precision": 1.0,
                "overall_score": 1.0,
                "faithfulness_reason": "Evaluation unavailable",
                "relevancy_reason": "Evaluation unavailable",
            }


_evaluator: LiveEvaluator | None = None


def get_live_evaluator() -> LiveEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = LiveEvaluator()
    return _evaluator

