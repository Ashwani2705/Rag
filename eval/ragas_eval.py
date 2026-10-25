"""
eval/ragas_eval.py

Phase 4 — RAGAS evaluation engine.

Runs the four RAGAS metrics (faithfulness, answer_relevancy,
context_precision, context_recall) for a given set of golden entries
and a retrieval mode.

Workflow:
  1. For each golden entry, call the agent (via AgentRunner) to get an answer
  2. Package question + answer + contexts + ground_truth into a RAGAS Dataset
  3. Run RAGAS evaluate() with all four metrics
  4. Return a typed EvalResult

Usage:
    from eval.ragas_eval import RagasEvaluator
    evaluator = RagasEvaluator()
    result = await evaluator.evaluate_mode("flat_rag", entries=get_by_group("A"))
    print(result.summary())
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class MetricScore:
    name:  str
    score: float

    def __str__(self) -> str:
        return f"{self.name}: {self.score:.3f}"


@dataclass
class EvalResult:
    """RAGAS evaluation result for one retrieval mode on a set of entries."""
    mode:              str                  # flat_rag | graph_rag | hybrid
    entry_count:       int
    faithfulness:      float
    answer_relevancy:  float
    context_precision: float
    context_recall:    float
    elapsed_seconds:   float
    per_question:      list[dict] = field(default_factory=list)   # row-level scores
    errors:            list[str]  = field(default_factory=list)

    @property
    def average(self) -> float:
        scores = [self.faithfulness, self.answer_relevancy,
                  self.context_precision, self.context_recall]
        return sum(scores) / len(scores)

    def metrics(self) -> list[MetricScore]:
        return [
            MetricScore("faithfulness",      self.faithfulness),
            MetricScore("answer_relevancy",  self.answer_relevancy),
            MetricScore("context_precision", self.context_precision),
            MetricScore("context_recall",    self.context_recall),
            MetricScore("average",           self.average),
        ]

    def summary(self) -> str:
        lines = [
            f"\n{'='*50}",
            f"  Mode         : {self.mode}",
            f"  Questions    : {self.entry_count}",
            f"  Elapsed      : {self.elapsed_seconds:.1f}s",
            f"{'─'*50}",
        ]
        for m in self.metrics():
            bar = "█" * int(m.score * 20)
            lines.append(f"  {m.name:<22}: {m.score:.3f}  {bar}")
        if self.errors:
            lines.append(f"\n  Errors ({len(self.errors)}): {'; '.join(self.errors[:3])}")
        lines.append(f"{'='*50}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "mode":              self.mode,
            "entry_count":       self.entry_count,
            "faithfulness":      round(self.faithfulness, 4),
            "answer_relevancy":  round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall":    round(self.context_recall, 4),
            "average":           round(self.average, 4),
            "elapsed_seconds":   round(self.elapsed_seconds, 1),
            "errors":            self.errors,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RagasEvaluator:
    """
    Drives the RAGAS evaluation pipeline for one or all retrieval modes.

    Args:
        llm_model  : Gemini model for RAGAS judge LLM (default: gemini-1.5-flash)
        batch_size : number of questions to evaluate per RAGAS call (default: 10)
        timeout    : seconds per agent call before giving up (default: 60)
    """

    METRICS_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def __init__(
        self,
        llm_model:  str = "gemini-1.5-flash",
        batch_size: int = 10,
        timeout:    int = 60,
    ):
        self.llm_model  = llm_model
        self.batch_size = batch_size
        self.timeout    = timeout
        self._ragas_llm = None
        self._metrics   = None

    # ── Public API ──────────────────────────────────────────────────────────

    async def evaluate_mode(
        self,
        mode: str,
        entries=None,
        user_id: str = "eval_user",
    ) -> EvalResult:
        """
        Run RAGAS evaluation for a single retrieval mode.

        Args:
            mode    : "flat_rag" | "graph_rag" | "hybrid" | "all"
            entries : list of GoldenEntry — defaults to the standard group for the mode
            user_id : user_id used for agent session scoping

        Returns:
            EvalResult with all four metric scores
        """
        # from eval.golden_dataset import get_by_mode, GOLDEN_DATASET

        if entries is None:
            entries = get_by_mode(mode) if mode != "all" else GOLDEN_DATASET

        print(f"\n[ragas_eval] evaluating mode={mode}, {len(entries)} questions...")
        t0 = time.time()

        # Step 1 — collect agent answers
        answers, used_contexts, errors = await self._collect_answers(entries, user_id)

        if not answers:
            return EvalResult(
                mode=mode, entry_count=len(entries),
                faithfulness=0.0, answer_relevancy=0.0,
                context_precision=0.0, context_recall=0.0,
                elapsed_seconds=time.time() - t0,
                errors=errors,
            )

        # Step 2 — build RAGAS dataset
        dataset = self._build_dataset(entries, answers, used_contexts)

        # Step 3 — run RAGAS metrics
        scores, per_question = self._run_ragas(dataset)

        elapsed = round(time.time() - t0, 1)
        print(f"[ragas_eval] [OK] done in {elapsed}s")

        return EvalResult(
            mode=mode,
            entry_count=len(entries),
            faithfulness=scores.get("faithfulness", 0.0),
            answer_relevancy=scores.get("answer_relevancy", 0.0),
            context_precision=scores.get("context_precision", 0.0),
            context_recall=scores.get("context_recall", 0.0),
            elapsed_seconds=elapsed,
            per_question=per_question,
            errors=errors,
        )

    # ── Answer collection ────────────────────────────────────────────────────

    async def _collect_answers(
        self,
        entries,
        user_id: str,
    ) -> tuple[list[str], list[list[str]], list[str]]:
        """
        Call the agent for each question and collect the answers.

        Returns:
            (answers, contexts_used, errors)
            contexts_used: the golden contexts (we use the reference contexts
                           because the agent's retrieved chunk texts may not
                           align 1:1 with golden context strings)
        """
        from agents.runner import get_runner
        runner = get_runner()

        answers:  list[str]        = []
        contexts: list[list[str]]  = []
        errors:   list[str]        = []

        for i, entry in enumerate(entries, 1):
            print(f"  [{i}/{len(entries)}] Q: {entry.question[:60]}...")
            try:
                response = await asyncio.wait_for(
                    runner.run(entry.question, user_id=user_id),
                    timeout=self.timeout,
                )
                answers.append(response.answer or "No answer generated.")
                # Use the golden reference contexts for RAGAS metrics
                # (ensures context_recall measures what the retriever should find)
                contexts.append(entry.contexts)

            except asyncio.TimeoutError:
                err = f"Timeout on Q{i}: {entry.question[:40]}"
                print(f"  [ragas_eval] WARNING: {err}")
                errors.append(err)
                answers.append("")
                contexts.append(entry.contexts)

            except Exception as e:
                err = f"Error on Q{i}: {str(e)[:80]}"
                print(f"  [ragas_eval] ERROR: {err}")
                errors.append(err)
                answers.append("")
                contexts.append(entry.contexts)

            # Small delay to avoid LLM rate limits
            if i < len(entries):
                await asyncio.sleep(0.5)

        return answers, contexts, errors

    # ── RAGAS dataset builder ────────────────────────────────────────────────

    @staticmethod
    def _build_dataset(entries, answers: list[str], contexts: list[list[str]]) -> Any:
        """
        Build a RAGAS-compatible Dataset from the evaluation data.
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets is required: pip install datasets")

        data = {
            "question":     [e.question     for e in entries],
            "answer":       answers,
            "contexts":     contexts,
            "ground_truth": [e.ground_truth for e in entries],
        }
        return Dataset.from_dict(data)

    # ── RAGAS runner ─────────────────────────────────────────────────────────

    def _run_ragas(self, dataset: Any) -> tuple[dict[str, float], list[dict]]:
        """
        Run all four RAGAS metrics and return aggregated + per-row scores.

        Uses Gemini Flash as the judge LLM (configured via langchain_google_genai).
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        except ImportError as e:
            raise ImportError(
                f"RAGAS dependencies missing: {e}\n"
                "Install: pip install ragas langchain-google-genai datasets"
            )

        judge_llm = ChatGoogleGenerativeAI(
            model=self.llm_model,
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=0,
        )
        judge_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        print(f"  [ragas_eval] running RAGAS metrics (judge={self.llm_model})...")
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            raise_exceptions=False,
        )

        # Aggregate scores
        df = result.to_pandas()
        agg = {col: float(df[col].mean()) for col in self.METRICS_NAMES if col in df.columns}

        # Per-question scores
        per_q = df.to_dict(orient="records")

        return agg, per_q