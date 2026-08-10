"""
eval/compare_modes.py

Phase 4 — Cross-mode comparison runner.

Runs RAGAS evaluation for all three retrieval modes and produces:
  - Side-by-side metric table
  - Per-group winner analysis (which mode wins Group A/B/C)
  - Routing accuracy report (did the orchestrator route correctly?)
  - JSON export for the Streamlit UI

Usage:
    from eval.compare_modes import ModeComparison
    comp = ModeComparison()
    report = await comp.run()
    print(report.summary())
    report.save("eval/results/latest.json")
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from eval.ragas_eval import RagasEvaluator, EvalResult
from eval.golden_dataset import (
    GOLDEN_DATASET, get_by_group, get_by_mode, GoldenEntry
)


# ---------------------------------------------------------------------------
# Routing accuracy tracker
# ---------------------------------------------------------------------------

@dataclass
class RoutingAccuracy:
    total:         int
    correct:       int
    by_group:      dict[str, dict]   # group → {total, correct}
    misrouted:     list[dict]        # questions that went to the wrong agent

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'─'*50}",
            f"  Routing accuracy : {self.correct}/{self.total} ({self.accuracy*100:.1f}%)",
        ]
        for g, counts in self.by_group.items():
            acc = counts["correct"] / counts["total"] if counts["total"] else 0
            lines.append(f"    Group {g}: {counts['correct']}/{counts['total']} ({acc*100:.0f}%)")
        if self.misrouted:
            lines.append(f"  Misrouted ({len(self.misrouted)}):")
            for m in self.misrouted[:5]:
                lines.append(f"    Q: {m['question'][:45]}...")
                lines.append(f"       expected={m['expected']}  actual={m['actual']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

@dataclass
class ComparisonReport:
    flat_rag:        EvalResult
    graph_rag:       EvalResult
    hybrid:          EvalResult
    routing:         RoutingAccuracy
    run_at:          str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_elapsed:   float = 0.0

    def winner(self, metric: str) -> str:
        """Return the mode with the highest score for a given metric."""
        scores = {
            "flat_rag":  getattr(self.flat_rag,  metric, 0.0),
            "graph_rag": getattr(self.graph_rag, metric, 0.0),
            "hybrid":    getattr(self.hybrid,    metric, 0.0),
        }
        return max(scores, key=scores.get)

    def overall_winner(self) -> str:
        return self.winner("average")

    def summary(self) -> str:
        METRICS = [
            "faithfulness", "answer_relevancy",
            "context_precision", "context_recall", "average"
        ]
        MODES = [
            ("flat_rag",  self.flat_rag),
            ("graph_rag", self.graph_rag),
            ("hybrid",    self.hybrid),
        ]

        lines = [
            "\n" + "=" * 70,
            "  RAG vs GraphRAG vs Hybrid — RAGAS Evaluation Report",
            f"  Run at: {self.run_at}   Total elapsed: {self.total_elapsed:.1f}s",
            "=" * 70,
            f"\n  {'Metric':<24} {'flat_rag':>10} {'graph_rag':>10} {'hybrid':>10} {'Winner':>12}",
            "  " + "─" * 66,
        ]

        for metric in METRICS:
            row = f"  {metric:<24}"
            for _, result in MODES:
                val = getattr(result, metric, result.average if metric == "average" else 0.0)
                row += f" {val:>10.3f}"
            w = self.winner(metric)
            row += f" {w:>12}"
            lines.append(row)

        lines += [
            "  " + "─" * 66,
            f"\n  Overall winner: {self.overall_winner().upper()}",
        ]

        # Per-group expected winner vs actual winner
        lines += [
            "\n  Per-group analysis (which mode scores highest on its intended group):",
            f"    Group A (factual)    — expected: flat_rag  | average: "
            f"flat={self.flat_rag.average:.3f} graph={self.graph_rag.average:.3f} hybrid={self.hybrid.average:.3f}",
        ]

        # Routing accuracy
        lines.append(self.routing.summary())
        lines.append("=" * 70)

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "run_at":        self.run_at,
            "total_elapsed": self.total_elapsed,
            "flat_rag":      self.flat_rag.to_dict(),
            "graph_rag":     self.graph_rag.to_dict(),
            "hybrid":        self.hybrid.to_dict(),
            "routing": {
                "accuracy":  self.routing.accuracy,
                "correct":   self.routing.correct,
                "total":     self.routing.total,
                "misrouted": self.routing.misrouted,
            },
            "winners": {
                m: self.winner(m)
                for m in ["faithfulness", "answer_relevancy",
                           "context_precision", "context_recall", "average"]
            },
        }

    def save(self, path: str = "eval/results/latest.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  [compare_modes] report saved → {path}")


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

class ModeComparison:
    """
    Orchestrates RAGAS evaluation across all three retrieval modes
    and computes routing accuracy.
    """

    def __init__(self, llm_model: str = "gemini-1.5-flash", timeout: int = 60):
        self.evaluator = RagasEvaluator(llm_model=llm_model, timeout=timeout)

    async def run(
        self,
        groups: list[str] | None = None,
        user_id: str = "eval_user",
    ) -> ComparisonReport:
        """
        Run the full comparison.

        Args:
            groups  : subset of ["A", "B", "C"] to evaluate. None = all 30 questions.
            user_id : user_id for agent session scoping

        Returns:
            ComparisonReport with all results
        """
        entries = GOLDEN_DATASET
        if groups:
            entries = [e for g in groups for e in get_by_group(g)]

        print(f"\n[compare_modes] starting full evaluation — {len(entries)} questions × 3 modes")
        t0 = time.time()

        # Run all three modes (sequentially to avoid LLM rate limits)
        flat_result  = await self.evaluator.evaluate_mode("flat_rag",  entries=entries, user_id=user_id)
        graph_result = await self.evaluator.evaluate_mode("graph_rag", entries=entries, user_id=user_id)
        hybrid_result= await self.evaluator.evaluate_mode("hybrid",    entries=entries, user_id=user_id)

        # Routing accuracy
        routing = await self._measure_routing_accuracy(entries, user_id)

        elapsed = round(time.time() - t0, 1)

        report = ComparisonReport(
            flat_rag=flat_result,
            graph_rag=graph_result,
            hybrid=hybrid_result,
            routing=routing,
            total_elapsed=elapsed,
        )
        return report

    async def run_group(self, group: str, user_id: str = "eval_user") -> ComparisonReport:
        """Convenience — run comparison on a single group only."""
        return await self.run(groups=[group], user_id=user_id)

    async def _measure_routing_accuracy(
        self,
        entries: list[GoldenEntry],
        user_id: str,
    ) -> RoutingAccuracy:
        """
        Run each question through the orchestrator and check which
        sub-agent was chosen vs the expected agent.
        """
        from agents.runner import get_runner
        from agents.models import RetrievalMode

        runner = get_runner()

        total    = len(entries)
        correct  = 0
        by_group: dict[str, dict] = {"A": {"total": 0, "correct": 0},
                                     "B": {"total": 0, "correct": 0},
                                     "C": {"total": 0, "correct": 0}}
        misrouted: list[dict] = []

        MODE_MAP = {
            "flat_rag":  RetrievalMode.FLAT.value,
            "graph_rag": RetrievalMode.GRAPH.value,
            "hybrid":    RetrievalMode.HYBRID.value,
        }

        print(f"\n[compare_modes] measuring routing accuracy ({total} questions)...")

        for i, entry in enumerate(entries, 1):
            by_group[entry.group]["total"] += 1
            try:
                response = await asyncio.wait_for(
                    runner.run(entry.question, user_id=user_id),
                    timeout=30,
                )
                actual   = response.retrieval_mode.value
                expected = MODE_MAP.get(entry.expected_mode, entry.expected_mode)

                if actual == expected:
                    correct += 1
                    by_group[entry.group]["correct"] += 1
                else:
                    misrouted.append({
                        "question": entry.question,
                        "expected": expected,
                        "actual":   actual,
                        "group":    entry.group,
                    })
                print(f"  [{i}/{total}] {entry.question[:45]:<45} expected={expected:<10} actual={actual}")

            except Exception as e:
                print(f"  [{i}/{total}] ERROR: {e}")
                misrouted.append({
                    "question": entry.question,
                    "expected": entry.expected_mode,
                    "actual":   "error",
                    "group":    entry.group,
                })

            if i < total:
                await asyncio.sleep(0.3)

        return RoutingAccuracy(
            total=total, correct=correct,
            by_group=by_group, misrouted=misrouted,
        )