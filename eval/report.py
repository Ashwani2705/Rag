"""
eval/report.py

Phase 4 — Report generator.

Converts a ComparisonReport into:
  - JSON  : machine-readable, consumed by the Streamlit UI in Phase 5
  - HTML  : self-contained, shareable demo report
  - Text  : terminal-friendly summary (already on ComparisonReport.summary())

Usage:
    from eval.report import ReportGenerator
    gen = ReportGenerator()
    gen.save_json(report, "eval/results/latest.json")
    gen.save_html(report, "eval/results/latest.html")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class ReportGenerator:
    """Generates JSON and HTML reports from a ComparisonReport."""

    # ── JSON ─────────────────────────────────────────────────────────────────

    @staticmethod
    def save_json(report, path: str = "eval/results/latest.json") -> str:
        """Save a ComparisonReport as a JSON file. Returns the saved path."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"  [report] JSON saved → {path}")
        return path

    # ── HTML ─────────────────────────────────────────────────────────────────

    @classmethod
    def save_html(report, path: str = "eval/results/latest.html") -> str:  # type: ignore[misc]
        """Save a self-contained HTML report. Returns the saved path."""
        raise NotImplementedError("Use generate_html() and write manually.")

    @classmethod
    def generate_html(cls, report) -> str:
        """
        Generate a self-contained HTML string from a ComparisonReport.
        No external dependencies — inline CSS + JS only.
        """
        data = report.to_dict()
        modes   = ["flat_rag", "graph_rag", "hybrid"]
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "average"]

        def _bar(score: float, colour: str) -> str:
            pct = int(score * 100)
            return (
                f'<div class="bar-wrap"><div class="bar" style="width:{pct}%;background:{colour}">'
                f'<span>{score:.3f}</span></div></div>'
            )

        MODE_COLOURS = {"flat_rag": "#1A56DB", "graph_rag": "#0D9488", "hybrid": "#7C3AED"}
        METRIC_LABELS = {
            "faithfulness":      "Faithfulness",
            "answer_relevancy":  "Answer Relevancy",
            "context_precision": "Context Precision",
            "context_recall":    "Context Recall",
            "average":           "Average",
        }

        # Metric rows
        metric_rows = ""
        for m in metrics:
            cells = "".join(
                f'<td>{_bar(data[mode].get(m, data[mode].get("average", 0)), MODE_COLOURS[mode])}</td>'
                for mode in modes
            )
            winner = data["winners"].get(m, "—")
            metric_rows += f"""
            <tr>
                <td class="metric-name">{METRIC_LABELS.get(m, m)}</td>
                {cells}
                <td class="winner" style="color:{MODE_COLOURS.get(winner,'#111')}">{winner}</td>
            </tr>"""

        # Routing table
        routing     = data.get("routing", {})
        misrouted   = routing.get("misrouted", [])
        misrouted_rows = "".join(
            f'<tr><td>{r["question"][:60]}</td><td>{r["expected"]}</td>'
            f'<td style="color:#DC2626">{r["actual"]}</td><td>{r["group"]}</td></tr>'
            for r in misrouted[:10]
        )

        # Per-question scores tables (flat, graph, hybrid)
        def _per_q_table(mode_key: str) -> str:
            rows_data = data.get(mode_key, {}).get("per_question", [])
            if not rows_data:
                return "<p><em>No per-question data available.</em></p>"
            metric_cols = [m for m in metrics if m != "average"]
            header = "<tr><th>#</th><th>Question</th>" + "".join(f"<th>{METRIC_LABELS[m]}</th>" for m in metric_cols) + "</tr>"
            rows = ""
            for i, row in enumerate(rows_data[:30], 1):
                q = str(row.get("question", ""))[:55]
                cells = "".join(
                    f'<td style="color:{"#16A34A" if float(row.get(m,0))>=0.7 else "#DC2626" if float(row.get(m,0))<0.4 else "#B45309"}">{float(row.get(m,0)):.2f}</td>'
                    for m in metric_cols
                )
                rows += f"<tr><td>{i}</td><td>{q}…</td>{cells}</tr>"
            return f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>"

        run_at      = data.get("run_at", "—")
        routing_acc = routing.get("accuracy", 0)
        routing_col = "#16A34A" if routing_acc >= 0.8 else "#B45309" if routing_acc >= 0.6 else "#DC2626"
        winners     = data.get("winners", {})
        overall_winner = winners.get("average", "flat_rag")
        flat_avg    = data.get("flat_rag",  {}).get("average", 0.0)
        graph_avg   = data.get("graph_rag", {}).get("average", 0.0)
        hybrid_avg  = data.get("hybrid",    {}).get("average", 0.0)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG + GraphRAG Evaluation Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #F9FAFB; color: #111827; padding: 2rem; }}
  h1 {{ font-size: 1.6rem; color: #1A56DB; margin-bottom: .25rem; }}
  .sub {{ color: #6B7280; font-size: .85rem; margin-bottom: 2rem; }}
  h2 {{ font-size: 1.1rem; margin: 2rem 0 .75rem; color: #374151; border-left: 4px solid #1A56DB; padding-left: .75rem; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden;
           box-shadow: 0 1px 3px rgba(0,0,0,.1); margin-bottom: 1.5rem; }}
  th {{ background: #1E3A5F; color: white; padding: .6rem 1rem; text-align: left; font-size: .85rem; }}
  td {{ padding: .55rem 1rem; font-size: .85rem; border-bottom: 1px solid #F3F4F6; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #F0F9FF; }}
  .metric-name {{ font-weight: 600; color: #374151; white-space: nowrap; width: 180px; }}
  .winner {{ font-weight: 700; }}
  .bar-wrap {{ background: #E5E7EB; border-radius: 4px; height: 20px; position: relative; min-width: 120px; }}
  .bar {{ height: 20px; border-radius: 4px; display: flex; align-items: center; padding: 0 6px;
          min-width: 30px; transition: width .4s; }}
  .bar span {{ color: white; font-size: .78rem; font-weight: 600; white-space: nowrap; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .stat {{ background: white; border-radius: 8px; padding: 1rem 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  .stat .label {{ font-size: .78rem; color: #6B7280; margin-bottom: .2rem; }}
  .stat .value {{ font-size: 1.5rem; font-weight: 700; }}
  .tabs {{ display: flex; gap: .5rem; margin-bottom: 1rem; }}
  .tab {{ padding: .4rem 1rem; border-radius: 6px; cursor: pointer; font-size: .85rem; font-weight: 500;
          background: #E5E7EB; border: none; color: #374151; }}
  .tab.active {{ background: #1A56DB; color: white; }}
  .panel {{ display: none; }} .panel.active {{ display: block; }}
</style>
</head>
<body>
<h1>RAG + GraphRAG — Evaluation Report</h1>
<p class="sub">Generated: {run_at} &nbsp;|&nbsp; Total elapsed: {data.get("total_elapsed", 0):.1f}s &nbsp;|&nbsp; 30 questions × 3 modes</p>

<div class="stat-grid">
  <div class="stat"><div class="label">Routing Accuracy</div>
    <div class="value" style="color:{routing_col}">{routing_acc*100:.1f}%</div></div>
  <div class="stat"><div class="label">Overall Winner</div>
    <div class="value" style="color:{MODE_COLOURS.get(overall_winner, '#111')}">{overall_winner}</div></div>
  <div class="stat"><div class="label">flat_rag avg</div>
    <div class="value" style="color:#1A56DB">{flat_avg:.3f}</div></div>
  <div class="stat"><div class="label">graph_rag avg</div>
    <div class="value" style="color:#0D9488">{graph_avg:.3f}</div></div>
  <div class="stat"><div class="label">hybrid avg</div>
    <div class="value" style="color:#7C3AED">{hybrid_avg:.3f}</div></div>
</div>

<h2>Metric Comparison</h2>
<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th style="color:#93C5FD">flat_rag</th>
      <th style="color:#5EEAD4">graph_rag</th>
      <th style="color:#C4B5FD">hybrid</th>
      <th>Winner</th>
    </tr>
  </thead>
  <tbody>{metric_rows}</tbody>
</table>

<h2>Routing Accuracy — Misrouted Questions</h2>
{"<p><em>All questions routed correctly ✓</em></p>" if not misrouted else f'''
<table>
  <thead><tr><th>Question</th><th>Expected</th><th>Actual</th><th>Group</th></tr></thead>
  <tbody>{misrouted_rows}</tbody>
</table>'''}

<h2>Per-Question Scores</h2>
<div class="tabs">
  <button class="tab active" onclick="showTab('flat')">flat_rag</button>
  <button class="tab" onclick="showTab('graph')">graph_rag</button>
  <button class="tab" onclick="showTab('hybrid')">hybrid</button>
</div>
<div id="flat"  class="panel active">{_per_q_table("flat_rag")}</div>
<div id="graph" class="panel">{_per_q_table("graph_rag")}</div>
<div id="hybrid-panel" class="panel">{_per_q_table("hybrid")}</div>

<script>
function showTab(id) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  const panelId = id === 'hybrid' ? 'hybrid-panel' : id;
  document.getElementById(panelId).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>"""
        return html

    @classmethod
    def write_html(cls, report, path: str = "eval/results/latest.html") -> str:
        """Generate HTML and write to disk."""
        html = cls.generate_html(report)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  [report] HTML saved → {path}")
        return path