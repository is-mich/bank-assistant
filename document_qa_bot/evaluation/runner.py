"""
evaluation/runner.py
--------------------
Runs a question set through the pipeline directly (no UI) and saves per-question results.

Usage (OPENAI_API_KEY must already be set in the environment):
    python -m document_qa_bot.evaluation.runner --questions eval/questions.csv \
        --experiment baseline --docs data/doc.pdf --set top_k=8 --set query_expansion=false

Question set columns: id, question, reference_answer, source_doc, category, answerable.
Results: results/<experiment>_<timestamp>.csv plus a .config.json snapshot of every setting.
"""

import argparse
import csv
import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional

from ..config import Config, get_config
from ..generation.prompts import NO_CONTEXT_FALLBACK_MESSAGE
from .metrics import cost_usd, parse_gold, reciprocal_rank, recall_at_k

logger = logging.getLogger(__name__)

QUESTION_COLUMNS = ["id", "question", "reference_answer", "source_doc", "category", "answerable"]
CATEGORIES = {"simple", "multi-doc", "unanswerable", "ambiguous", "out-of-scope"}
TRUE_VALUES = {"true", "1", "yes", "ναι"}
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


@dataclass
class Question:
    id: str
    question: str
    reference_answer: str
    source_doc: str
    category: str
    answerable: bool


@dataclass
class QuestionResult:
    id: str
    category: str
    answerable: bool
    question: str
    reference_answer: str
    answer: str = ""
    fallback_used: Optional[bool] = None
    retrieved: Optional[list] = None          # [{"source", "page", "score", "chunk"}], best first
    top_score: Optional[float] = None
    recall_at_k: Optional[float] = None
    mrr: Optional[float] = None
    correctness: Optional[float] = None
    faithfulness: Optional[float] = None
    judge_reasoning: str = ""
    latency_seconds: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    judge_prompt_tokens: Optional[int] = None
    judge_completion_tokens: Optional[int] = None
    error: str = ""


def load_questions(path: str | Path) -> list[Question]:
    """Read and validate the question set CSV."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in QUESTION_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Question set is missing columns: {missing}")
        questions = []
        for line, row in enumerate(reader, start=2):
            category = row["category"].strip()
            if category not in CATEGORIES:
                raise ValueError(f"Line {line}: unknown category {category!r}; expected one of {sorted(CATEGORIES)}")
            questions.append(Question(
                id=row["id"].strip(),
                question=row["question"].strip(),
                reference_answer=row["reference_answer"].strip(),
                source_doc=row["source_doc"].strip(),
                category=category,
                answerable=row["answerable"].strip().lower() in TRUE_VALUES,
            ))
    return questions


def run_evaluation(pipeline: Any, questions: Iterable[Question], k: int,
                   config: Config, judge: Any = None) -> list[QuestionResult]:
    """Answer every question with `pipeline.answer` and score it; failures are recorded, not raised."""
    questions = list(questions)
    results = []
    for i, q in enumerate(questions, start=1):
        result = QuestionResult(q.id, q.category, q.answerable, q.question, q.reference_answer)
        try:
            out = pipeline.answer(q.question)
            retrieved = [
                {"source": s["source"], "page": s["page"], "score": s["score"], "chunk": s["chunk"]}
                for s in out["sources"]
            ]
            gold = parse_gold(q.source_doc) if q.answerable else []
            result.answer = out["answer"]
            result.fallback_used = out["answer"] == NO_CONTEXT_FALLBACK_MESSAGE
            result.retrieved = retrieved
            scores = [c["score"] for c in retrieved if c["score"] is not None]
            result.top_score = max(scores) if scores else None
            result.recall_at_k = recall_at_k(retrieved, gold, k)
            result.mrr = reciprocal_rank(retrieved, gold, k)
            result.latency_seconds = out["latency_seconds"]
            result.prompt_tokens = out["prompt_tokens"]
            result.completion_tokens = out["completion_tokens"]
            result.cost_usd = cost_usd(
                out["prompt_tokens"], out["completion_tokens"],
                config.price_per_1m_input_tokens, config.price_per_1m_output_tokens,
            )
            if judge is not None:
                verdict = judge.judge(
                    q.question, q.reference_answer, q.answerable,
                    [c["chunk"] for c in retrieved], out["answer"],
                )
                result.correctness = verdict.correctness
                result.faithfulness = verdict.faithfulness
                result.judge_reasoning = verdict.reasoning
                result.judge_prompt_tokens = verdict.prompt_tokens
                result.judge_completion_tokens = verdict.completion_tokens
        except Exception as e:  # keep going so one bad question does not lose the run
            logger.exception(f"Question {q.id} failed")
            result.error = f"{type(e).__name__}: {e}"
        results.append(result)
        print(
            f"[{i:>3}/{len(questions)}] {q.id:<8} {q.category:<13} "
            f"recall={_fmt(result.recall_at_k)} mrr={_fmt(result.mrr)} "
            f"correct={_fmt(result.correctness)} faithful={_fmt(result.faithfulness)} "
            f"fallback={result.fallback_used} {result.error}"
        )
    return results


def _fmt(value: Optional[float]) -> str:
    return "  - " if value is None else f"{value:.2f}"


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return mean(present) if present else None


def summarize(results: list[QuestionResult]) -> list[dict]:
    """Aggregate per category and overall; failed questions are counted but excluded from the means."""
    groups: dict[str, list[QuestionResult]] = {"ALL": results}
    for r in results:
        groups.setdefault(r.category, []).append(r)
    rows = []
    for name, rs in groups.items():
        ok = [r for r in rs if not r.error]
        rows.append({
            "group": name,
            "n": len(rs),
            "errors": len(rs) - len(ok),
            "recall_at_k": _mean(r.recall_at_k for r in ok),
            "mrr": _mean(r.mrr for r in ok),
            "correctness": _mean(r.correctness for r in ok),
            "faithfulness": _mean(r.faithfulness for r in ok),
            "fallback_rate": _mean(float(r.fallback_used) for r in ok if r.fallback_used is not None),
            "latency_s": _mean(r.latency_seconds for r in ok),
            "cost_usd_per_q": _mean(r.cost_usd for r in ok),
        })
    return rows


def format_summary(rows: list[dict]) -> str:
    """Plain-text table of `summarize` output."""
    cols = ["group", "n", "errors", "recall_at_k", "mrr", "correctness", "faithfulness",
            "fallback_rate", "latency_s", "cost_usd_per_q"]

    def cell(value: Any) -> str:
        if value is None:
            return "-"
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    table = [cols] + [[cell(row[c]) for c in cols] for row in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(cols))]
    return "\n".join("  ".join(v.ljust(w) for v, w in zip(r, widths)) for r in table)


def write_results(results: list[QuestionResult], config: Config, k: int,
                  questions_file: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Save results/<experiment>_<timestamp>.csv and a .config.json snapshot next to it."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"{config.experiment_name}_{stamp}.csv"

    fields = [f.name for f in dataclasses.fields(QuestionResult)]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = dataclasses.asdict(r)
            row["retrieved"] = json.dumps(
                [{**c, "chunk": c["chunk"][:200]} for c in (r.retrieved or [])], ensure_ascii=False
            )
            writer.writerow(row)

    snapshot = {
        "timestamp": stamp,
        "questions_file": questions_file,
        "k": k,
        "config": dataclasses.asdict(config),
    }
    csv_path.with_suffix(".config.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return csv_path


def apply_overrides(config: Config, pairs: list[str]) -> None:
    """Apply KEY=VALUE overrides to Config, casting to the current value's type."""
    protected = {"experiment_name", "base_path", "chroma_db_path", "sample_docs_path"}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not hasattr(config, key):
            raise SystemExit(f"Invalid --set {pair!r}: expected an existing Config field as KEY=VALUE")
        if key in protected:
            raise SystemExit(f"--set cannot change {key}; use --experiment instead")
        current = getattr(config, key)
        if isinstance(current, bool):
            value: Any = raw.strip().lower() in TRUE_VALUES
        else:
            value = type(current)(raw)
        setattr(config, key, value)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on a question set.")
    parser.add_argument("--questions", required=True, help="CSV with the question set")
    parser.add_argument("--experiment", help="experiment name; selects its own Chroma directory")
    parser.add_argument("--docs", nargs="*", default=[], help="documents to ingest into the experiment first")
    parser.add_argument("--k", type=int, help="cut-off for recall@k and MRR (default: Config.top_k)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a Config value, e.g. --set top_k=12")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge (retrieval metrics only)")
    args = parser.parse_args(argv)

    # Tracing stays off for evaluation runs.
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

    config = get_config()
    if args.experiment:
        config.use_experiment(args.experiment)
    apply_overrides(config, args.overrides)
    k = args.k or config.top_k

    questions = load_questions(args.questions)

    from ..rag_pipeline import RAGPipeline  # lazy: needs the OpenAI stack
    pipeline = RAGPipeline()
    for doc in args.docs:
        try:
            print(f"Ingested {pipeline.ingest(doc)} chunks from {doc}")
        except ValueError as e:  # already ingested in this experiment
            print(f"Skipped {doc}: {e}")
    if pipeline.get_stats()["total_chunks"] == 0:
        raise SystemExit("The experiment's collection is empty; pass --docs to ingest documents first.")

    judge = None
    if not args.no_judge:
        from .judge import LLMJudge
        judge = LLMJudge()

    results = run_evaluation(pipeline, questions, k, config, judge)
    path = write_results(results, config, k, args.questions)
    print("\n" + format_summary(summarize(results)))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
