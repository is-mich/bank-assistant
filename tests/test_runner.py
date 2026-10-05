import csv
import json

import pytest

from document_qa_bot.config import Config
from document_qa_bot.evaluation.judge import JudgeResult
from document_qa_bot.evaluation.runner import (
    Question, apply_overrides, format_summary, load_questions, run_evaluation, summarize, write_results,
)
from document_qa_bot.generation.prompts import NO_CONTEXT_FALLBACK_MESSAGE

HEADER = "id,question,reference_answer,source_doc,category,answerable\n"


def source(page, score):
    return {"source": "doc.pdf", "page": page, "chunk": f"text {page}", "score": score}


class FakePipeline:
    """Answers from a dict {question: output}; raises for questions mapped to an Exception."""

    def __init__(self, outputs):
        self.outputs = outputs

    def answer(self, question):
        out = self.outputs[question]
        if isinstance(out, Exception):
            raise out
        return out


class FakeJudge:
    def judge(self, question, reference, answerable, context, answer):
        return JudgeResult(1.0, 0.5, "ok", 10, 5)


def output(answer, sources, prompt_tokens=1000, completion_tokens=200):
    return {
        "answer": answer, "sources": sources, "latency_seconds": 1.5,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
    }


def questions():
    return [
        Question("q1", "Q1", "ref", "doc.pdf:2", "simple", True),
        Question("q2", "Q2", "", "", "out-of-scope", False),
        Question("q3", "Q3", "ref", "doc.pdf", "simple", True),
    ]


@pytest.fixture
def config():
    return Config()


def test_load_questions_parses_rows(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text(HEADER + 'q1,Ερώτηση,Απάντηση,doc.pdf:3,simple,true\nq2,Άλλη,,,out-of-scope,false\n',
                    encoding="utf-8")
    loaded = load_questions(path)
    assert [(q.id, q.category, q.answerable) for q in loaded] == [("q1", "simple", True), ("q2", "out-of-scope", False)]
    assert loaded[0].source_doc == "doc.pdf:3"


def test_load_questions_rejects_unknown_category(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text(HEADER + "q1,Q,A,doc.pdf,weird,true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown category"):
        load_questions(path)


def test_load_questions_rejects_missing_columns(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text("id,question\nq1,Q\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_questions(path)


def test_run_evaluation_scores_retrieval_cost_and_fallback(config):
    pipeline = FakePipeline({
        "Q1": output("answer", [source(0, 0.5), source(1, 0.4)]),         # gold page 2 = index 1 -> rank 2
        "Q2": output(NO_CONTEXT_FALLBACK_MESSAGE, []),
        "Q3": RuntimeError("boom"),
    })

    results = run_evaluation(pipeline, questions(), k=5, config=config, judge=FakeJudge())
    q1, q2, q3 = results

    assert q1.recall_at_k == 1.0 and q1.mrr == 0.5 and q1.top_score == 0.5
    assert q1.fallback_used is False
    assert q1.cost_usd == pytest.approx((1000 * 0.15 + 200 * 0.60) / 1_000_000)
    assert (q1.correctness, q1.faithfulness, q1.judge_reasoning) == (1.0, 0.5, "ok")
    assert q2.fallback_used is True and q2.recall_at_k is None and q2.top_score is None
    assert q3.error.startswith("RuntimeError") and q3.answer == ""


def test_run_evaluation_without_judge_leaves_generation_metrics_empty(config):
    pipeline = FakePipeline({"Q1": output("a", [source(1, 0.5)])})
    [result] = run_evaluation(pipeline, questions()[:1], k=5, config=config, judge=None)
    assert result.correctness is None and result.faithfulness is None
    assert result.recall_at_k == 1.0


def test_summarize_groups_by_category_and_excludes_errors(config):
    pipeline = FakePipeline({
        "Q1": output("a", [source(1, 0.5)]),
        "Q2": output(NO_CONTEXT_FALLBACK_MESSAGE, []),
        "Q3": RuntimeError("boom"),
    })
    rows = {r["group"]: r for r in summarize(run_evaluation(pipeline, questions(), 5, config))}

    assert rows["ALL"]["n"] == 3 and rows["ALL"]["errors"] == 1
    assert rows["simple"]["n"] == 2 and rows["simple"]["errors"] == 1
    assert rows["simple"]["recall_at_k"] == 1.0
    assert rows["out-of-scope"]["fallback_rate"] == 1.0
    assert "recall_at_k" in format_summary(list(rows.values()))


def test_write_results_creates_csv_and_config_snapshot(tmp_path, config):
    config.experiment_name = "unit_test_experiment"
    pipeline = FakePipeline({"Q1": output("a", [source(1, 0.5)])})
    results = run_evaluation(pipeline, questions()[:1], 5, config)

    path = write_results(results, config, k=5, questions_file="q.csv", results_dir=tmp_path)

    assert path.parent == tmp_path and path.name.startswith("unit_test_experiment_")
    with open(path, newline="", encoding="utf-8-sig") as f:
        row = next(csv.DictReader(f))
    assert row["id"] == "q1" and float(row["recall_at_k"]) == 1.0
    assert json.loads(row["retrieved"])[0]["score"] == 0.5
    snapshot = json.loads(path.with_suffix(".config.json").read_text(encoding="utf-8"))
    assert snapshot["config"]["experiment_name"] == "unit_test_experiment"
    assert snapshot["k"] == 5


def test_apply_overrides_casts_types(config):
    apply_overrides(config, ["top_k=12", "query_expansion=false", "min_relevance_score=0.4"])
    assert config.top_k == 12 and config.query_expansion is False and config.min_relevance_score == 0.4


@pytest.mark.parametrize("pair", ["nope=1", "top_k", "experiment_name=x", "chroma_db_path=x"])
def test_apply_overrides_rejects_invalid_or_protected_keys(config, pair):
    with pytest.raises(SystemExit):
        apply_overrides(config, [pair])
