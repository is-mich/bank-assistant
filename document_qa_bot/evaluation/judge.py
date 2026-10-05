"""
evaluation/judge.py
-------------------
LLM-as-judge for correctness (vs the reference answer) and faithfulness (vs the retrieved context).
"""

from dataclasses import dataclass
from typing import Literal

# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..config import get_config
from .judge_prompts import JUDGE_HUMAN_PROMPT, JUDGE_SYSTEM_PROMPT, NO_ANSWER_REFERENCE

SCORES = {"correct": 1.0, "faithful": 1.0, "partial": 0.5, "incorrect": 0.0, "unfaithful": 0.0}


class Verdict(BaseModel):
    """Structured judge output."""
    correctness: Literal["correct", "partial", "incorrect"]
    faithfulness: Literal["faithful", "partial", "unfaithful"]
    reasoning: str = Field(description="One-sentence justification")


@dataclass
class JudgeResult:
    correctness: float
    faithfulness: float
    reasoning: str
    prompt_tokens: int
    completion_tokens: int


class LLMJudge:
    """Scores one answer per call; the model comes from `Config.judge_model`."""

    def __init__(self, model: str | None = None):
        llm = ChatOpenAI(model=model or get_config().judge_model, temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", JUDGE_SYSTEM_PROMPT),
            ("human", JUDGE_HUMAN_PROMPT),
        ])
        self._chain = prompt | llm.with_structured_output(Verdict, include_raw=True)

    def judge(self, question: str, reference: str, answerable: bool,
              context: list[str], answer: str) -> JudgeResult:
        """Rate `answer`; `context` is the text of the retrieved chunks."""
        out = self._chain.invoke({
            "question": question,
            "reference": reference if answerable and reference else NO_ANSWER_REFERENCE,
            "context": "\n\n---\n\n".join(context) or "(no context retrieved)",
            "answer": answer,
        })
        verdict = out["parsed"]
        if verdict is None:
            raise ValueError(f"Judge output could not be parsed: {out.get('parsing_error')}")
        usage = getattr(out["raw"], "usage_metadata", None) or {}
        return JudgeResult(
            correctness=SCORES[verdict.correctness],
            faithfulness=SCORES[verdict.faithfulness],
            reasoning=verdict.reasoning,
            prompt_tokens=int(usage.get("input_tokens", 0)),
            completion_tokens=int(usage.get("output_tokens", 0)),
        )
