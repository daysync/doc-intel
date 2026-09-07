"""LLM-as-judge with a fixed rubric. The judge is never the model under test.

Two uses: grading an answer against an expected value when string matching is too strict
(paraphrases, formats), and scoring faithfulness claim by claim. Every verdict is a
structured output with a one-line reason, so disagreements can be read, not just counted.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from doc_intel.llm.base import LLM
from doc_intel.llm.types import LLMRequest, Message, TextPart

RUBRIC = """You are a strict grader. Compare the candidate answer with the reference answer for the
question. Judge only whether the candidate conveys the same fact as the reference:
- correct: same value or entity, formatting and wording may differ (78,60 EUR = 78.60 EUR).
- partial: the fact is there but incomplete or mixed with a wrong detail.
- incorrect: different value, wrong entity, or the candidate declines while a reference exists.
Give a one-line reason."""


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grade: Literal["correct", "partial", "incorrect"]
    reason: str = Field(max_length=300)


class ClaimCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    supported: bool


class ClaimChecks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ClaimCheck]


class Judge:
    def __init__(self, llm: LLM, model: str, model_under_test: str | None = None) -> None:
        if model_under_test is not None and model == model_under_test:
            raise ValueError(f"the judge ({model}) must not be the model under test")
        self.llm = llm
        self.model = model

    async def grade(self, question: str, reference: str, candidate: str) -> Verdict:
        text = f"Question: {question}\nReference answer: {reference}\nCandidate answer: {candidate}"
        request = LLMRequest(
            model=self.model,
            system=RUBRIC,
            messages=[Message(role="user", parts=[TextPart(text=text)])],
            max_tokens=256,
        )
        return (await self.llm.complete(request, Verdict)).output

    async def check_claims(self, answer: str, contexts: list[str]) -> ClaimChecks:
        """Split the answer into atomic claims and mark each as supported by the contexts or not."""
        joined = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        system = (
            "Break the answer into its atomic factual claims (one fact each, keep numbers exact). "
            "For every claim decide whether the contexts state or directly entail it. Be strict: a "
            "number, name or date that does not appear in the contexts is unsupported."
        )
        text = f"Contexts:\n{joined}\n\nAnswer: {answer}"
        request = LLMRequest(
            model=self.model,
            system=system,
            messages=[Message(role="user", parts=[TextPart(text=text)])],
            max_tokens=1024,
        )
        return (await self.llm.complete(request, ClaimChecks)).output
