from collections import deque

from bank_assistant.rag.grader import RelevanceGrade
from bank_assistant.rag.retriever import Retriever
from bank_assistant.rag.schemas import RetrievedChunk
from bank_assistant.workflows.faq.baseline import ABSTENTION, BaselineFaq, FaqDraft
from bank_assistant.workflows.faq.crag import CorrectiveFaq
from tests.fakes import FakeStructuredLLM


def chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("#")[0],
        source=f"{chunk_id.split('#')[0]}.md",
        chunking_version="test",
        text=f"evidence for {chunk_id}",
        score=0.9,
    )


class SequenceIndex:
    def __init__(self, results: dict[str, list[RetrievedChunk]]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str, _k: int) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.results.get(query, [])


class SequenceGrader:
    def __init__(self, grades: list[RelevanceGrade | Exception]) -> None:
        self.grades = deque(grades)
        self.calls = 0

    def grade(self, _question: str, _retrieval) -> RelevanceGrade:
        self.calls += 1
        grade = self.grades.popleft()
        if isinstance(grade, Exception):
            raise grade
        return grade


class FixedRewriter:
    def __init__(self, query: str = "rewritten") -> None:
        self.query = query
        self.calls = 0

    def rewrite(self, _question: str) -> str:
        self.calls += 1
        return self.query


def workflow(index, grader, rewriter, generator, attempts=2):
    retriever = Retriever(index, top_k=2)
    baseline = BaselineFaq(retriever, generator)
    return CorrectiveFaq(retriever, baseline, grader, rewriter, attempts)


def test_poor_then_relevant_retrieval_generates_from_accepted_context() -> None:
    index = SequenceIndex({"question": [chunk("weak#0")], "rewritten": [chunk("hours#0")]})
    grader = SequenceGrader(
        [
            RelevanceGrade(relevant=False, reason="weak"),
            RelevanceGrade(relevant=True, reason="supported"),
        ]
    )
    rewriter = FixedRewriter()
    generator = FakeStructuredLLM(FaqDraft(answerable=True, answer="8:30", source_ids=["hours#0"]))

    result = workflow(index, grader, rewriter, generator).answer("question")

    assert result.grounded and result.answer == "8:30"
    assert result.source_ids == ("hours#0",)
    assert result.retrieval_attempts == 2
    assert index.queries == ["question", "rewritten"]
    assert grader.calls == 2 and rewriter.calls == 1 and generator.calls == 1


def test_attempt_limit_abstains_and_never_generates_rejected_context() -> None:
    index = SequenceIndex({"question": [chunk("weak#0")], "rewritten": [chunk("weak#1")]})
    grader = SequenceGrader(
        [
            RelevanceGrade(relevant=False, reason="weak"),
            RelevanceGrade(relevant=False, reason="weak"),
        ]
    )
    generator = FakeStructuredLLM(
        FaqDraft(answerable=True, answer="unsupported", source_ids=["weak#1"])
    )

    result = workflow(index, grader, FixedRewriter(), generator, attempts=2).answer("question")

    assert not result.grounded and result.answer == ABSTENTION
    assert result.retrieval_attempts == 2
    assert len(index.queries) == 2 and grader.calls == 2 and generator.calls == 0


def test_grader_failure_fails_closed_without_rewrite_or_generation() -> None:
    index = SequenceIndex({"question": [chunk("maybe#0")]})
    grader = SequenceGrader([RuntimeError("private grader failure")])
    rewriter = FixedRewriter()
    generator = FakeStructuredLLM(
        FaqDraft(answerable=True, answer="unsupported", source_ids=["maybe#0"])
    )

    result = workflow(index, grader, rewriter, generator).answer("question")

    assert result.answer == ABSTENTION
    assert result.retrieval_attempts == 1
    assert rewriter.calls == 0 and generator.calls == 0


def test_empty_retrieval_can_rewrite_but_remains_bounded() -> None:
    index = SequenceIndex({"rewritten": [chunk("hours#0")]})
    grader = SequenceGrader([RelevanceGrade(relevant=True, reason="supported")])
    generator = FakeStructuredLLM(FaqDraft(answerable=True, answer="8:30", source_ids=["hours#0"]))

    result = workflow(index, grader, FixedRewriter(), generator).answer("question")

    assert result.grounded and result.retrieval_attempts == 2
    assert index.queries == ["question", "rewritten"]
