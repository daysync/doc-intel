import json

from psycopg_pool import AsyncConnectionPool

from doc_intel.rag.answer import Answerer
from doc_intel.rag.index import Indexer, Retriever
from doc_intel.rag.qa import QuestionAnswering, RagConfig
from doc_intel.rag.rerank import Reranker
from tests.conftest import requires_postgres
from tests.llm.fakes import FakeLLM
from tests.rag.fakes import HashEmbedder
from tests.rag.test_index import _stored

pytestmark = requires_postgres


async def test_ask_scopes_retrieves_and_cites(pool: AsyncConnectionPool) -> None:
    embedder = HashEmbedder()
    a = await _stored(pool, "INV-7001", "Beauty Supplies Ltd")
    b = await _stored(pool, "INV-7002", "Glow Wholesale")
    await Indexer(pool, embedder).index(a)
    await Indexer(pool, embedder).index(b)
    retriever = Retriever(pool, embedder)

    class EchoLLM(FakeLLM):
        """Cites the first excerpt it is shown with a quote taken from it."""

        async def _complete_raw(self, request, schema):  # type: ignore[no-untyped-def]
            text = request.messages[0].parts[0].text
            first_id = int(text.split("[", 1)[1].split("]", 1)[0])
            quote = text.split("]", 1)[1].split("\n", 2)[1][:30]
            self.texts = [
                json.dumps(
                    {
                        "answer": f"see {first_id}",
                        "citations": [{"chunk_id": first_id, "quote": quote}],
                        "not_in_documents": False,
                        "confidence": 0.8,
                    }
                )
            ]
            return await super()._complete_raw(request, schema)

    qa = QuestionAnswering(retriever, Answerer(EchoLLM(""), "m"), config=RagConfig(k=3))
    result = await qa.ask("Who is the supplier on invoice INV-7002?")
    assert result.scope == [b.document_id]
    assert all(h.document_id == b.document_id for h in result.hits)
    assert (
        result.supported and result.citations and result.citations[0].document_id == b.document_id
    )

    reranked = QuestionAnswering(
        retriever,
        Answerer(EchoLLM(""), "m"),
        Reranker(FakeLLM(json.dumps({"scores": []})), "m"),
        RagConfig(k=2, candidates=10, rerank=True),
    )
    result = await reranked.ask("grand total of INV-7001")
    assert len(result.hits) == 2 and result.scope == [a.document_id]
