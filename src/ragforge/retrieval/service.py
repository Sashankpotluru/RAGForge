"""Consistent single-worker retrieval snapshots across both indexes."""

from collections.abc import Callable

from ragforge.domain.models import RetrievedChunk, SearchRequest
from ragforge.domain.protocols import Reranker
from ragforge.retrieval.fusion import reciprocal_rank_fusion
from ragforge.runtime import Runtime


class SearchService:
    def __init__(
        self,
        runtime: Runtime,
        reranker: Reranker | None = None,
        rewriter: Callable[[str], str] | None = None,
    ) -> None:
        self.runtime = runtime
        self.reranker = reranker
        self.rewriter = rewriter

    def search(self, request: SearchRequest) -> list[RetrievedChunk]:
        if request.rewrite_query and self.rewriter is None:
            raise ValueError("query rewriter is not configured")
        if request.mode == "hybrid_rerank" and self.reranker is None:
            raise ValueError("reranker is not configured")
        with self.runtime.lock:
            queries = [request.query]
            if request.rewrite_query and self.rewriter is not None:
                rewritten = self.rewriter(request.query).strip()
                if not rewritten:
                    raise ValueError("query rewriter returned empty text")
                if rewritten != request.query:
                    queries.append(rewritten)
            active_ids = [
                c.chunk_id for c in self.runtime.catalog.active_chunks(request.collection)
            ]
            rankings = {}
            for index, query in enumerate(queries):
                suffix = "" if index == 0 else "_rewritten"
                rankings["dense" + suffix] = self.runtime.dense.search(
                    query, request.metadata_filter, request.candidate_limit, allowed_ids=active_ids
                )
                if request.mode != "dense":
                    rankings["lexical" + suffix] = self.runtime.lexical.search(
                        query,
                        request.metadata_filter,
                        request.candidate_limit,
                        allowed_ids=set(active_ids),
                    )
            fused = (rankings["dense"] if len(rankings) == 1 else reciprocal_rank_fusion(rankings))[
                : request.candidate_limit
            ]
            if request.mode == "hybrid_rerank" and self.reranker is not None:
                fused = self.reranker.rank(request.query, fused)
            return fused[: request.result_limit]
