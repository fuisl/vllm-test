"""
Rerank Adapter — bridges Cohere/Jina rerank format to vLLM /v1/score format.

Cohere request:  POST /v2/rerank
  {"model": ..., "query": ..., "documents": [...]}

vLLM request:    POST /v1/score
  {"model": ..., "text_1": "query", "text_2": ["doc1", ...]}

Cohere response: {"results": [{"index": ..., "relevance_score": ...}]}
vLLM response:   {"data": [{"index": ..., "score": ...}]}

Min-max normalization converts raw vLLM logits to [0, 1].
"""

import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union

VLLM_RERANK_URL = os.getenv("VLLM_RERANK_URL", "http://vllm-rerank:8001/v1/score")

app = FastAPI(title="Rerank Adapter")


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Union[int, None] = None


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: List[RerankResult]


def _minmax_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v2/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    payload = {
        "model": req.model,
        "text_1": req.query,
        "text_2": req.documents,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(VLLM_RERANK_URL, json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"vLLM error: {exc}")

    data = resp.json().get("data", [])
    raw_scores = [item["score"] for item in sorted(data, key=lambda x: x["index"])]
    normalized = _minmax_normalize(raw_scores)

    results = [
        RerankResult(index=i, relevance_score=normalized[i])
        for i in range(len(normalized))
    ]

    if req.top_n is not None:
        results = sorted(results, key=lambda x: x.relevance_score, reverse=True)[: req.top_n]

    return RerankResponse(results=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
