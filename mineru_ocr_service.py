"""
MinerU OCR Service — wraps MinerU2.5-Pro-2604-1.2B as an HTTP API.

POST /v1/ocr
  Request:  {"image_b64": "<base64-encoded PNG>", "lang": "vi"}
  Response: {"markdown": "<extracted markdown>"}

GET /v1/ocr/health
  Response: {"status": "ok", "model": "<model name>"}
"""

import os
import base64
import io
import logging
from contextlib import asynccontextmanager

from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_NAME = os.getenv("OCR_MODEL", "opendatalab/MinerU2.5-Pro-2604-1.2B")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.70"))

_client = None


def _load():
    global _client
    log.info(f"Loading MinerU model: {MODEL_NAME}")
    from vllm import LLM
    from mineru_vl_utils import MinerUClient, MinerULogitsProcessor

    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=8192,
        logits_processors=[MinerULogitsProcessor],
    )
    _client = MinerUClient(
        backend="vllm-engine",
        vllm_llm=llm,
        image_analysis=False,
    )
    log.info("MinerU model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield


app = FastAPI(title="MinerU OCR Service", lifespan=lifespan)


class OCRRequest(BaseModel):
    image_b64: str
    lang: str = "vi"


class OCRResponse(BaseModel):
    markdown: str


def _b64_to_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _normalize_vietnamese(text: str) -> str:
    try:
        from underthesea import text_normalize
        return text_normalize(text)
    except Exception:
        return text


@app.get("/v1/ocr/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/ocr", response_model=OCRResponse)
async def ocr(req: OCRRequest):
    if _client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image = _b64_to_image(req.image_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    try:
        from mineru_vl_utils.post_process import json2md
        content_list = _client.two_step_extract(image)
        markdown = json2md(content_list)
        markdown = _normalize_vietnamese(markdown)
    except Exception as exc:
        log.exception("OCR inference failed")
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}")

    return OCRResponse(markdown=markdown)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
