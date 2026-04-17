"""
OCR Service — wraps DeepSeek-OCR-2 as an HTTP API.

POST /v1/ocr
  Request:  {"image_b64": "<base64-encoded JPEG/PNG>", "prompt": "<optional>"}
  Response: {"text": "<ocr output>", "markdown": "<cleaned markdown>"}

GET /v1/ocr/health
  Response: {"status": "ok", "model": "<model name>"}

deepseek_ocr_client.py calls this service when REMOTE_OCR_URL is set.
"""

import os
import base64
import io
import logging
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_NAME = os.getenv("OCR_MODEL", "deepseek-ai/DeepSeek-OCR-2")
BASE_SIZE   = int(os.getenv("OCR_BASE_SIZE", "1024"))
IMAGE_SIZE  = int(os.getenv("OCR_IMAGE_SIZE", "768"))

_model     = None
_tokenizer = None

DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "


def _load():
    global _model, _tokenizer
    log.info(f"Loading OCR model: {MODEL_NAME}")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True,
        attn_implementation="flash_attention_2",
    ).eval().cuda().to(torch.bfloat16)
    log.info("OCR model loaded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield


app = FastAPI(title="DeepSeek-OCR-2 Service", lifespan=lifespan)


class OCRRequest(BaseModel):
    image_b64: str
    prompt: str = DEFAULT_PROMPT


class OCRResponse(BaseModel):
    text: str
    markdown: str


def _b64_to_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _run_ocr(image: Image.Image, prompt: str) -> str:
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded")

    result = _model.infer(
        _tokenizer,
        prompt=prompt,
        image_file=image,
        base_size=BASE_SIZE,
        image_size=IMAGE_SIZE,
        crop_mode=True,
        save_results=False,
    )

    if isinstance(result, dict):
        return result.get("markdown", "") or result.get("text", "") or str(result)
    return str(result)


_SKIP_TOKENS = {"image:", "other:", "PATCHES", "====", "BASE:", "%|", "torch.Size"}


def _clean(raw: str) -> str:
    lines = [l for l in raw.splitlines() if not any(s in l for s in _SKIP_TOKENS)]
    return "\n".join(lines).strip()


@app.get("/v1/ocr/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/ocr", response_model=OCRResponse)
async def ocr(req: OCRRequest):
    try:
        image = _b64_to_image(req.image_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    try:
        raw = _run_ocr(image, req.prompt)
    except Exception as exc:
        log.exception("OCR inference failed")
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}")

    md = _clean(raw)
    return OCRResponse(text=md, markdown=md)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
