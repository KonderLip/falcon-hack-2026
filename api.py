"""
FastAPI demo for Vehicle ReID extractor.

Запуск:
    uvicorn api:app --host 0.0.0.0 --port 8000
Открыть Swagger:
    http://localhost:8000/docs
"""

from __future__ import annotations

import io
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from extractor import VehicleExtractor

app = FastAPI(title="Vehicle ReID — Extractor API", version="1.0.0")
extractor = VehicleExtractor(weights_dir="weights/", device="cuda:0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": str(extractor.device)}


@app.post("/extract")
async def extract(
    image: UploadFile = File(..., description="JPEG/PNG полного кадра"),
    bbox: str = Form(..., description="x,y,w,h в пикселях исходного кадра"),
) -> dict:
    try:
        x, y, w, h = map(int, bbox.split(","))
    except Exception:
        raise HTTPException(status_code=422, detail="bbox must be 'x,y,w,h'")

    raw = await image.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")

    # Сохраняем во временный файл, потому что extractor работает с путями
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil.save(tmp.name, "JPEG", quality=95)
        tmp_path = tmp.name
    try:
        emb = extractor.extract(tmp_path, (x, y, w, h))
    finally:
        os.unlink(tmp_path)

    return {"embedding": emb.tolist(), "dim": int(emb.shape[0])}


@app.post("/extract_batch")
async def extract_batch(
    images: List[UploadFile] = File(...),
    bboxes: List[str] = Form(...),
) -> dict:
    if len(images) != len(bboxes):
        raise HTTPException(status_code=422, detail="len(images) != len(bboxes)")

    import tempfile, os

    tmp_paths = []
    parsed_bboxes = []
    try:
        for img, bb in zip(images, bboxes):
            x, y, w, h = map(int, bb.split(","))
            parsed_bboxes.append((x, y, w, h))
            raw = await img.read()
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                pil.save(tmp.name, "JPEG", quality=95)
                tmp_paths.append(tmp.name)

        emb = extractor.extract_batch(tmp_paths, parsed_bboxes)
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    return {"embeddings": emb.tolist(), "shape": list(emb.shape)}
