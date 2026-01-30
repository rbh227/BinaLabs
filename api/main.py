"""
DA-SegFormer FastAPI backend — runs on Jetstream with GPU.

Usage:
    export MODEL_PATH=/working/runs/rescuenet_final_b4_ohem_cosine_V2/BEST_MODELS_ARCHIVE/checkpoint-mIoU-0.7461-Ep255.0
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import io
import base64

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from api.inference import load_model, predict, CLASS_NAMES, PALETTE_RGB

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/working/runs/rescuenet_final_b4_ohem_cosine_V2/BEST_MODELS_ARCHIVE/checkpoint-mIoU-0.7461-Ep255.0",
)

app = FastAPI(title="DA-SegFormer API")

# Allow the Next.js frontend (Vercel or localhost) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this to your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
processor = None
device = None


@app.on_event("startup")
def startup():
    global model, processor, device
    print(f"Loading model from: {MODEL_PATH}")
    model, processor, device = load_model(MODEL_PATH)
    print(f"Model loaded on {device}")


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": str(device)}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    mask_pil, overlay_pil = predict(model, processor, image, device)

    return JSONResponse({
        "mask": pil_to_base64(mask_pil),
        "overlay": pil_to_base64(overlay_pil),
        "classes": [
            {"name": name, "color": [r, g, b]}
            for name, (r, g, b) in zip(CLASS_NAMES, PALETTE_RGB)
        ],
    })
