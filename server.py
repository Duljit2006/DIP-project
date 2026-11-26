"""
server.py - NeuroDCT Research System FastAPI Web Server
------------------------------------------------------
Serves the responsive 3-stage Swiss UI (web/index.html) and provides
real-time GPU-accelerated JPEG artifact restoration endpoints powered by
FrequencyGNN + SpatialRefinerCNN.
"""

import os
import io
import sys
import uuid
import socket
import base64
from collections import OrderedDict
from typing import Optional

import qrcode

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference import NeuroDCTEngine, PRESETS, rgb_float_to_png_b64

app = FastAPI(title="NeuroDCT Research System", version="3.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount results directory if it exists so Benchmarks tab can showcase saved comparisons
if os.path.exists("results"):
    app.mount("/results", StaticFiles(directory="results"), name="results")

# Initialize single model engine cleanly at startup
engine = NeuroDCTEngine()

# Bounded LRU cache for exportable evaluation runs
RUN_CACHE: OrderedDict = OrderedDict()
MAX_CACHED_RUNS = 12


def _cache_run(run_id: str, payload: dict):
    RUN_CACHE[run_id] = payload
    while len(RUN_CACHE) > MAX_CACHED_RUNS:
        RUN_CACHE.popitem(last=False)


def _get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html_path = os.path.join("web", "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>NeuroDCT Web UI not found at web/index.html</h1>", status_code=404)
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/health")
async def health():
    safe_info = {
        k: (str(v) if k == "device" else v)
        for k, v in engine.device_info.items()
    }
    return {
        "status": "ready",
        "device": str(engine.device),
        "device_info": safe_info,
        "gnn_loaded": engine.gnn_model is not None,
        "cnn_loaded": engine.cnn_model is not None,
        "lan_url": f"http://{_get_lan_ip()}:8000",
    }


@app.get("/api/presets")
async def list_presets():
    items = []
    for key, info in PRESETS.items():
        path = info["path"]
        exists = os.path.exists(path)
        size_kb = round(os.path.getsize(path) / 1024.0, 2) if exists else 0.0
        items.append({
            "id": info["id"],
            "name": info["name"],
            "filename": info["filename"],
            "tag": info["tag"],
            "desc": info["desc"],
            "size_kb": size_kb,
            "url": f"/api/preset_image/{key}",
            "available": exists,
        })
    return {"presets": items}


@app.get("/api/preset_image/{preset_id}")
async def get_preset_image(preset_id: str):
    if preset_id not in PRESETS:
        raise HTTPException(status_code=404, detail="Unknown preset ID")
    path = PRESETS[preset_id]["path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Preset image file missing on disk")
    with open(path, "rb") as f:
        return Response(content=f.read(), media_type="image/png")


@app.post("/api/preview_meta")
async def preview_meta(
    file: Optional[UploadFile] = File(None),
    preset_id: Optional[str] = Form(None),
):
    """Inspects an uploaded file or preset, returning prepared dimensions and preview URI."""
    try:
        if file is not None:
            raw_bytes = await file.read()
            filename = file.filename or "custom_upload.png"
        elif preset_id and preset_id in PRESETS:
            with open(PRESETS[preset_id]["path"], "rb") as f:
                raw_bytes = f.read()
            filename = PRESETS[preset_id]["filename"]
        else:
            raise HTTPException(status_code=400, detail="Either file or preset_id must be provided.")

        prep = engine._prepare_image(raw_bytes)
        w_crop, h_crop = prep["shape"][1], prep["shape"][0]
        return {
            "status": "ok",
            "filename": filename,
            "width": w_crop,
            "height": h_crop,
            "orig_width": prep["orig_shape"][1],
            "orig_height": prep["orig_shape"][0],
            "resized": prep["resized"],
            "buffer_kb": round(len(raw_bytes) / 1024.0, 2),
            "preview_b64": rgb_float_to_png_b64(prep["rgb"]),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/evaluate")
async def evaluate_endpoint(
    file: Optional[UploadFile] = File(None),
    preset_id: Optional[str] = Form(None),
    qf: int = Form(15),
    arch: str = Form("cnn_gnn"),
    eq6_lock: bool = Form(True),
    mode: str = Form("simulation"),
    roi_x: Optional[int] = Form(None),
    roi_y: Optional[int] = Form(None),
):
    """Executes all 5 restoration pipelines on the provided image and returns full telemetry."""
    try:
        if file is not None:
            image_bytes = await file.read()
            sample_name = file.filename or "uploaded_image.png"
        elif preset_id and preset_id in PRESETS:
            path = PRESETS[preset_id]["path"]
            with open(path, "rb") as f:
                image_bytes = f.read()
            sample_name = PRESETS[preset_id]["filename"]
        else:
            raise HTTPException(status_code=400, detail="No image or preset supplied.")

        result = engine.evaluate_all_pipelines(
            image_bytes=image_bytes,
            qf=qf,
            arch=arch,
            eq6_lock=eq6_lock,
            mode=mode,
            roi_x=roi_x,
            roi_y=roi_y,
        )
        run_id = uuid.uuid4().hex[:8].upper()
        result["run_id"] = run_id
        result["sample_name"] = sample_name
        _cache_run(run_id, result)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/export/{run_id}/json")
async def export_json(run_id: str):
    if run_id not in RUN_CACHE:
        raise HTTPException(status_code=404, detail="Run expired or not found")
    data = RUN_CACHE[run_id]
    # Strip heavy base64 strings for clean quantitative JSON export
    clean_export = {
        "run_id": data["run_id"],
        "sample_name": data["sample_name"],
        "mode": data["mode"],
        "qf": data["qf"],
        "cutoff": data["cutoff"],
        "eq6_lock": data["eq6_lock"],
        "selected_arch": data["selected_arch"],
        "shape": data["shape"],
        "device": data["device"],
        "total_time_ms": data["total_time_ms"],
        "selected_metrics": {k: v for k, v in data["selected"].items() if k != "image_b64"},
        "all_methods": [{k: v for k, v in m.items() if k != "image_b64"} for m in data["methods"]],
    }
    return JSONResponse(
        content=clean_export,
        headers={"Content-Disposition": f'attachment; filename="NeuroDCT_Run_{run_id}.json"'},
    )


@app.get("/api/export/{run_id}/png")
async def export_png(run_id: str, arch: Optional[str] = None):
    if run_id not in RUN_CACHE:
        raise HTTPException(status_code=404, detail="Run expired or not found")
    data = RUN_CACHE[run_id]
    target_b64 = data["selected"]["image_b64"]
    if arch:
        for m in data["methods"]:
            if m["key"] == arch:
                target_b64 = m["image_b64"]
                break
    raw_b64 = target_b64.split(",", 1)[1]
    png_bytes = base64.b64decode(raw_b64)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="NeuroDCT_Restored_{run_id}.png"'},
    )


@app.get("/api/benchmarks")
async def get_benchmarks():
    """Returns pre-computed Set5 and Set14 gallery data for the Benchmarks view."""
    set14_gallery = []
    if os.path.exists("results"):
        for fname in sorted(os.listdir("results")):
            if fname.endswith("_comparison.png"):
                base = fname.replace("_comparison.png", "")
                heat = f"{base}_error_heatmap.png"
                set14_gallery.append({
                    "id": base,
                    "comparison_url": f"/results/{fname}",
                    "heatmap_url": f"/results/{heat}" if os.path.exists(os.path.join("results", heat)) else None,
                })

    scores_txt = ""
    scores_path = os.path.join("results", "final_evaluation_scores.txt")
    if os.path.exists(scores_path):
        with open(scores_path, "r") as f:
            scores_txt = f.read()

    # Verified Set5 numbers from Colab training/eval
    set5_summary = [
        {"image": "img_001 (Baby)", "base": 36.41, "gnn": 36.41, "gnn_cnn": 36.72, "cnn_gnn": 37.00, "gain": "+0.59", "ssim": 0.9668},
        {"image": "img_002 (Bird)", "base": 36.14, "gnn": 36.18, "gnn_cnn": 38.30, "cnn_gnn": 38.77, "gain": "+2.63", "ssim": 0.9847},
        {"image": "img_003 (Butterfly)", "base": 26.40, "gnn": 26.40, "gnn_cnn": 30.12, "cnn_gnn": 30.53, "gain": "+4.13", "ssim": 0.9542},
        {"image": "img_004 (Head)", "base": 33.73, "gnn": 33.73, "gnn_cnn": 33.89, "cnn_gnn": 34.06, "gain": "+0.33", "ssim": 0.8756},
        {"image": "img_005 (Woman)", "base": 31.09, "gnn": 31.10, "gnn_cnn": 32.65, "cnn_gnn": 32.94, "gain": "+1.85", "ssim": 0.9671},
    ]

    return {
        "set5_rows": set5_summary,
        "set5_avg": {"base": 32.75, "gnn": 32.76, "gnn_cnn": 34.34, "cnn_gnn": 34.66, "gain": "+1.91", "ssim": 0.9497},
        "set14_gallery": set14_gallery[:6],
        "scores_text": scores_txt,
    }


@app.get("/api/qr.png")
async def generate_qr(url: Optional[str] = None):
    """Generates a clinical indigo-on-white scannable QR code PNG for judges."""
    target_url = url or f"http://{_get_lan_ip()}:8000"
    qr = qrcode.QRCode(version=1, box_size=8, border=2)
    qr.add_data(target_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#141B2B", back_color="#FFFFFF")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("  NeuroDCT Research System Web Server Starting...")
    print(f"  Local URL:  http://localhost:8000")
    print(f"  Mobile LAN: http://{_get_lan_ip()}:8000")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
