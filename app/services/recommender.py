from flask import Blueprint, jsonify, request
from io import BytesIO
import math
from typing import Tuple, List, Dict, Optional
import numpy as np
import requests
import torch
from PIL import Image
from sqlalchemy.sql import text
from app.database import SessionLocal
from rembg import remove

recommender_bp = Blueprint('recommender', __name__)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    from transformers import CLIPProcessor, CLIPModel
    _MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _MODEL.eval().to(_DEVICE)
    print("[recommender] CLIP model loaded successfully")
except ImportError as e:
    print(f"[recommender] Error loading CLIP model: {e}")
    raise Exception("Please install the transformers library: pip install transformers")
except Exception as e:
    print(f"[recommender] Unexpected error loading CLIP model: {e}")
    raise

EPS = 1e-8

def _finite(x: Optional[float], default: float = 0.0) -> float:
    try:
        f = float(x)
        return f if math.isfinite(f) else default
    except Exception:
        return default

def _embed_image(url_or_path: str) -> np.ndarray:
    try:
        if url_or_path.startswith("http"):
            r = requests.get(url_or_path, timeout=15)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
        else:
            img = Image.open(url_or_path).convert("RGB")
        inputs = _PROCESSOR(images=img, return_tensors="pt").to(_DEVICE)
        with torch.no_grad():
            feats = _MODEL.get_image_features(**inputs)
            norm = torch.linalg.norm(feats, dim=-1, keepdim=True).clamp_min(EPS)
            vec = (feats / norm).detach().cpu().numpy().astype(np.float32).ravel()
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            if vec.shape[0] != 512:
                raise ValueError(f"Expected 512 dims, got {vec.shape[0]}")
            print(f"[recommender] Generated embedding for {url_or_path}, norm: {np.linalg.norm(vec):.4f}, first 5 values: {vec[:5]}, all zeros: {np.all(vec == 0)}")
            return vec
    except Exception as e:
        print(f"[recommender] Embedding error for {url_or_path}: {e}")
        return np.zeros(512, dtype=np.float32)

def _create_cutout(url_or_path: str) -> str:
    try:
        if url_or_path.startswith("http"):
            r = requests.get(url_or_path, timeout=15)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
        else:
            img = Image.open(url_or_path).convert("RGB")
        cutout = remove(img)
        cutout_path = "cutout.png"  # Placeholder
        cutout.save(cutout_path)
        return cutout_path
    except Exception as e:
        print(f"[matting] Error creating cutout for {url_or_path}: {e}")
        return ""

@recommender_bp.route('/widget/recommendations', methods=['GET'])
def get_recommendations_route():
    room_photo_url = request.args.get('room_photo_url')
    if not room_photo_url:
        return jsonify({'error': 'Missing room_photo_url'}), 400
    filters = {}
    limit = int(request.args.get('limit', 20))
    cursor = request.args.get('cursor', None)
    try:
        items, next_cursor = get_recommendations(room_photo_url, filters, cursor, limit)
        return jsonify({
            'recommendations': items,
            'next_cursor': next_cursor
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@recommender_bp.route('/matting/preview', methods=['GET'])
def preview_matting():
    image_url = request.args.get('image_url')
    if not image_url:
        return jsonify({'error': 'Missing image_url'}), 400
    cutout_url = _create_cutout(image_url)
    return jsonify({'image_url': image_url, 'cutout_url': cutout_url})

@recommender_bp.route('/analytics/event', methods=['POST'])
def track_event():
    event = request.json.get('event')
    return jsonify({"status": f"Tracked {event}"})

def get_recommendations(
    room_photo_url: str,
    filters: Dict,
    cursor: Optional[str] = None,
    limit: int = 20,
) -> Tuple[List[Dict], Optional[str]]:
    q = _embed_image(room_photo_url)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    q_vec = "[" + ",".join(f"{v:.7f}" for v in q) + "]"
    print(f"[recommender] Query vector sample: {q_vec[:100]}...")
    try:
        offset = int(cursor) if cursor is not None else 0
        if offset < 0:
            offset = 0
    except Exception:
        offset = 0
    sql = text(
        """
        SELECT
            id,
            (embedding <=> (:room_emb)::vector(512)) AS cos_dist,
            1.0 - (embedding <=> (:room_emb)::vector(512)) AS raw_sim
        FROM variants
        WHERE embedding IS NOT NULL
          AND inventory_quantity > 0
          AND dims_parsed = TRUE
        ORDER BY cos_dist ASC
        LIMIT :limit OFFSET :offset
        """
    )
    params = {"room_emb": q_vec, "limit": int(limit), "offset": int(offset)}
    with SessionLocal() as db:
        rows = db.execute(sql, params).fetchall()
        if not rows:
            return [], None
        dists = [_finite(r[1], 2.0) for r in rows]
        d_min = min(dists)
        d_max = max(dists)
        span = max(1e-9, d_max - d_min)
        items = []
        for (vid, cos_dist, raw_sim) in rows:
            d = _finite(cos_dist, 0.0)
            sim = _finite(raw_sim, 1.0)
            score = 1.0 - ((d - d_min) / span)
            score = 0.0 if score < 0 else (1.0 if score > 1 else score)
            items.append({
                "variant_id": str(vid),
                "distance": float(d),
                "raw_sim": float(sim),
                "score": float(score)
            })
        next_cursor = str(offset + limit) if len(rows) == limit else None
        return items, next_cursor