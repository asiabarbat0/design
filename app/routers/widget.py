from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.recommender import get_recommendations
from app.services.analytics import log_event
from app.config import S3_BUCKET
import boto3, os
from uuid import uuid4
from urllib.parse import quote_plus
from botocore.config import Config

router = APIRouter()

# --- Resolve bucket region (us-east-1 returns None) ---
_meta = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
try:
    loc = _meta.get_bucket_location(Bucket=S3_BUCKET)
    BUCKET_REGION = loc.get("LocationConstraint") or "us-east-2"
except Exception:
    BUCKET_REGION = os.getenv("AWS_REGION") or "us-east-2"

# --- FORCE Signature V4 + virtual-hosted style (prevents SigV2 URLs) ---
cfg = Config(
    signature_version="s3v4",
    s3={"addressing_style": "virtual"}  # https://<bucket>.s3.<region>.amazonaws.com
)

s3 = boto3.client(
    "s3",
    region_name=BUCKET_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    config=cfg,
)

@router.post("/upload-room")
async def upload_room(file: UploadFile = File(...)):
    """
    Uploads to PRIVATE S3 and returns a presigned GET URL (SigV4).
    """
    try:
        key = f"rooms/{uuid4()}/{quote_plus(file.filename)}"
        s3.upload_fileobj(
            Fileobj=file.file,
            Bucket=S3_BUCKET,
            Key=key,
            ExtraArgs={"ContentType": file.content_type or "application/octet-stream"},
        )
        # Return a SigV4 presigned GET (contains X-Amz-Algorithm=AWS4-HMAC-SHA256)
        photo_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=3600,
        )
        # Optional sanity check: assert SigV4 params are present
        if "X-Amz-Algorithm=AWS4-HMAC-SHA256" not in photo_url:
            raise RuntimeError("expected SigV4 presigned URL")

        log_event("upload", {"key": key, "photo_url": photo_url, "region": BUCKET_REGION})
        return {"key": key, "photo_url": photo_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upload_failed: {e}")

@router.get("/recommendations")
async def recs(photo_url: str, cursor: str | None = None, fits_only: bool = True):
    recs_list, next_cursor = get_recommendations(photo_url, {"fits_only": fits_only}, cursor)
    log_event("recs_shown", {"count": len(recs_list)})
    return {"recommendations": recs_list, "next_cursor": next_cursor}
