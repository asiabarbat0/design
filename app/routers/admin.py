from fastapi import APIRouter

router = APIRouter()

@router.get("/test")
async def test_admin():
    return {"message": "Admin endpoint"}