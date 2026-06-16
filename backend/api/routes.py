from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}