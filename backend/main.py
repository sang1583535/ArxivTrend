from fastapi import FastAPI

from backend.api.routes import router as api_router

app = FastAPI(title="ArxivTrend API", version="0.0.1")
app.include_router(api_router)