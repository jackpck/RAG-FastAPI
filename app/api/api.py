from fastapi import APIRouter
from app.api.endpoints import chat, home

api_router = APIRouter()

api_router.include_router(chat.router)
api_router.include_router(home.router)
