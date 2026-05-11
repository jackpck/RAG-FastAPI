import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

from app.api.api import api_router
from app.services.rag_chat_service.src.bootstrap import initialize_resources
from langsmith import Client


load_dotenv()
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await initialize_resources()

app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="localhost", port=8000, reload=True)

