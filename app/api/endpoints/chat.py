import logging
from dotenv import load_dotenv

load_dotenv()

from fastapi import APIRouter

from app.services.rag_chat_service.src.chat_service import ChatService
from app.model.chat_request import ChatRequest

router = APIRouter()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@router.post("/chat")
def rag_chat(request: ChatRequest):
    try:
        logger.info("Load ChatService")
        chat_service = ChatService()
        response = chat_service.run_RAG(request.message)

        return {
            "message": "RAG successfully ran",
            "output": response,
            "status": "completed"
        }

    except Exception as e:
        logger.error(f"Error in /chat {e}", exc_info=True)
        return {
            "message": "RAG run failed",
            "output": "N/A",
            "status": "failed"
        }