import logging
from dotenv import load_dotenv

load_dotenv()

from fastapi import APIRouter
import datetime

import app.services.rag_chat_service.src.bootstrap as bootstrap
from app.services.rag_chat_service.src.chat_service import ChatService
from app.model.chat_request import ChatRequest
from app.model.chat_response import ChatResponse

FEEDBACK_TABLENAME = "chatbot_log"

router = APIRouter()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@router.post("/chat")
def rag_chat(request: ChatRequest):
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        logger.info("Load ChatService")
        chat_service = ChatService()
        response = chat_service.run_RAG(request.message)

        neon_db = bootstrap.get_resource("db")
        logger.info("Save user message and chatbot response")
        neon_db.insert_feedback_to_table(tablename=FEEDBACK_TABLENAME,
                                         datetime=datetime_str,
                                         user_query=request.message,
                                         response=response,
                                         user_feedback=None)
        neon_db.commit()

        return ChatResponse(response_status="success",
                            user_message=request.message,
                            model_response=response,
                            timestamp=datetime_str,
                            feedback=None)

    except Exception as e:
        logger.error(f"Error in /chat {e}", exc_info=True)
        return ChatResponse(response_status="failed",
                            user_message=request.message,
                            model_response="NULL",
                            timestamp=datetime_str,
                            feedback=None)

@router.patch("/chat-feedback")
def rag_feedback(user_feedback: int):
    try:
        logger.info("Save user feedback")
        neon_db = bootstrap.get_resource("db")
        neon_db.cur.execute(f"""
            UPDATE {FEEDBACK_TABLENAME}
            SET user_feedback = {user_feedback}
            WHERE id = (SELECT MAX(id) FROM {FEEDBACK_TABLENAME})
        """)
        neon_db.commit()

    except Exception as e:
        logger.error(f"Error in /chat-feedback {e}", exc_info=True)
