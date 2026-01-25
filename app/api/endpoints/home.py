import logging
from fastapi import APIRouter

router = APIRouter()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@router.post("/home")
def app_introduction():
    try:
        return "This is a RAG chatbot. AMA about Generative AI Risk Management in Financial Institutions."
    except Exception as e:
        logger.error(f"Error in /chat {e}", exc_info=True)
        return "Error"

@router.get("/home/{input}")
def read_input(input: str):
    """
    dummy endpoint. Not used in the RAG
    """
    return {"input": input}
