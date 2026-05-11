import time
import os
from dotenv import load_dotenv

load_dotenv()

from app.services.rag_chat_service.src.components.runner import ChainRunner
from app.services.rag_chat_service.src.base_service import BaseService

PIPELINE_PATH = os.environ.get("PIPELINE_PATH")

class ChatService(BaseService):
    def __init__(self):
        try:
            super().__init__()
            self.logger.info("ChatService.__init__ starting")
        except Exception as e:
            logger.error(f"{e}", exc_info=True)

    def run_RAG(self,
                user_query: str) -> str:
        starttime = time.time()
        RAG_chain = ChainRunner(config_path=PIPELINE_PATH)
        response = RAG_chain.run(user_query)
        endtime = time.time()
        return f"Response:\n{response['result'].content}\n\nLatency: {endtime - starttime:.4f} seconds"


if __name__ == "__main__":
    from langsmith import Client
    from app.services.rag_chat_service.src.bootstrap import initialize_resources
    import asyncio

    client = Client()
    asyncio.run(initialize_resources())

    user_query = "What are the top risks in GenAI?"

    chat_service = ChatService()
    print(chat_service.run_RAG(user_query))


