import os
import logging
from dotenv import load_dotenv
load_dotenv()

from app.services.rag_chat_service.src.utils.sql_connection import NeonPostgres
from app.services.rag_chat_service.src.utils.langsmith_loader import load_prompt

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_resources = {"db": None,
              "prompt_loader": None}

async def initialize_resources():
    db_name = os.environ.get("DB_NAME")
    db_user = os.environ.get("DB_USER")
    db_pwd = os.environ.get("DB_PWD")
    db_host = os.environ.get("DB_HOST")
    db_port = os.environ.get("DB_port")

    _resources["db"] = NeonPostgres(dbname=db_name,
                                    user=db_user,
                                    password=db_pwd,
                                    host=db_host,
                                    port=db_port,
                                    sslmode="require")
    logger.info(f"Load neon database")

    _resources["prompt_loader"] = load_prompt
    logger.info(f"Load prompt loader")

def get_resource(name: str):
    if name not in _resources:
        raise KeyError(f"{name} is not found. Available resources: {list(_resources.keys())}")
    return _resources[name]



