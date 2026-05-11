from langchain_core.runnables.base import RunnableSerializable

from app.services.rag_chat_service.src.utils.chainer import chain_from_yaml
from app.services.rag_chat_service.src.utils.syncify import sync

class ChainRunner:
    def __init__(self, config_path: str):
        self._load_chain(config_path=config_path)

    def _load_chain(self, config_path: str) -> RunnableSerializable:
        self.rag_chain = chain_from_yaml(config_path)

    @sync
    async def run(self, user_query: str) -> str:
        response = await self.rag_chain.ainvoke(user_query)
        return response


if __name__ == "__main__":
    from dotenv import load_dotenv
    from langsmith import Client
    import asyncio
    from app.services.rag_chat_service.src.bootstrap import initialize_resources

    load_dotenv()
    client = Client()
    asyncio.run(initialize_resources())

    runner = ChainRunner(config_path="app/services/rag_chat_service/src/pipeline/inference_pipeline_config.yaml")
    response = runner.run("How to mitigate risks in GenAI?")
    print(response)