from langchain_core.documents import Document
from typing import List
from langchain.chat_models import init_chat_model
import asyncio

from app.services.rag_chat_service.src.utils.syncify import sync
import app.services.rag_chat_service.src.bootstrap as bootstrap

class Autorater:
    """
    determine if the reranked documents are sufficient to answer the query question
    in order to mitigate giving answer to an un-answerable question (i.e. to abstain)
    """
    def __init__(self,
                 model_autorate: str,
                 model_autorate_provider: str,
                 temperature_autorate: float,
                 top_k_autorate: int,
                 top_p_autorate: float,
                 prompt_name: str,
                 prompt_version: str):
        self.autorater_llm = init_chat_model(model=model_autorate,
                                             model_provider=model_autorate_provider,
                                             temperature=temperature_autorate,
                                             top_k=top_k_autorate,
                                             top_p=top_p_autorate)
        self.prompt_loader = bootstrap.get_resource("prompt_loader")
        self._setup_prompt(prompt_name, prompt_version)

    def _setup_prompt(self, prompt_name, prompt_version):
        prompt = self.prompt_loader(prompt_name, prompt_version)
        self.autorater_prompt = prompt.format_messages()[0].content

    @sync
    async def autorate(self,
                 reranked_document: List[Document],
                 query: str) -> List[Document]:

        async def score_doc(query, doc):
            try:
                response = await self.autorater_llm.ainvoke(self.autorater_prompt.format(query, doc.page_content))
                score = int(response.content)
            except:
                score = 0
            return (score, doc)

        rated_tasks = [score_doc(query, doc) for doc in reranked_document]
        rated = await asyncio.gather(*rated_tasks)
        return [d[1] for d in rated if d[0] == 1]


if __name__ == "__main__":
    import app.services.rag_chat_service.src.bootstrap as bootstrap

    import asyncio

    asyncio.run(bootstrap.initialize_resources())

    test_doc_str = ["the apple is in the box",
                    "the banana is on the tree",
                    "the apple is red"]
    test_doc = list(map(lambda s: Document(page_content=s), test_doc_str))
    print(test_doc)
    test_query = Document(page_content="where can I find the apple?")

    model_config = {
        "model_autorate": "gemini-2.5-flash",
        "model_autorate_provider": "google_genai",
        "temperature_autorate": 0,
        "top_k_autorate": 5,
        "top_p_autorate": 0.8,
        "prompt_name": "system-autorater-prompt",
        "prompt_version": "latest"
    }

    autorater = Autorater(**model_config)
    context = autorater.autorate(reranked_document=test_doc, query=test_query)
    print(context)




