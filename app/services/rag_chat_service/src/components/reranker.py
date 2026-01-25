from typing import List
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
import asyncio

from app.services.rag_chat_service.src.utils.syncify import sync
import app.services.rag_chat_service.src.bootstrap as bootstrap

class Reranker:
    k_rerank: int = 3
    model_rerank: str = None
    model_rerank_provider: str = None
    temperature_rerank: float = 0
    top_k_rerank: int = 1
    top_p_rerank: float = 0.9
    prompt_name: str
    prompt_version: str

    def __init__(self,
                 k_rerank: int,
                 model_rerank: str,
                 model_rerank_provider: str,
                 temperature_rerank: float,
                 top_k_rerank: int,
                 top_p_rerank: float,
                 prompt_name: str,
                 prompt_version: str):
        self.k_rerank = k_rerank
        self.rerank_llm = init_chat_model(model=model_rerank,
                                          model_provider=model_rerank_provider,
                                          temperature=temperature_rerank,
                                          top_k=top_k_rerank,
                                          top_p=top_p_rerank)
        self.prompt_loader = bootstrap.get_resource("prompt_loader")
        self._setup_prompt(prompt_name, prompt_version)

    def _setup_prompt(self, prompt_name, prompt_version):
        prompt = self.prompt_loader(prompt_name, prompt_version)
        self.reranker_prompt = prompt.format_messages()[0].content

    @sync
    async def rerank(self, retrieved_docs: List[Document],
                     query: str) -> List[Document]:

        async def score_doc(doc: Document) -> tuple[int, Document]:
            try:
                response = await self.rerank_llm.ainvoke(self.reranker_prompt.format(query, doc.page_content))
                score = int(response.content)
            except:
                score = 0
            return (score, doc)

        ranked_tasks = [score_doc(doc) for doc in retrieved_docs]
        ranked = await asyncio.gather(*ranked_tasks)
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:self.k_rerank]]

if __name__ == "__main__":
    from app.services.rag_chat_service.src.components.retriever import PostgresRetriever
    import app.services.rag_chat_service.src.bootstrap as bootstrap

    import os
    import asyncio

    asyncio.run(bootstrap.initialize_resources())

    retriever_param = {
        "tablename": "faiss_index_google_genai_risk_mgmt",
        "embedding_model_name": "all-MiniLM-L6-v2"
    }
    reranker_param = {
        "k_rerank": 5,  # choose top k reranker score
        "model_rerank": "gemini-2.5-flash",
        "model_rerank_provider": "google_genai",
        "temperature_rerank": 0,
        "top_k_rerank": 5,  # top k token in generation of the rerank score
        "top_p_rerank": 0.8,
        "prompt_name": "system-reranker-prompt",
        "prompt_version": "latest"
    }

    test_query = "What are the Socio-technical Considerations?"

    retriever = PostgresRetriever(**retriever_param)
    top_k_retrieved = retriever.retrieve(query=test_query,
                                         k_retrieval=5)

    reranker = Reranker(**reranker_param)
    context = reranker.rerank(retrieved_docs=top_k_retrieved,
                              query=test_query)
    print(context)
