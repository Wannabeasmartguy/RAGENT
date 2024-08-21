import asyncio
from typing import List, Dict, Any, Optional
from collections import defaultdict
from modules.retrievers.base import BaseRetriever

class EnsembleRetriever(BaseRetriever):
    """Base class for all ensemble retrievers"""
    def __init__(self, retrievers: List[BaseRetriever], weights: Optional[List[float]] = None, c: int = 60):
        self.retrievers = retrievers
        self.weights = weights if weights else [1 / len(retrievers)] * len(retrievers)
        self.c = c

    # 如果传入了contextual_retriever，需要继承它的实例属性context_messages
    @property
    def context_messages(self):
        # 遍历所有retriever，如果有任何一个retriever有context_messages属性，则返回其context_messages
        for retriever in self.retrievers:
            if hasattr(retriever, 'context_messages'):
                return retriever.context_messages
        return None  # 如果没有找到任何retriever有context_messages属性，则返回None

    def invoke(self, query: str) -> List[Dict[str, Any]]:
        return self.rank_fusion(query)
    
    def invoke_format_to_str(self, query: str) -> str:
        """将包含多个检索器的结果格式化为字符串，其中invoke的结果为dict，包含'page_content'字段"""
        results = self.invoke(query)
        results_str = "\n\n".join([f"Document {i+1}:\n{doc['page_content']}" for i, doc in enumerate(results)])
        page_content = [doc['page_content'] for doc in results]
        metadatas = [doc['metadatas'] for doc in results if 'metadats' in doc]
        return dict(result=results_str, page_content=page_content, metadatas=metadatas)

    async def ainvoke(self, query: str) -> List[Dict[str, Any]]:
        return await self.arank_fusion(query)

    def rank_fusion(self, query: str) -> List[Dict[str, Any]]:
        retriever_docs = [retriever.invoke(query) for retriever in self.retrievers]
        return self.weighted_reciprocal_rank(retriever_docs)

    async def arank_fusion(self, query: str) -> List[Dict[str, Any]]:
        retriever_docs = await asyncio.gather(*[retriever.ainvoke(query) for retriever in self.retrievers])
        return self.weighted_reciprocal_rank(retriever_docs)

    def weighted_reciprocal_rank(self, doc_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        rrf_score = defaultdict(float)
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[doc['page_content']] += weight / (rank + self.c)

        all_docs = [doc for doc_list in doc_lists for doc in doc_list]
        sorted_docs = sorted(
            {doc['page_content']: doc for doc in all_docs}.values(),
            key=lambda doc: rrf_score[doc['page_content']],
            reverse=True
        )
        return sorted_docs


if __name__ == '__main__':
    # 测试
    
    class BaseRetriever:
        """Base class for all retrievers"""
        def __init__(self, docs: List[str]):
            self.docs = [{'content': doc} for doc in docs]

        def invoke(self, query: str) -> List[Dict[str, Any]]:
            return self.docs
        
        async def ainvoke(self, query: str) -> List[Dict[str, Any]]:
            return self.docs
    
    retriever1 = BaseRetriever(['doc1', 'doc2'])
    retriever2 = BaseRetriever(['doc2', 'doc3'])
    ensemble_retriever = EnsembleRetriever([retriever1, retriever2])

    # 同步调用
    print(ensemble_retriever.invoke('tes'))

    # 异步调用
    async def test_async():
        result = await ensemble_retriever.ainvoke('test')
        print(result)

    loop = asyncio.get_running_loop()

    if loop is not None:
        # 如果存在，使用 run_coroutine_threadsafe() 将异步函数运行在不同的线程中
        asyncio.run_coroutine_threadsafe(test_async(), loop)
    else:
        # 否则，使用 asyncio.run()
        asyncio.run(test_async())