from typing import List, Dict, Any, Optional, Callable
from modules.retrievers.base import BaseRetriever
import asyncio


class ContextualCompressionRetriever(BaseRetriever):
    """Retriever that wraps a base retriever and compresses the results."""

    def __init__(
        self,
        base_compressor: Any,
        base_retriever: Any,
    ):
        self.base_compressor = base_compressor
        self.base_retriever = base_retriever

    def invoke(self, query: str) -> List[Dict[str, Any]]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            Sequence of relevant documents
        """
        docs = self.base_retriever.invoke(query)
        if docs:
            compressed_docs = self.base_compressor.compress_documents(docs, query)
            return list(compressed_docs)
        else:
            return []

    def invoke_format_to_str(self, query: str) -> str:
        """将处理过（一般是重排序）的结果格式化为字符串，其中invoke的结果为dict，包含'page_content'字段"""
        results = self.invoke(query)
        results_str = "\n\n".join(
            [f"Document {i+1}:\n{doc['page_content']}" for i, doc in enumerate(results)]
        )
        page_content = [doc["page_content"] for doc in results]
        metadatas = [doc["metadatas"] for doc in results if "metadats" in doc]
        return dict(result=results_str, page_content=page_content, metadatas=metadatas)

    async def ainvoke(self, query: str) -> List[Dict[str, Any]]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        docs = await self.base_retriever.ainvoke(query)
        if docs:
            compressed_docs = await self.base_compressor.acompress_documents(
                docs, query
            )
            return list(compressed_docs)
        else:
            return []


if __name__ == "__main__":

    class BaseCompressor:
        def compress_documents(
            self, docs: List[Dict[str, Any]], query: str
        ) -> List[Dict[str, Any]]:
            # 实现文档压缩逻辑
            return docs

        async def acompress_documents(
            self, docs: List[Dict[str, Any]], query: str
        ) -> List[Dict[str, Any]]:
            # 实现异步文档压缩逻辑
            return docs

    class BaseRetriever:
        def invoke(self, query: str) -> List[Dict[str, Any]]:
            # 实现文档检索逻辑
            return [{"page_content": "doc1 content", "metadata": {"source": "source1"}}]

        async def ainvoke(self, query: str) -> List[Dict[str, Any]]:
            # 实现异步文档检索逻辑
            return [{"page_content": "doc1 content", "metadata": {"source": "source1"}}]

    # 创建 BaseCompressor 和 BaseRetriever 实例
    base_compressor = BaseCompressor()
    base_retriever = BaseRetriever()

    # 创建 ContextualCompressionRetriever 实例
    contextual_compression_retriever = ContextualCompressionRetriever(
        base_compressor=base_compressor,
        base_retriever=base_retriever,
    )

    # 同步调用
    result = contextual_compression_retriever.invoke("query")
    print(result)

    # 异步调用
    async def test_async():
        result = await contextual_compression_retriever.ainvoke("query")
        print(result)

    test_loop = asyncio.get_running_loop()

    # 运行异步调用
    if test_loop is not None:
        asyncio.run_coroutine_threadsafe(test_async(), test_loop)
    else:
        # 否则，使用 asyncio.run()
        asyncio.run(test_async())
