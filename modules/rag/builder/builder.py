from modules.retrievers.base import BaseRetriever
from modules.rag.base import BaseRAG
from modules.rag.basic import BasicRAG
from modules.rag.conversation import ConversationRAG
from modules.llm.openai import OpenAILLM

class RAGBuilder:
    def __init__(self):
        self.llm = None
        self.retriever = None
        self.default_system_prompt = None
        self.rag_type = None

    def with_llm(self, llm: OpenAILLM) -> 'RAGBuilder':
        self.llm = llm
        return self

    def with_retriever(self, retriever: BaseRetriever) -> 'RAGBuilder':
        self.retriever = retriever
        return self

    def with_default_system_prompt(self, prompt: str) -> 'RAGBuilder':
        self.default_system_prompt = prompt
        return self

    def for_rag_type(self, rag_type: str) -> 'RAGBuilder':
        self.rag_type = rag_type
        return self

    def build(self) -> BaseRAG:
        if self.llm is None:
            raise ValueError("LLM must be provided")
        if self.retriever is None:
            raise ValueError("Retriever must be provided")
        if self.rag_type is None:
            raise ValueError("RAG type must be specified")

        if self.rag_type == 'ConversationRAG':
            rag = ConversationRAG(self.llm, self.retriever)
        elif self.rag_type == 'BasicRAG':
            rag = BasicRAG(self.llm, self.retriever)
        else:
            raise ValueError(f"Unknown RAG type: {self.rag_type}")

        if self.default_system_prompt is not None:
            rag.default_system_prompt = self.default_system_prompt

        return rag

# Example usage
# builder = RAGBuilder()
# rag = builder.with_llm(OpenAILLM(...)).with_context_retriever(BaseContextualRetriever(...)).for_rag_type('ConversationRAG').build()