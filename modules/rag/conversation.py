import copy
from uuid import uuid4
from loguru import logger
from modules.retrievers.base import BaseContextualRetriever
from modules.llm.openai import OpenAILLM
from modules.types.rag import BaseRAGResponse
from modules.rag.base import BaseRAG
from typing import Union, List, Dict, Any, Optional, Generator

class ConversationRAG(BaseRAG):
    def __init__(self, llm: OpenAILLM, context_retriever: BaseContextualRetriever):
        self.llm = llm
        self.retriever = context_retriever
        self.default_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
        )

    def _build_system_prompt_with_documents_and_messages(
        self, 
        documents: Union[List[Dict[str,str]], str], 
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> str:
        if system_prompt is None:
            system_prompt = self.default_system_prompt

        # Handle different types of documents, convert to string
        if isinstance(documents, List):
            try:
                documents = "\n\n".join([f"Document {index}: \n{result}" for index, result in enumerate(documents['documents'])])
            except Exception as e:
                raise f"Unsupported document format: {e}"
            
        messages_to_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        prompt_template = (
        "{system_prompt}"
        "\n\n"
        "<chat_history>"
        "\n\n"
        "{messages}"
        "\n\n"
        "<context>"
        "\n\n"
        "{documents}"
        )
        
        return prompt_template.format(
            system_prompt=system_prompt,
            documents=documents,
            messages=messages_to_str
        )
    
    def _build_query_prompt_with_documents_and_messages(
        self,
        query: str,
        documents: Union[List[Dict[str,str]], str],
        messages: List[Dict[str, Any]],
    ) -> str:
        messages_to_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt_template = (
            "<chat_history>"
            "\n\n"
            "{messages}"
            "\n\n"
            "<context>"
            "\n\n{documents}"
            "\n\n<Query>"
            "\n\nQuestion: {query}"
        )
        return prompt_template.format(
            messages=messages_to_str,
            documents=documents,
            query=query
        )
    
    def invoke(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        retrieve_result = self.retriever.invoke_format_to_str(
            query=query
        )
        documents = retrieve_result.get('result')
        system_prompt = self._build_system_prompt_with_documents_and_messages(
            documents=documents,
            messages=self.retriever.context_messages,
            system_prompt=system_prompt if system_prompt is not None else None
        )
        logger.info(f"ConversationRAG's system prompt: {system_prompt}")
        
        # 在ConversationRAG中，messages是不包含query的，所以这里需要将query添加到messages中
        # deepcopy是为了防止messages被修改
        messages = copy.deepcopy(self.retriever.context_messages)
        messages.append({"role": "user", "content": query})

        return BaseRAGResponse(
            response_id=str(uuid4()),
            answer = self.llm.invoke(
                messages=messages,
                stream=stream,
            ),
            source_documents = retrieve_result
        )
    
    def invoke_with_wrapped_prompt(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        retrieve_result = self.retriever.invoke_format_to_str(
            query=query,
            messages=messages,
        )
        documents = retrieve_result.get('result')

        if system_prompt is None:
            system_prompt = self.default_system_prompt
        
        prompt = self._build_query_prompt_with_documents_and_messages(
            query=query,
            documents=documents,
            messages=messages,
        )
        logger.info(f"Prompt is wrapped, actual prompt: {prompt}")

        # 在ConversationRAG中，messages是不包含query的，所以这里需要将query添加到messages中
        messages.append({"role": "user", "content": query})

        # 如果messages中包含system prompt且与self.default_system_prompt不同，则将其移除
        for i, message in enumerate(messages):
            if message["role"] == "system" and message["content"] != self.default_system_prompt:
                del messages[i]
        # 如果messages中不包含system prompt，则添加
        if not any(message["role"] == "system" for message in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        return BaseRAGResponse(
            response_id=str(uuid4()),
            answer = self.llm.invoke(
                messages=messages,
                stream=stream,
            ),
            source_documents = retrieve_result
        )