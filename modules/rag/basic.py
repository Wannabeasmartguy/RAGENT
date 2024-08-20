from loguru import logger
from typing import List, Dict, Optional, Union, Any, Generator
from modules.llm.openai import OpenAILLM
from modules.rag.base import BaseRAG
from modules.retrievers.vector.chroma import ChromaRetriever

class BasicRAG(BaseRAG):
    def __init__(self, llm: OpenAILLM, retriever: ChromaRetriever):
        self.llm = llm
        self.retriever = retriever
        self.default_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
        )
    
    def _build_system_prompt_with_documents(
            self, 
            documents: Union[List[Dict[str,str]], str], 
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

        prompt_template = (
        "{system_prompt}"
        "\n\n"
        "<context>"
        "\n\n"
        "{documents}"
        )
        
        return prompt_template.format(
            system_prompt=system_prompt,
            documents=documents
        )
    
    def _build_query_prompt_with_documents(
        self,
        query: str,
        documents: Union[List[Dict[str,str]], str],
    ) -> str:
        prompt_template = (
            "<context>"
            "\n\n{documents}"
            "\n\n<Query>"
            "\n\nQuestion: {query}"
        )
        return prompt_template.format(
            query=query,
            documents=documents,
        )
    
    def invoke(
            self, 
            query: str, 
            system_prompt: Optional[str] = None,
            stream: bool = False
        ) -> Dict[str, Any]:
        retrieve_result = self.retriever.invoke_format_to_str(
            query_texts=[query],
        )
        documents = retrieve_result.get('result')
        system_prompt = self._build_system_prompt_with_documents(
            documents=documents,
            system_prompt=system_prompt if system_prompt is not None else None
        )
        logger.info(f"System prompt: {system_prompt}")
        return dict(
            awswer=self.llm.invoke(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                stream=stream
            ), 
            source_documents = retrieve_result
        )
    
    def invoke_with_wrapped_prompt(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        retrieve_result = self.retriever.invoke_format_to_str(
            query_texts=[query],
        )
        documents = retrieve_result.get('result')
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        prompt = self._build_query_prompt_with_documents(
            query=query,
            documents=documents,
        )
        logger.info(f"Prompt is wrapped, actual prompt: {prompt}")
        return dict(
                answer = self.llm.invoke(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=stream
            ), 
            source_documents = retrieve_result
        )