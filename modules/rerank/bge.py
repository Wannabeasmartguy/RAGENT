import os
from typing import List, Dict, Union, Optional, Sequence
from sentence_transformers import CrossEncoder
from huggingface_hub import snapshot_download
from modules.types.document import Document
from modules.rerank.base import BaseDocumentCompressor


class BgeRerank(BaseDocumentCompressor):
    '''
    Bge Rerank, typically used after similar search
    '''
    model_name:str = 'bge-reranker-large'  
    """Model name to use for reranking.""" 
    top_n: int = 10   
    """Number of documents to return."""
    # model:CrossEncoder = CrossEncoder(os.path.join('embedding model',model_name))
    model:CrossEncoder = None
    """CrossEncoder instance to use for reranking."""

    def __init__(self, model_name: Optional[str] = None, top_n: Optional[int] = None):
        super().__init__()
        if model_name is not None:
            self.model_name = model_name
        if top_n is not None:
            self.top_n = top_n
        self.define_model()

    def define_model(self):
        model_path = os.path.join('embeddings',self.model_name)
        try:
            self.model:CrossEncoder = CrossEncoder(model_name=model_path)
        except:
            snapshot_download(repo_id="BAAI/"+self.model_name,
                              local_dir=model_path)
            self.model:CrossEncoder = CrossEncoder(model_name=model_path)

    def bge_rerank(self,query,docs):
        model_inputs =  [[query, doc] for doc in docs]
        scores = self.model.predict(model_inputs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[:self.top_n]

    class Config:
        """Configuration for this pydantic object."""

        extra = "forbid"
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Union[Sequence[Document], List[Dict[str, Union[str, Dict]]]],
        query: str,
        callbacks = None,
    ) -> Sequence[Document] | List[Dict[str, Union[str, Dict]]]:
        """
        Compress documents using BAAI/bge-reranker models.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        try:
            _docs = [d.page_content for d in doc_list]
        except:
            _docs = [d["page_content"] for d in doc_list]
        results = self.bge_rerank(query, _docs)
        final_results = []
        for r in results:
            doc = doc_list[r[0]]
            try:
                doc.metadata["relevance_score"] = r[1]
            except:
                doc["metadata"]["relevance_score"] = r[1]
            final_results.append(doc)
        return final_results
    