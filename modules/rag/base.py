from abc import ABC, abstractmethod


class BaseRAG(ABC):
    @abstractmethod
    def invoke(self, *args, **kwargs):
        pass