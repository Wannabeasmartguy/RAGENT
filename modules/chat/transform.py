from typing import List, Dict, Tuple, Optional, TypeVar, Generic

T = TypeVar('T')

class ListLimiter(Generic[T]):
    """A generic list limiter that keeps only the most recent elements.
    
    This class can be used to limit any list to a maximum size, keeping the most recent elements.
    It provides validation, transformation, and logging capabilities.
    """

    def __init__(self, max_size: Optional[int] = None):
        """
        Args:
            max_size (Optional[int]): Maximum number of elements to keep. 
                                    Must be greater than 0 if not None.
                                    If None, no limit will be applied.
        
        Raises:
            ValueError: If max_size is less than 1.
        """
        self._validate_max_size(max_size)
        self._max_size = max_size

    def transform(self, items: List[T]) -> List[T]:
        """Limits the list to the specified maximum size.

        Args:
            items (List[T]): The list to be limited.

        Returns:
            List[T]: A new list containing the most recent elements up to max_size.
        """
        if not self._max_size:
            return items
        return items[-self._max_size:]

    def transform_with_logs(self, items: List[T]) -> Tuple[List[T], str]:
        """Transforms the list and returns both the result and a log message.

        Args:
            items (List[T]): The list to be limited.

        Returns:
            Tuple[List[T], str]: A tuple containing the transformed list and a log message.
        """
        result = self.transform(items)
        log_msg = self._generate_log(len(items), len(result))
        return result, log_msg

    @staticmethod
    def _validate_max_size(max_size: Optional[int]) -> None:
        """Validates the max_size parameter.

        Args:
            max_size (Optional[int]): The maximum size to validate.

        Raises:
            ValueError: If max_size is invalid.
        """
        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be None or greater than 0")

    @staticmethod
    def _generate_log(original_size: int, new_size: int) -> str:
        """Generates a log message based on the transformation result.

        Args:
            original_size (int): The original list size.
            new_size (int): The size after transformation.

        Returns:
            str: A descriptive log message.
        """
        if new_size < original_size:
            return (f"Removed {original_size - new_size} items. "
                   f"Size reduced from {original_size} to {new_size}.")
        return "No items were removed."


class MessageHistoryTransform(ListLimiter[Dict]):
    def __init__(self, max_size: Optional[int] = None):
        super().__init__(max_size)

    def transform(self, items: List[Dict]) -> List[Dict]:
        return super().transform(items)
    
    def transform_with_logs(self, items: List[Dict]) -> Tuple[List[Dict], str]:
        return super().transform_with_logs(items)
