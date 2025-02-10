from typing import List, Dict, Tuple, Optional, TypeVar, Generic, Union
from core.models.app import MessageType

import unittest


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


class MessageHistoryTransform(ListLimiter[Union[Dict, MessageType]]):
    def __init__(self, max_size: Optional[int] = None):
        super().__init__(max_size)

    def transform(self, items: List[MessageType]) -> List[MessageType]:
        return super().transform(items)
    
    def transform_with_logs(self, items: List[MessageType]) -> Tuple[List[MessageType], str]:
        return super().transform_with_logs(items)


class TagProcessor:
    """A class for processing strings containing tagged sections.

    This class can detect, modify, or delete sections of a string that are enclosed by specified tag pairs.
    """

    def __init__(self, start_tag: str, end_tag: str):
        """
        Args:
            start_tag (str): The starting tag (e.g., "<think>").
            end_tag (str): The ending tag (e.g., "</think>").

        Raises:
            ValueError: If tags are empty or not properly formatted.
        """
        self._validate_tags(start_tag, end_tag)
        self._start_tag = start_tag
        self._end_tag = end_tag

    def add(self, text: str, if_newline: bool = True) -> str:
        """Adds a new tagged section to the text.

        Args:
            text (str): The text to which the tag will be added.

        Returns:
            str: The modified text with the new tag added.
        """
        if if_newline:
            return f"{self._start_tag}{text}{self._end_tag}" + "\n\n"
        return f"{self._start_tag}{text}{self._end_tag}"

    def detect(self, text: str) -> bool:
        """Detects if the text contains any section enclosed by the specified tags.

        Args:
            text (str): The text to be checked.

        Returns:
            bool: True if a tagged section is found, False otherwise.
        """
        self._validate_tags(self._start_tag, self._end_tag)
        return self._start_tag in text and self._end_tag in text

    def extract(self, text: str) -> Tuple[str, str]:
        """Detects if the text contains any section enclosed by the specified tags and returns the first occurrence and the rest of the text.

        Args:
            text (str): The text to be checked.

        Returns:
            Tuple[str, str]: The first occurrence of the tagged section if found and the rest of the text, otherwise an empty string and the original text.
        """
        start_index = text.find(self._start_tag)
        if start_index == -1:
            return "", text

        end_index = self._find_matching_end_tag(text, start_index)
        if end_index == -1:
            return "", text

        return text[start_index:end_index], text[:start_index] + text[end_index:]

    def _find_matching_end_tag(self, text: str, start_index: int) -> int:
        """Finds the matching end tag for the start tag at the given index.

        Args:
            text (str): The text to be checked.
            start_index (int): The index of the start tag.

        Returns:
            int: The index of the matching end tag, or -1 if not found.
        """
        tag_depth = 1
        index = start_index + len(self._start_tag)

        while index < len(text):
            if text.startswith(self._start_tag, index):
                tag_depth += 1
            elif text.startswith(self._end_tag, index):
                tag_depth -= 1
                if tag_depth == 0:
                    return index + len(self._end_tag)
            index += 1

        return -1

    def extract_all(self, text: str) -> Tuple[List[str], str]:
        """Detects if the text contains any section enclosed by the specified tags and returns all occurrences and the rest of the text.

        Args:
            text (str): The text to be checked.

        Returns:
            Tuple[List[str], str]: A Tuple containing a list of all occurrences of the tagged section if found and the rest of the text, otherwise an empty list and the original text.
        """
        occurrences = []
        remaining_text = text

        while True:
            start_index = remaining_text.find(self._start_tag)
            if start_index == -1:
                break  # No more start tags found, exit loop

            end_index = self._find_matching_end_tag(remaining_text, start_index)
            if end_index == -1:
                break  # No matching end tag found, exit loop

            # Extract the tagged section
            tagged_section = remaining_text[start_index:end_index]
            occurrences.append(tagged_section)

            # Remove the tagged section from the remaining text
            remaining_text = remaining_text[:start_index] + remaining_text[end_index:]

        return occurrences, remaining_text

    def modify(self, text: str, replacement: str) -> str:
        """Modifies the first occurrence of the tagged section with the given replacement.

        Args:
            text (str): The text to be modified.
            replacement (str): The replacement string.

        Returns:
            str: The modified text.
        """
        start_index = text.find(self._start_tag)
        if start_index == -1:
            return text

        end_index = self._find_matching_end_tag(text, start_index)
        if end_index == -1:
            return text

        return text[:start_index] + replacement + text[end_index:]

    def delete(self, text: str) -> str:
        """Deletes the first occurrence of the tagged section.
        If the beginning of the text remains line breaks, they will be removed.

        Args:
            text (str): The text to be modified.

        Returns:
            str: The modified text with the tagged section removed.
        """
        deleted_text = self.modify(text, "")
        return deleted_text.lstrip("\n")

    def modify_all(self, text: str, replacement: str) -> str:
        """Modifies all occurrences of the tagged sections with the given replacement.

        Args:
            text (str): The text to be modified.
            replacement (str): The replacement string.

        Returns:
            str: The modified text.
        """
        while self.detect(text):
            text = self.modify(text, replacement)
        return text

    def delete_all(self, text: str) -> str:
        """Deletes all occurrences of the tagged sections.
        If the beginning of the text remains line breaks, they will be removed.

        Args:
            text (str): The text to be modified.

        Returns:
            str: The modified text with all tagged sections removed.
        """
        deleted_text = self.modify_all(text, "")
        return deleted_text.lstrip("\n")

    def process_with_logs(self, text: str, operation: str, replacement: Optional[str] = None) -> Tuple[str, str]:
        """Processes the text with the specified operation and returns both the result and a log message.

        Args:
            text (str): The text to be processed.
            operation (str): The operation to perform ("modify", "delete", "modify_all", "delete_all").
            replacement (Optional[str]): The replacement string for modify operations.

        Returns:
            Tuple[str, str]: A tuple containing the processed text and a log message.
        """
        original_text = text
        if operation == "modify":
            text = self.modify(text, replacement or "")
        elif operation == "add":
            text = self.add(text)
        elif operation == "delete":
            text = self.delete(text)
        elif operation == "extract":
            _occurance, text = self.extract(text)
        elif operation == "modify_all":
            text = self.modify_all(text, replacement or "")
        elif operation == "delete_all":
            text = self.delete_all(text)
        elif operation == "extract_all":
            _occurances, text = self.extract_all(text)
        else:
            return text, "Invalid operation specified."

        log_msg = self._generate_log(original_text, text, operation)
        return text, log_msg

    @staticmethod
    def _validate_tags(start_tag: str, end_tag: str) -> None:
        """Validates the tag parameters.

        Args:
            start_tag (str): The starting tag.
            end_tag (str): The ending tag.

        Raises:
            ValueError: If tags are empty or not properly formatted.
        """
        if (
            not start_tag 
            or not end_tag 
            or not start_tag.startswith("<")
            or not start_tag.endswith(">")
            or not end_tag.startswith("<")
            or not end_tag.endswith(">")
        ):
            raise ValueError("Tags must be non-empty and properly formatted (e.g., '<tag>' and '</tag>').")

    @staticmethod
    def _generate_log(original_text: str, new_text: str, operation: str) -> str:
        """Generates a log message based on the processing result.

        Args:
            original_text (str): The original text.
            new_text (str): The text after processing.
            operation (str): The operation performed.

        Returns:
            str: A descriptive log message.
        """
        if original_text != new_text:
            return f"Performed {operation} operation. Text changed from '{original_text}' to '{new_text}'."
        return f"No changes made during {operation} operation."


class ReasoningContentTagProcessor(TagProcessor):
    """A tag processor for handling reasoning content tags."""

    def __init__(self, start_tag: str = "<think>", end_tag: str = "</think>"):
        """Initializes the ReasoningContentTagProcessor.
        
        Args:
            start_tag (str, optional): The starting tag. Defaults to "<think>".
            end_tag (str, optional): The ending tag. Defaults to "</think>".
        """
        super().__init__(start_tag, end_tag)


class TestTagProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = TagProcessor("<think>", "</think>")

    def test_add(self):
        self.assertEqual(self.processor.add("Hello, world!"), "<think>Hello, world!</think>\n\n")
        self.assertEqual(self.processor.add("Hello, world!", if_newline=False), "<think>Hello, world!</think>")

    def test_detect(self):
        self.assertTrue(self.processor.detect("<think>Hello, world!</think>"))
        self.assertFalse(self.processor.detect("Hello, world!"))

    def test_extract(self):
        self.assertEqual(self.processor.extract("<think>Hello, world!</think>"), ("<think>Hello, world!</think>", ""))
        self.assertEqual(self.processor.extract("Hello, world!"), ("", "Hello, world!"))

    def test_extract_all(self):
        self.assertEqual(self.processor.extract_all("<think>Hello</think><think>World</think>"), (["<think>Hello</think>", "<think>World</think>"], ""))
        self.assertEqual(self.processor.extract_all("Hello, world!"), ([], "Hello, world!"))

    def test_modify(self):
        self.assertEqual(self.processor.modify("<think>Hello, world!</think>", "Hi, universe!"), "Hi, universe!")
        self.assertEqual(self.processor.modify("Hello, world!", "Hi, universe!"), "Hello, world!")

    def test_delete(self):
        self.assertEqual(self.processor.delete("<think>Hello, world!</think>"), "")
        self.assertEqual(self.processor.delete("Hello, world!"), "Hello, world!")

    def test_modify_all(self):
        self.assertEqual(self.processor.modify_all("<think>Hello</think><think>World</think>", "Hi"), "HiHi")
        self.assertEqual(self.processor.modify_all("Hello, world!", "Hi"), "Hello, world!")

    def test_delete_all(self):
        self.assertEqual(self.processor.delete_all("<think>Hello</think><think>World</think>"), "")
        self.assertEqual(self.processor.delete_all("Hello, world!"), "Hello, world!")

    def test_process_with_logs(self):
        self.assertEqual(self.processor.process_with_logs("<think>Hello, world!</think>", "modify", "Hi, universe!"), ("Hi, universe!", "Performed modify operation. Text changed from '<think>Hello, world!</think>' to 'Hi, universe!'."))
        self.assertEqual(self.processor.process_with_logs("Hello, world!", "modify", "Hi, universe!"), ("Hello, world!", "No changes made during modify operation."))

    def test_validate_tags(self):
        with self.assertRaises(ValueError):
            TagProcessor("", "</think>")
        with self.assertRaises(ValueError):
            TagProcessor("<think>", "")
        with self.assertRaises(ValueError):
            TagProcessor("think", "</think>")
        with self.assertRaises(ValueError):
            TagProcessor("<think", "</think>")
        with self.assertRaises(ValueError):
            TagProcessor("<think>", "think>")

if __name__ == "__main__":
    unittest.main()
