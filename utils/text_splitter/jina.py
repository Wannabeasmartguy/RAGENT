import requests
from typing import Optional, Literal, Dict

class JinaAI_Tokenizer:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Jina AI class.

        :param api_key: The Bearer token for authentication.
        """

        # Set up the URL and headers
        self.url = 'https://tokenize.jina.ai/'
        self.headers = {
            'Content-Type': 'application/json'
        }

        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'

    def _prepare_input_data(self, page_content: str) -> Dict[str, str]:
        """
        Prepare the input data for tokenization.

        :param page_content: The content to be processed.
        :return: A dictionary containing the prepared data.
        """

        # Initialize an empty dictionary to store the input data
        input_data = {}

        # Add the content to the input data dictionary
        input_data['content'] = page_content

        return input_data

    def tokenize_content(self, 
                            content: str,
                            tokenizer: Optional[Literal["c100k_base", "o200k_base", "p50k_base", "r50k_base", "p50k_edit", "gpt2"]] = None,
                            return_tokens: bool = False,
                            return_chunks: bool = False
                           ) -> Dict[str, str]:
        """
        Tokenize the given content using Jina AI API.

        :param content: The text to be tokenized.
        :param tokenizer: The tokenizer to be used.
        :param return_tokens: Whether to return tokens.
        :param return_chunks: Whether to return chunks.
        :return: A dictionary containing the response from the API.
        """

        # Prepare the data to be sent in the request body
        data = {
            "content": content,
            "return_tokens": return_tokens,
            "return_chunks": return_chunks
        }
        if tokenizer:
            data["tokenizer"] = tokenizer

        try:
            # Send a POST request to the API with the prepared data
            response = requests.post(self.url, headers=self.headers, json=data)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Request failed with status code {response.status_code}")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Example usage
    api_key = "type_your_api_key_here"
    jina_ai = JinaAI_Tokenizer(
        #api_key
    )
    page_content = "Jina AI: Your Search Foundation, Supercharged! 🚀Ihrer Suchgrundlage, aufgeladen! 🚀Your search base, from now on different! 🚀検索ベース,もう二度と同じことはありません！🚀"
    input_data = jina_ai.prepare_input_data(page_content)
    tokenized_content = jina_ai.tokenize_content(input_data['content'], return_tokens=True, return_chunks=True)
    print(tokenized_content)