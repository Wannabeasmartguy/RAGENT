import requests


SUPPORTED_SOURCES = {
    "sources": {
        "openai": "sdk",
        "aoai": "sdk",
        "llamafile": "sdk",
        "ollama": "request"
    }
}


class APIRequestHandler:
    '''
    用于与FastAPI服务器进行交互的类。
    该类封装了与FastAPI服务器进行HTTP请求的方法。
    '''
    def __init__(self, server_address, port):
        self.base_url = f"http://{server_address}:{port}"

    def get(self, endpoint):
        response = requests.get(self.base_url + endpoint)
        return response.json()

    def post(self, endpoint, data):
        response = requests.post(self.base_url + endpoint, json=data)
        return response.json()

    def put(self, endpoint, data):
        response = requests.put(self.base_url + endpoint, json=data)
        return response.json()

    def delete(self, endpoint):
        response = requests.delete(self.base_url + endpoint)
        return response.json()

    def get_file(self, endpoint):
        response = requests.get(self.base_url + endpoint)
        return response.content

    def post_file(self, endpoint, file_path):
        with open(file_path, 'rb') as file:
            response = requests.post(self.base_url + endpoint, files={'file': file})
        return response.json()

    def delete_file(self, endpoint):
        response = requests.delete(self.base_url + endpoint)
        return response.json()


async def return_supported_sources():
    return SUPPORTED_SOURCES