import requests
from typing import Dict, Union, Generator, Optional


SUPPORTED_SOURCES = {
    "sources": {
        "openai": "sdk",
        "aoai": "sdk",
        "llamafile": "sdk",
        "ollama": "sdk"
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
        return self._handle_response(response)

    def post(
            self, 
            endpoint: str, 
            data: Dict, 
            headers: Optional[Dict] = {'Content-Type': 'application/json'}, 
            params: Optional[Dict] = None
        ) -> Dict:
        """发送 POST 请求到指定的 endpoint。

        Args:
            endpoint (str): 请求的 API 端点。
            data (dict): 发送的数据，应该是可被序列化为 JSON 的字典。
            headers (dict, optional): 请求头，默认为 {'Content-Type': 'application/json'}。
            params (dict, optional): 查询参数，默认为 None。

        Returns:
            dict: 服务器响应。
        """
        response = requests.post(self.base_url + endpoint, json=data, headers=headers, params=params)
        return self._handle_response(response)

    def put(self, endpoint, data):
        response = requests.put(self.base_url + endpoint, json=data)
        return self._handle_response(response)

    def delete(self, endpoint):
        response = requests.delete(self.base_url + endpoint)
        return self._handle_response(response)

    def get_file(self, endpoint):
        response = requests.get(self.base_url + endpoint)
        return self._handle_response(response)

    def post_file(self, endpoint, file_path):
        with open(file_path, 'rb') as file:
            response = requests.post(self.base_url + endpoint, files={'file': file})
        return self._handle_response(response)

    def delete_file(self, endpoint):
        response = requests.delete(self.base_url + endpoint)
        return self._handle_response(response)

    def _handle_response(self, response) -> Union[Dict, Generator]:
        if response.status_code == 200:
            # 检查响应是否为 StreamingResponse
            if 'content-type' in response.headers and 'plain' in response.headers['content-type']:
                # 处理 StreamingResponse
                return response  # 返回响应的文本内容
            else:
                return response.json()  # 返回 JSON 响应
        else:
            return {'error': response.status_code, 'message': response.text}


async def return_supported_sources():
    return SUPPORTED_SOURCES