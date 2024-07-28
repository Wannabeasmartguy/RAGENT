import json
import random
from loguru import logger
from typing import Any, Dict, List, Union, Optional
from openai import OpenAI, Stream
from openai.types.chat.chat_completion import ChatCompletion

class ToolsParameterOutputParser:
    """将工具调用后返回的消息转换为JSON格式"""
    
    def __call__(self, message: Union[ChatCompletion, Stream]) -> List[Dict[str, Any]]:
        """
        根据输入消息的类型，选择适当的方法进行解析。
        
        :param message: 包含工具调用信息的消息， 可以是ChatCompletion或Stream类型
        :return: 解析后的JSON列表
        """
        if isinstance(message, Stream):
            return self._parse_stream_to_json(message)
        else:
            return self.parse_tools_to_json(message)

    def _generate_unique_id(self) -> str:
    # 生成一个随机的8位数字和字符的组合
        random_str = ''.join(random.choices('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', k=8))
        
        # 生成一个唯一的ID
        unique_id = f"call_{random_str}"
        
        return unique_id

    def _parse_stream_to_json(self, stream_output: Stream) -> List[Dict[str, Any]]:
        """将流式输出转换为字典格式的JSON字符串列表"""
        chunks = list(stream_output)
        stream2str = ''.join(chunk.choices[0].delta.content for chunk in chunks if chunk.choices and chunk.choices[0].delta.content is not None)
        logger.debug(stream2str)
        
        params =  [json.loads(param) for param in stream2str.split('\n') if param]
        # 给每个参数添加一个id字段
        # 创建一个可以生成唯一ID的生成器，格式为"call_{id}",id为8位随机数字和字符的组合
        
        return [{'name': param['name'], 'parameters': param['parameters'], 'tool_call_id': self._generate_unique_id()} for param in params]

    def parse_tools_to_json(self, output: ChatCompletion) -> List[Dict[str, Any]]:
        """
        将非流式输出转换为JSON格式的工具参数列表。
        
        :param output: 包含工具调用信息的消息
        :return: 解析后的JSON列表，每个元素是一个包含工具名称和参数的字典
        """
        params = [param.function for param in output.choices[0].message.tool_calls]
        tool_call_ids = [param.id for param in output.choices[0].message.tool_calls]
        
        return [{'name': param.name, 'parameters': json.loads(param.arguments), 'tool_call_id': tool_call_ids[params.index(param)]} for param in params]


def create_completion(
        messages: List[Dict[str, Any]], 
        model:str = "llama3.1:latest", 
        tools: Optional[List[Dict[str, Any]]] = None, 
        function_map: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> List[Dict[str, Any]]:
    """
    创建一个响应，如果检测到工具调用，则使用工具调用参数
    
    :param messages: 消息列表
    :param tools: 工具列表
    :param function_map: 函数映射，根据工具名称映射到本地函数
    :return: 包含工具调用参数的一轮完整对话
    """
    parser = ToolsParameterOutputParser()

    # 第一步， 调用模型生成工具调用参数
    client = OpenAI(
        api_key='ollama',
        base_url='http://localhost:11434/v1'
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0
        # stream=stream
    )
    parsed_params = parser(response)
    logger.debug(parsed_params)
    # 添加工具调用参数到消息列表
    messages.append(dict(response.choices[0].message))

    # 第二步， 根据工具调用参数，本地运行工具，并返回结果
    function_results = [function_map[param['name']](**param['parameters']) for param in parsed_params]
    # 添加所有工具调用结果到消息列表
    for result in function_results:
        messages.append({
            "role":"tool",
            "name": parsed_params[function_results.index(result)]['name'],
            "content":result,
            "tool_call_id":parsed_params[function_results.index(result)]['tool_call_id']
        })
    
    # 第三步， 调用模型生成最终的回复
    logger.debug(f"final messages input: {messages}")
    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=stream
    )
    messages.append(dict(final_response.choices[0].message))
    return final_response, messages