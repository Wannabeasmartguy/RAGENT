from typing import Annotated, Literal, List, Dict, Callable, Union
from trafilatura import fetch_url, extract
from utils.tool_utils import function_to_json
from utils.log.logger_config import setup_logger
from loguru import logger
import json


Operator = Literal["+", "-", "*", "/"]
def tool_calculator(expression:str) -> str:
    """
    A tool that calculates the result of a given mathematical expression, support basic operations of addition, subtraction, multiplication and division.The format for the expression is like "(3+5)*8/2".

    :param expression: The mathematical expression to be calculated.
    """
    logger.info(f"Calculating expression: {expression}")

    def precedence(op:str):
        if op in ('+', '-'):
            return 1
        if op in ('*', '/'):
            return 2
        return 0

    def apply_operator(operators, values):
        operator = operators.pop()
        right = values.pop()
        left = values.pop()
        if operator == '+':
            values.append(left + right)
        elif operator == '-':
            values.append(left - right)
        elif operator == '*':
            values.append(left * right)
        elif operator == '/':
            values.append(left / right)
        logger.info(f"Applied operator {operator}: {left} {operator} {right} = {values[-1]}")

    # 计算表达式
    values = []
    operators = []
    i = 0
    while i < len(expression):
        # 如果当前字符是数字
        if expression[i].isdigit():
            # 找到数字的末尾
            j = i
            while j < len(expression) and expression[j].isdigit():
                j += 1
            # 将数字添加到values列表中
            values.append(int(expression[i:j]))
            logger.info(f"Pushed value: {values[-1]}")
            i = j
        # 如果当前字符是左括号
        elif expression[i] == '(':
            # 将左括号添加到operators列表中
            operators.append(expression[i])
            logger.info(f"Pushed operator: {expression[i]}")
            i += 1
        # 如果当前字符是右括号
        elif expression[i] == ')':
            # 弹出operators列表中的左括号
            while operators and operators[-1] != '(':
                apply_operator(operators, values)
            operators.pop()
            logger.info(f"Popped operator: {expression[i]}")
            i += 1
        # 如果当前字符是运算符
        elif expression[i] in "+-*/":
            # 弹出operators列表中的运算符，直到遇到左括号或者当前运算符的优先级小于等于operators[-1]的优先级
            while (operators and precedence(operators[-1]) >= precedence(expression[i])):
                apply_operator(operators, values)
            # 将当前运算符添加到operators列表中
            operators.append(expression[i])
            logger.info(f"Pushed operator: {expression[i]}")
            i += 1
        else:
            i += 1

    while operators:
        apply_operator(operators, values)

    logger.info(f"Final result: {values[0]}")
    return str(values[0])


def tool_web_scraper(url: str) -> str:
    '''Useful to scrape web pages, and extract text content.'''
    return extract(fetch_url(url),url=url,include_links=True)


# ************************************
# Write all the tool functions above
# ************************************


# 自动将所有整个py文件里的tool添加到to_tools字典中
TO_TOOLS: Dict[str, Dict[str, Union[Callable, str]]] = {
    tool.__name__: {
        "name": tool.__name__,
        "func": tool,
        "description": tool.__doc__ if tool.__doc__ is not None else ""
    }
    for tool in globals().values()
    if callable(tool) and hasattr(tool, "__name__") and tool.__name__.startswith("tool_")
}

TOOLS_LIST = [json.loads(function_to_json(tool["func"])) for tool in TO_TOOLS.values()]

TOOLS_MAP = {
    tool["name"]: tool["func"]
    for tool in TO_TOOLS.values()
}


# ************************************
# other tools would be used in codes write below
# ************************************


def filter_out_selected_tools_list(selected_tools: List[str]):
    """
    筛选出所有被选中的工具

    :param selected_tools: 被选中的工具名称列表
    :return: 从 TOOLS_LIST 中筛选出的工具列表
    """
    return [tool for tool in TOOLS_LIST if tool['function']['name'] in selected_tools]


def filter_out_selected_tools_dict(selected_tools: List[str]):
    """
    筛选出所有被选中的工具，获得一个工具名称到工具的映射

    :param selected_tools: 被选中的工具名称列表
    :return: 从 TOOLS_MAP 中筛选出的工具列表
    """
    return {tool['function']['name']: TOOLS_MAP[tool['function']['name']] for tool in TOOLS_LIST if tool['function']['name'] in selected_tools}