from typing import Annotated, Literal, Dict, Callable, Union
from trafilatura import fetch_url, extract
from autogen.coding.func_with_reqs import with_requirements,ImportFromModule

Operator = Literal["+", "-", "*", "/"]

def tool_calculator(a: int, b: int, operator: Annotated[Operator, "operator"]) -> int:
    '''A basic int calculator can do addition, subtraction, multiplication, and division.'''
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return int(a / b)
    else:
        raise ValueError("Invalid operator")
    

@with_requirements(python_packages=["trafilatura"],global_imports=[ImportFromModule("fetch_url", "extract")])
def tool_web_scraper(url: str) -> str:
    '''Useful to scrape web pages, and extract text content.'''
    return extract(fetch_url(url),url=url,include_links=True)


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