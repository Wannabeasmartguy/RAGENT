from typing import Annotated, Literal
from trafilatura import fetch_url, extract
from autogen.coding.func_with_reqs import with_requirements,ImportFromModule

Operator = Literal["+", "-", "*", "/"]


def calculator(a: int, b: int, operator: Annotated[Operator, "operator"]) -> int:
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
def web_scraper(url: str) -> str:
    '''Useful to scrape web pages, and extract text content.'''
    return extract(fetch_url(url),url=url,include_links=True)