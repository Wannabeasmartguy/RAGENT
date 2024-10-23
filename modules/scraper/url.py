import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Literal
from fake_useragent import UserAgent
import time
import random


class UrlScraper:
    def __init__(self, custom_tags: Optional[List[str]] = None):
        self.custom_tags = custom_tags or ["p", "article", "div"]
        self.timeout = 10
        self.ua = UserAgent()
        self.min_delay = 1
        self.max_delay = 5

    def manage_tags(
        self, action: Literal["add", "remove", "set"], tags: Optional[List[str]] = None
    ):
        if action == "add":
            self.custom_tags.extend(
                [tag for tag in tags if tag not in self.custom_tags]
            )
        elif action == "remove":
            self.custom_tags = [tag for tag in self.custom_tags if tag not in tags]
        elif action == "set":
            self.custom_tags = tags
        else:
            raise ValueError(f"Invalid action: {action}")

    def get_headers(self):
        return {
            "User-Agent": self.ua.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def random_delay(self):
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)

    def scrape(self, url: str) -> dict:
        try:
            self.random_delay()

            headers = self.get_headers()
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            content = []

            for tag in self.custom_tags:
                elements = soup.find_all(tag)
                content.extend(
                    [
                        elem.get_text().strip()
                        for elem in elements
                        if elem.get_text().strip()
                    ]
                )

            return {"status": "success", "content": "\n".join(content), "url": url}

        except requests.Timeout:
            return {"status": "error", "message": "Request timed out", "url": url}

        except requests.RequestException as e:
            return {
                "status": "error",
                "message": f"Request error: {str(e)}",
                "url": url,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Unknown error: {str(e)}",
                "url": url,
            }


class JinaScraper:
    url_prefix = "https://r.jina.ai/"

    def __init__(self):
        self.timeout = 10
        self.ua = UserAgent()

    def get_headers(self):
        return {"User-Agent": self.ua.random}

    def scrape(self, url: str) -> dict:
        try:
            response = requests.get(
                url=self.url_prefix + url,
                headers=self.get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return dict(
                status_code=response.status_code,
                headers=response.headers,
                encoding=response.encoding,
                content=response.text,
                url=response.url,
            )
        except requests.exceptions.Timeout:
            return {"status_code": 408, "status": "error", "message": "Request timed out", "url": url}
        except requests.exceptions.RequestException as e:
            return {"status_code": 400, "status": "error", "message": f"Request error: {str(e)}", "url": url}
        except Exception as e:
            return {"status_code": 500, "status": "error", "message": f"Unknown error: {str(e)}", "url": url}
