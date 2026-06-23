from __future__ import annotations

import json
import textwrap
from typing import Any

import requests
from langchain.tools import tool
from pydantic import BaseModel, Field
from requests import RequestException


USER_AGENT = "bom-agent/0.1 (+public-web-research)"


class SearchInput(BaseModel):
    query: str = Field(..., description="Search query about the target product or component.")
    max_results: int = Field(default=5, ge=1, le=10)


class FetchInput(BaseModel):
    url: str = Field(..., description="Web page URL to fetch through a readable mirror.")


def _limit_results(items: list[dict[str, Any]], max_results: int) -> list[dict[str, Any]]:
    return items[: max(1, max_results)]


def _tavily_search(api_key: str, query: str, max_results: int) -> dict[str, Any]:
    response = requests.post(
        "https://api.tavily.com/search",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        json={
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    source = data if isinstance(data, dict) else {}
    return {
        "answer": source.get("answer"),
        "results": [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "score": item.get("score"),
            }
            for item in source.get("results", [])
        ],
    }


def _tavily_extract(api_key: str, url: str) -> dict[str, Any]:
    response = requests.post(
        "https://api.tavily.com/extract",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        json={
            "urls": [url],
            "extract_depth": "advanced",
            "format": "markdown",
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    source = data if isinstance(data, dict) else {}
    results = source.get("results", [])
    first = results[0] if isinstance(results, list) and results else {}
    content = first.get("raw_content", "") or ""
    if not content.strip():
        failed_results = source.get("failed_results", [])
        failure_detail = failed_results[0] if isinstance(failed_results, list) and failed_results else {}
        failure_reason = ""
        if isinstance(failure_detail, dict):
            failure_reason = str(
                failure_detail.get("error")
                or failure_detail.get("reason")
                or failure_detail.get("message")
                or ""
            ).strip()
        raise RequestException(f"Tavily extract returned empty content. {failure_reason}".strip())
    return {
        "url": first.get("url", url),
        "content": content,
        "failed_results": source.get("failed_results", []),
    }


def _serper_search(api_key: str, query: str, max_results: int) -> dict[str, Any]:
    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        json={"q": query, "num": max_results},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    source = data if isinstance(data, dict) else {}
    organic = source.get("organic", [])
    answer_parts: list[str] = []
    answer_box = source.get("answerBox")
    if isinstance(answer_box, dict):
        for key in ("answer", "snippet", "title"):
            value = answer_box.get(key)
            if isinstance(value, str) and value.strip():
                answer_parts.append(value.strip())
                break
    knowledge_graph = source.get("knowledgeGraph")
    if isinstance(knowledge_graph, dict):
        description = knowledge_graph.get("description")
        if isinstance(description, str) and description.strip():
            answer_parts.append(description.strip())
    return {
        "answer": "\n".join(answer_parts) if answer_parts else None,
        "results": _limit_results(
            [
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "score": None,
                }
                for item in organic
            ],
            max_results,
        ),
    }


def _brave_search(api_key: str, query: str, max_results: int) -> dict[str, Any]:
    response = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
        params={"q": query, "count": max_results, "text_decorations": False},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    source = data if isinstance(data, dict) else {}
    web = source.get("web", {})
    results = web.get("results", []) if isinstance(web, dict) else []
    return {
        "answer": None,
        "results": _limit_results(
            [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", ""),
                    "score": None,
                }
                for item in results
            ],
            max_results,
        ),
    }


def _searxng_search(base_url: str, query: str, max_results: int) -> dict[str, Any]:
    response = requests.get(
        f"{base_url.rstrip('/')}/search",
        headers={"User-Agent": USER_AGENT},
        params={"q": query, "format": "json"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    source = data if isinstance(data, dict) else {}
    results = source.get("results", [])
    return {
        "answer": None,
        "results": _limit_results(
            [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "score": item.get("score"),
                }
                for item in results
            ],
            max_results,
        ),
    }


def _search_missing_config_error(provider: str) -> str:
    if provider == "tavily":
        return "Missing Tavily API key. Set tavily_api_key in config.local.toml or TAVILY_API_KEY in the environment."
    if provider == "serper":
        return "Missing Serper API key. Set serper_api_key in config.local.toml or SERPER_API_KEY in the environment."
    if provider == "brave":
        return "Missing Brave API key. Set brave_api_key in config.local.toml or BRAVE_API_KEY in the environment."
    if provider == "searxng":
        return "Missing SearXNG base URL. Set searxng_base_url in config.local.toml or SEARXNG_BASE_URL in the environment."
    return f"Unsupported search provider: {provider}"


def _search_web(
    provider: str,
    *,
    query: str,
    max_results: int,
    tavily_api_key: str,
    serper_api_key: str,
    brave_api_key: str,
    searxng_base_url: str,
) -> dict[str, Any]:
    if provider == "tavily":
        if not tavily_api_key:
            raise ValueError(_search_missing_config_error(provider))
        return _tavily_search(tavily_api_key, query, max_results)
    if provider == "serper":
        if not serper_api_key:
            raise ValueError(_search_missing_config_error(provider))
        return _serper_search(serper_api_key, query, max_results)
    if provider == "brave":
        if not brave_api_key:
            raise ValueError(_search_missing_config_error(provider))
        return _brave_search(brave_api_key, query, max_results)
    if provider == "searxng":
        if not searxng_base_url:
            raise ValueError(_search_missing_config_error(provider))
        return _searxng_search(searxng_base_url, query, max_results)
    raise ValueError(_search_missing_config_error(provider))


def build_tools(
    *,
    search_provider: str = "tavily",
    default_max_results: int = 5,
    tavily_api_key: str = "",
    serper_api_key: str = "",
    brave_api_key: str = "",
    searxng_base_url: str = "",
):
    @tool("web_search", args_schema=SearchInput)
    def web_search(query: str, max_results: int = default_max_results) -> str:
        """Search the public web for product teardowns, specs, datasheets, filings, and material composition clues."""
        try:
            data = _search_web(
                search_provider,
                query=query,
                max_results=max_results,
                tavily_api_key=tavily_api_key,
                serper_api_key=serper_api_key,
                brave_api_key=brave_api_key,
                searxng_base_url=searxng_base_url,
            )
            payload: dict[str, Any] = {
                "ok": True,
                "provider": search_provider,
                "query": query,
                "answer": data.get("answer"),
                "results": data.get("results", []),
            }
        except ValueError as exc:
            payload = {
                "ok": False,
                "provider": search_provider,
                "query": query,
                "error": str(exc),
                "results": [],
            }
        except RequestException as exc:
            payload = {
                "ok": False,
                "provider": search_provider,
                "query": query,
                "error": f"{type(exc).__name__}: {exc}",
                "results": [],
            }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @tool("read_web_page", args_schema=FetchInput)
    def read_web_page(url: str) -> str:
        """Fetch readable page content, preferring Tavily Extract when available."""
        try:
            if tavily_api_key:
                extracted = _tavily_extract(tavily_api_key, url)
                text = extracted.get("content", "").strip()
                payload: dict[str, Any] = {
                    "ok": True,
                    "provider": "tavily_extract",
                    "url": extracted.get("url", url),
                    "content": textwrap.shorten(text, width=12000, placeholder=" ...[truncated]"),
                }
            else:
                mirror_url = (
                    f"https://r.jina.ai/http://{url.removeprefix('https://').removeprefix('http://')}"
                )
                response = requests.get(
                    mirror_url,
                    headers={"User-Agent": USER_AGENT},
                    timeout=30,
                )
                response.raise_for_status()
                text = response.text.strip()
                payload = {
                    "ok": True,
                    "provider": "jina_reader",
                    "url": url,
                    "content": textwrap.shorten(text, width=12000, placeholder=" ...[truncated]"),
                }
        except RequestException as exc:
            payload = {
                "ok": False,
                "provider": "tavily_extract" if tavily_api_key else "jina_reader",
                "url": url,
                "error": f"{type(exc).__name__}: {exc}",
                "content": "",
            }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    return [web_search, read_web_page]
