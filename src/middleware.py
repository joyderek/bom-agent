from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain.agents.middleware.types import after_model


_TRACKED_TOOLS = {"web_search", "read_web_page"}
_QUERY_SPACE_RE = re.compile(r"\s+")
_DROP_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
}


def _normalize_query(query: str) -> str:
    query = query.strip().lower()
    return _QUERY_SPACE_RE.sub(" ", query)


def _normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower() or "https"
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parts.path.rstrip("/") or "/"
    filtered_query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() not in _DROP_QUERY_PARAMS
        )
    )
    return urlunsplit((scheme, netloc, path, filtered_query, ""))


def _tool_signature(tool_name: str, args: dict[str, Any]) -> str | None:
    if tool_name == "web_search":
        query = args.get("query")
        if isinstance(query, str) and query.strip():
            return f"web_search::{_normalize_query(query)}"
        return None
    if tool_name == "read_web_page":
        url = args.get("url")
        if isinstance(url, str) and url.strip():
            return f"read_web_page::{_normalize_url(url)}"
        return None
    return None


def _blocked_tool_message(tool_name: str, tool_call_id: str, reason: str) -> ToolMessage:
    payload = {
        "ok": False,
        "duplicate": True,
        "reason": reason,
        "next_action": "复用对话里已有证据；如果覆盖度已经够了，就停止调用工具并直接产出 BOM。",
    }
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False, indent=2),
        tool_call_id=tool_call_id,
        name=tool_name,
        status="error",
    )


def _prior_signatures(messages: list[Any]) -> set[str]:
    signatures: set[str] = set()
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls:
            tool_name = tool_call.get("name")
            if tool_name not in _TRACKED_TOOLS:
                continue
            tool_call_id = tool_call.get("id")
            if not isinstance(tool_call_id, str):
                continue
            args = tool_call.get("args")
            if not isinstance(args, dict):
                continue
            signature = _tool_signature(tool_name, args)
            if signature:
                signatures.add(signature)
    return signatures


def _count_tool_messages(messages: list[Any]) -> int:
    return sum(1 for message in messages if isinstance(message, ToolMessage))


def _count_blocked_duplicates(messages: list[Any]) -> int:
    total = 0
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = message.content
        if isinstance(content, str) and '"duplicate": true' in content:
            total += 1
    return total


def _recent_tool_messages(messages: list[Any], limit: int = 4) -> list[ToolMessage]:
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    return tool_messages[-limit:]


def _parse_tool_payload(message: ToolMessage) -> dict[str, Any]:
    content = message.content
    if not isinstance(content, str):
        return {}
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _recent_failures_or_duplicates(messages: list[Any], limit: int = 4) -> int:
    total = 0
    for message in _recent_tool_messages(messages, limit=limit):
        payload = _parse_tool_payload(message)
        if not payload:
            continue
        if payload.get("duplicate") is True or payload.get("ok") is False:
            total += 1
    return total


def _last_ai_message(messages: list[Any]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _clear_ai_tool_calls(message: AIMessage) -> AIMessage:
    metadata = dict(getattr(message, "response_metadata", {}))
    guard_state = dict(metadata.get("research_guard", {}))
    guard_state["forced_finalize"] = True
    metadata["research_guard"] = guard_state
    return message.model_copy(update={"tool_calls": [], "response_metadata": metadata})


class ResearchGuardMiddleware(AgentMiddleware):
    def __init__(self, *, max_tool_calls: int, near_limit_buffer: int = 2) -> None:
        self.max_tool_calls = max_tool_calls
        self.near_limit_buffer = near_limit_buffer

    def wrap_model_call(self, request, handler):
        messages = request.state.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        tool_message_count = _count_tool_messages(messages)
        blocked_duplicate_count = _count_blocked_duplicates(messages)

        reminders: list[str] = []
        if tool_message_count >= max(1, self.max_tool_calls - self.near_limit_buffer):
            reminders.append(
                "你已经接近工具调用预算上限。不要再为了边际提升继续探索；除非缺失的是关键来源，否则直接综合现有信息产出尽力而为的 BOM。"
            )
        if blocked_duplicate_count > 0:
            reminders.append(
                "重复搜索或重复读页已被拦截。不要再次使用相同查询或 URL；在覆盖度足够时复用现有证据并结束。"
            )

        if reminders:
            base_prompt = request.system_prompt or ""
            extra_prompt = "\n".join(f"- {reminder}" for reminder in reminders)
            request = request.override(
                system_message=SystemMessage(
                    content=f"{base_prompt}\n\n执行约束：\n{extra_prompt}"
                )
            )

        return handler(request)

    def wrap_tool_call(self, request, handler):
        tool_name = request.tool_call.get("name")
        if tool_name not in _TRACKED_TOOLS:
            return handler(request)

        args = request.tool_call.get("args")
        if not isinstance(args, dict):
            return handler(request)

        signature = _tool_signature(tool_name, args)
        if not signature:
            return handler(request)

        messages = request.state.get("messages", [])
        if isinstance(messages, list) and signature in _prior_signatures(messages[:-1]):
            reason = (
                "重复搜索已被拦截。"
                if tool_name == "web_search"
                else "重复页面读取已被拦截。"
            )
            return _blocked_tool_message(tool_name, request.tool_call["id"], reason)

        return handler(request)


@after_model(can_jump_to=["end"], name="research_guard_after_model")
def research_guard_after_model(state, runtime):
    messages = state.get("messages", [])
    if not isinstance(messages, list):
        return None

    last_ai = _last_ai_message(messages)
    if last_ai is None or not last_ai.tool_calls:
        return None

    recent_problem_count = _recent_failures_or_duplicates(messages, limit=4)
    if recent_problem_count < 3:
        return None

    updated_messages = list(messages)
    updated_messages[updated_messages.index(last_ai)] = _clear_ai_tool_calls(last_ai)
    return {
        "messages": updated_messages,
        "jump_to": "end",
    }
