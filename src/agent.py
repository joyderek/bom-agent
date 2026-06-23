from __future__ import annotations

from typing import Any

from openai import APIConnectionError, APITimeoutError
from langchain.agents import create_agent
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langchain_openai import ChatOpenAI
from src.config import AgentConfig
from src.middleware import ResearchGuardMiddleware, research_guard_after_model
from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.web_tools import build_tools


def _build_model(config: AgentConfig) -> ChatOpenAI:
    extra_body = None
    if "api.deepseek.com" in config.base_url:
        extra_body = {
            "thinking": {
                "type": "enabled" if config.thinking_enabled else "disabled",
            }
        }

    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        api_key=config.api_key or None,
        base_url=config.base_url or None,
        extra_body=extra_body,
        timeout=config.model_timeout_seconds,
        max_retries=config.model_max_retries,
    )


def build_agent(config: AgentConfig):
    tools = build_tools(
        search_provider=config.search_provider,
        default_max_results=config.max_search_results,
        tavily_api_key=config.tavily_api_key,
        serper_api_key=config.serper_api_key,
        brave_api_key=config.brave_api_key,
        searxng_base_url=config.searxng_base_url,
    )
    model = _build_model(config)
    middleware = [
        ResearchGuardMiddleware(max_tool_calls=config.max_tool_calls),
        research_guard_after_model,
        ToolCallLimitMiddleware(run_limit=config.max_tool_calls, exit_behavior="end"),
    ]
    return create_agent(
        tools=tools,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        debug=False,
    )


def _extract_text_from_message(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def run_bom_decomposition(
    product_name: str,
    product_context: str | None = None,
    config: AgentConfig | None = None,
) -> str:
    active_config = config or AgentConfig.from_env()
    agent = build_agent(active_config)
    try:
        raw = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": build_user_prompt(product_name, product_context),
                    }
                ]
            },
            config={"recursion_limit": active_config.recursion_limit},
        )
    except APIConnectionError as exc:
        raise RuntimeError(
            "模型请求失败，错误类型为 APIConnectionError。"
            f"base_url={active_config.base_url!r}, model={active_config.model!r}, "
            f"timeout={active_config.model_timeout_seconds}s, retries={active_config.model_max_retries}. "
            "这个问题与搜索提供方无关。"
            "请检查模型接口地址，或尝试更小更快的模型，或提高 model_timeout_seconds。"
        ) from exc
    except GraphRecursionError as exc:
        raise RuntimeError(
            "Agent 在达到停止条件前触发了递归上限。"
            f"recursion_limit={active_config.recursion_limit}, "
            f"max_tool_calls={active_config.max_tool_calls}, "
            f"max_search_results={active_config.max_search_results}. "
            "这通常表示模型持续调用工具但没有收敛到最终输出。"
        ) from exc
    messages = raw.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = _extract_text_from_message(message)
            if text:
                return text
    last_message = messages[-1] if messages else None
    raise RuntimeError(
        "Agent 执行结束后没有返回可用文本结果。"
        f"message_count={len(messages)}, "
        f"last_message_type={type(last_message).__name__ if last_message is not None else 'None'}."
    )
