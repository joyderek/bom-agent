from __future__ import annotations

from typing import Any

from openai import APIConnectionError, APITimeoutError
from langchain.agents import create_agent
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from src.config import AgentConfig
from src.middleware import ResearchGuardMiddleware, research_guard_after_model
from src.models import BomDecomposition, BomItem
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
        response_format=BomDecomposition,
        debug=True,
    )


def _format_message_for_finalize(message: Any) -> str:
    if isinstance(message, HumanMessage):
        return f"用户：\n{message.content}"
    if isinstance(message, AIMessage):
        tool_calls = getattr(message, "tool_calls", []) or []
        suffix = f"\n工具调用：{tool_calls}" if tool_calls else ""
        return f"助手：\n{message.content}{suffix}"
    if isinstance(message, ToolMessage):
        return f"工具 [{message.name}]：\n{message.content}"
    return f"其他：\n{getattr(message, 'content', str(message))}"


def _finalize_without_tools(
    config: AgentConfig,
    product_name: str,
    product_context: str | None,
    raw: dict[str, Any],
) -> BomDecomposition:
    model = _build_model(config)
    messages = raw.get("messages", [])
    transcript = "\n\n".join(
        _format_message_for_finalize(message)
        for message in messages
        if isinstance(message, (HumanMessage, AIMessage, ToolMessage))
    )
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "研究流程已经结束。不要再调用任何工具。"
        "请产出一个尽力而为的 BomDecomposition，并且必须严格符合 schema。"
        "如果工具失败，或者没有拿到可用网页证据，可以依赖行业知识，但必须在 scope_notes 中明确说明。\n\n"
        "只返回 JSON，不要使用 Markdown 代码块包裹。"
        "JSON 对象必须严格匹配下面这个 schema：\n"
        "{\n"
        '  "product_name": "string",\n'
        '  "product_description": "string",\n'
        '  "scope_notes": ["string"],\n'
        '  "top_level_bom": [\n'
        "    {\n"
        '      "name": "string",\n'
        '      "category": "string",\n'
        '      "quantity": "string or null",\n'
        '      "material_type": "assembly|component|subcomponent|raw_material",\n'
        '      "confidence": "high|medium|low",\n'
        '      "rationale": "string",\n'
        '      "evidence": [{"url": "https://...", "title": "string", "snippet": "string"}],\n'
        '      "children": []\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"产品：{product_name}\n"
        f"产品上下文：{product_context or ''}\n\n"
        "对话记录：\n"
        f"{transcript}"
    )
    response = model.invoke(prompt)
    content = response.content
    if not isinstance(content, str):
        raise ValueError("收尾步骤没有返回文本内容。")
    try:
        return BomDecomposition.model_validate_json(content)
    except ValidationError as exc:
        raise ValueError(f"收尾步骤返回的 BOM JSON 不合法：{exc}") from exc


def _fallback_bom(product_name: str, product_context: str | None, reason: str) -> BomDecomposition:
    notes = [
        "由于未能成功获取网页证据，当前结果为尽力而为的兜底输出。",
        reason,
        "由于搜索或工具链失败，这份 BOM 无法完全基于来源证据构建，因此部分内容依赖通用智能手机架构知识。",
    ]
    if product_context:
        notes.append(f"用户补充上下文：{product_context}")
    return BomDecomposition(
        product_name=product_name,
        product_description=(
            f"基于典型高端智能手机架构，对 {product_name} 做出的尽力拆解结果。"
        ),
        scope_notes=notes,
        top_level_bom=[
            BomItem(
                name="整机外壳总成",
                category="enclosure",
                quantity="1",
                material_type="assembly",
                confidence="low",
                rationale="高端智能手机通常包含金属中框、玻璃盖板、密封件和紧固件等外壳相关部件。",
                evidence=[],
                children=[],
            ),
            BomItem(
                name="显示总成",
                category="display",
                quantity="1",
                material_type="assembly",
                confidence="low",
                rationale="典型旗舰手机的显示堆栈通常包括盖板玻璃、OLED 面板、触控层、胶材和金属屏蔽件。",
                evidence=[],
                children=[],
            ),
            BomItem(
                name="主板总成",
                category="pcb",
                quantity="1",
                material_type="assembly",
                confidence="low",
                rationale="智能手机主板通常集成 SoC、存储器、电源管理、射频器件、连接器和 PCB 基材。",
                evidence=[],
                children=[],
            ),
            BomItem(
                name="电池包",
                category="battery",
                quantity="1",
                material_type="assembly",
                confidence="low",
                rationale="典型手机电池包通常采用锂离子软包结构，并包含铝箔、铜箔等集流体。",
                evidence=[],
                children=[],
            ),
            BomItem(
                name="摄像头模组组",
                category="camera",
                quantity="multiple modules",
                material_type="assembly",
                confidence="low",
                rationale="旗舰手机通常集成后摄模组、前摄、镜头筒、执行器和图像传感器。",
                evidence=[],
                children=[],
            ),
        ],
    )


def run_bom_decomposition(
    product_name: str,
    product_context: str | None = None,
    config: AgentConfig | None = None,
) -> BomDecomposition:
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
    output = raw.get("structured_response")
    if output is None:
        try:
            return _finalize_without_tools(active_config, product_name, product_context, raw)
        except (APITimeoutError, APIConnectionError, ValueError) as exc:
            return _fallback_bom(
                product_name,
                product_context,
                f"收尾步骤失败：{type(exc).__name__}: {exc}",
            )
    return BomDecomposition.model_validate(output)
