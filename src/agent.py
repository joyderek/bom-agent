from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError
from langchain.agents import create_agent
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langchain_openai import ChatOpenAI
from src.config import AgentConfig
from src.middleware import ResearchGuardMiddleware, research_guard_after_model
from src.models import (
    BomDecomposition,
    BomDecompositionRun,
    BomResearchTrace,
    ResearchTraceMessage,
)
from src.prompts import (
    SYSTEM_PROMPT,
    build_structuring_prompt,
    build_supplier_enrichment_prompt,
    build_user_prompt,
)
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


def _parse_bom_decomposition(raw_text: str) -> BomDecomposition:
    text = raw_text.strip()
    if not text:
        raise RuntimeError("模型返回了空文本，无法解析为结构化结果。")

    def _extract_fenced_blocks(value: str) -> list[str]:
        blocks: list[str] = []
        marker = "```"
        start = 0
        while True:
            fence_start = value.find(marker, start)
            if fence_start == -1:
                break
            line_end = value.find("\n", fence_start)
            if line_end == -1:
                break
            fence_end = value.find(marker, line_end + 1)
            if fence_end == -1:
                break
            block = value[line_end + 1 : fence_end].strip()
            if block.lower().startswith("json"):
                block = block[4:].strip()
            if block:
                blocks.append(block)
            start = fence_end + len(marker)
        return blocks

    def _extract_balanced_json(value: str) -> list[str]:
        results: list[str] = []
        for opener, closer in (("{", "}"), ("[", "]")):
            search_start = 0
            while True:
                start = value.find(opener, search_start)
                if start == -1:
                    break
                depth = 0
                in_string = False
                escape = False
                found = False
                for index in range(start, len(value)):
                    char = value[index]
                    if in_string:
                        if escape:
                            escape = False
                        elif char == "\\":
                            escape = True
                        elif char == '"':
                            in_string = False
                        continue
                    if char == '"':
                        in_string = True
                        continue
                    if char == opener:
                        depth += 1
                    elif char == closer:
                        depth -= 1
                        if depth == 0:
                            results.append(value[start : index + 1].strip())
                            search_start = start + 1
                            found = True
                            break
                if not found:
                    break
        return results

    candidate_texts: list[str] = [text]
    candidate_texts.extend(_extract_fenced_blocks(text))
    candidate_texts.extend(_extract_balanced_json(text))

    seen: set[str] = set()
    deduped_candidates: list[str] = []
    for candidate in candidate_texts:
        normalized = candidate.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped_candidates.append(normalized)

    decoder = json.JSONDecoder()
    for candidate in deduped_candidates:
        try:
            return _normalize_decomposition_tree(BomDecomposition.model_validate_json(candidate))
        except Exception:
            pass
        try:
            parsed, _ = decoder.raw_decode(candidate)
            return _normalize_decomposition_tree(BomDecomposition.model_validate(parsed))
        except Exception:
            pass

    try:
        return _normalize_decomposition_tree(BomDecomposition.model_validate_json(text))
    except Exception as first_error:
        try:
            return _normalize_decomposition_tree(BomDecomposition.model_validate(json.loads(text)))
        except Exception as second_error:
            raise RuntimeError(
                "模型返回内容未能通过 BomDecomposition 校验。"
                f" first_error={first_error}; second_error={second_error}; "
                f"raw_prefix={text[:300]!r}"
            ) from second_error


def _repair_bom_output(
    raw_text: str,
    config: AgentConfig,
) -> BomDecomposition:
    model = _build_model(config)
    repair_prompt = (
        "你会收到一段本应表示 BomDecomposition 的输出，但它不是可直接解析的合法 JSON。"
        "请提取其中已有信息并重写为一个合法 JSON 对象。"
        "不要输出 Markdown，不要输出代码块，不要输出解释文字，只返回 JSON 对象本身。\n\n"
        "必须满足这些要求：\n"
        "- 顶层至少包含 subject_name, subject_kind, subject_description, scope_notes, top_level_nodes\n"
        "- top_level_nodes 只保留直接下游一级节点\n"
        "- 每个节点只需要 name, description, supplier_market, cost_share, sources 五个业务字段\n"
        "- sources 是数组，元素包含 name 和 url\n"
        "- 不要输出 children，不要输出多层 BOM\n\n"
        "待修复内容如下：\n"
        f"{raw_text}"
    )
    response = model.invoke(repair_prompt)
    repaired_text = _extract_text_from_message(response)
    return _parse_bom_decomposition(repaired_text)


def _message_role(message: Any) -> str:
    message_type = getattr(message, "type", "")
    if message_type == "human" or isinstance(message, HumanMessage):
        return "user"
    if message_type == "ai" or isinstance(message, AIMessage):
        return "assistant"
    if message_type == "tool" or isinstance(message, ToolMessage):
        return "tool"
    return str(message_type or type(message).__name__)


def _serialize_tool_calls(message: Any) -> list[dict[str, Any]]:
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return []
    serializable: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            serializable.append(json.loads(json.dumps(tool_call, ensure_ascii=False, default=str)))
        else:
            serializable.append(
                json.loads(
                    json.dumps(
                        {
                            "name": getattr(tool_call, "name", None),
                            "args": getattr(tool_call, "args", None),
                            "id": getattr(tool_call, "id", None),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                )
            )
    return serializable


def _build_research_trace(messages: list[Any]) -> BomResearchTrace:
    intermediate_messages: list[ResearchTraceMessage] = []
    final_research_output = ""

    for message in messages:
        content = _extract_text_from_message(message)
        trace_message = ResearchTraceMessage(
            role=_message_role(message),
            message_type=str(getattr(message, "type", type(message).__name__)),
            content=content,
            tool_name=getattr(message, "name", None),
            tool_call_id=getattr(message, "tool_call_id", None),
            tool_calls=_serialize_tool_calls(message),
        )
        intermediate_messages.append(trace_message)
        if isinstance(message, AIMessage) and content:
            final_research_output = content

    if not final_research_output:
        ai_outputs = [
            trace_message.content
            for trace_message in intermediate_messages
            if trace_message.role == "assistant" and trace_message.content
        ]
        final_research_output = "\n\n".join(ai_outputs).strip()

    if not final_research_output:
        final_research_output = "\n\n".join(
            trace_message.content
            for trace_message in intermediate_messages
            if trace_message.content
        ).strip()

    return BomResearchTrace(
        research_output=final_research_output,
        intermediate_messages=intermediate_messages,
    )


def save_research_trace(trace: BomResearchTrace, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(trace.model_dump_json(indent=2), encoding="utf-8")


def save_bom_decomposition(decomposition: BomDecomposition, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(decomposition.model_dump_json(indent=2), encoding="utf-8")


def save_bom_run(run: BomDecompositionRun, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(run.model_dump_json(indent=2), encoding="utf-8")


def load_research_trace(path: str | Path) -> BomResearchTrace:
    return BomResearchTrace.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _format_intermediate_process(trace: BomResearchTrace) -> str:
    parts: list[str] = []
    for index, message in enumerate(trace.intermediate_messages, start=1):
        title = f"[{index}] role={message.role}, type={message.message_type}"
        if message.tool_name:
            title += f", tool={message.tool_name}"
        if message.tool_calls:
            title += f", tool_calls={json.dumps(message.tool_calls, ensure_ascii=False, default=str)}"
        content = message.content.strip()
        if content:
            parts.append(f"{title}\n{content}")
        else:
            parts.append(title)
    return "\n\n".join(parts).strip()


def _structure_bom_output(
    product_name: str,
    product_context: str | None,
    trace: BomResearchTrace,
    config: AgentConfig,
) -> BomDecomposition:
    model = _build_model(config)
    prompt = build_structuring_prompt(
        product_name=product_name,
        product_context=product_context,
        research_output=trace.research_output,
        intermediate_process=_format_intermediate_process(trace),
    )
    errors: list[str] = []
    for method, kwargs in (
        ("json_schema", {"strict": True}),
        ("json_mode", {}),
    ):
        try:
            structured_model = model.with_structured_output(
                BomDecomposition,
                method=method,
                **kwargs,
            )
            structured = structured_model.invoke(prompt)
            if isinstance(structured, BomDecomposition):
                return _normalize_decomposition_tree(structured)
            return _normalize_decomposition_tree(BomDecomposition.model_validate(structured))
        except Exception as exc:
            errors.append(f"{method}: {type(exc).__name__}: {exc}")

    try:
        response = model.invoke(prompt)
        text = _extract_text_from_message(response)
        try:
            return _normalize_decomposition_tree(_parse_bom_decomposition(text))
        except RuntimeError:
            return _normalize_decomposition_tree(_repair_bom_output(text, config))
    except Exception as exc:
        errors.append(f"plain_json_repair: {type(exc).__name__}: {exc}")
        raise RuntimeError("; ".join(errors)) from exc


def _enrich_bom_suppliers(
    decomposition: BomDecomposition,
    product_name: str,
    product_context: str | None,
    trace: BomResearchTrace,
    config: AgentConfig,
) -> BomDecomposition:
    model = _build_model(config)
    prompt = build_supplier_enrichment_prompt(
        product_name=product_name,
        product_context=product_context,
        research_output=trace.research_output,
        decomposition_json=decomposition.model_dump_json(indent=2),
    )
    errors: list[str] = []
    for method, kwargs in (
        ("json_schema", {"strict": True}),
        ("json_mode", {}),
    ):
        try:
            structured_model = model.with_structured_output(
                BomDecomposition,
                method=method,
                **kwargs,
            )
            structured = structured_model.invoke(prompt)
            if isinstance(structured, BomDecomposition):
                return _normalize_decomposition_tree(structured)
            return _normalize_decomposition_tree(BomDecomposition.model_validate(structured))
        except Exception as exc:
            errors.append(f"{method}: {type(exc).__name__}: {exc}")

    try:
        response = model.invoke(prompt)
        text = _extract_text_from_message(response)
        return _normalize_decomposition_tree(_parse_bom_decomposition(text))
    except Exception as exc:
        errors.append(f"plain_json: {type(exc).__name__}: {exc}")
        decomposition.scope_notes.append(
            "供应商与市占补充阶段失败，已保留原始结构化分解结果；"
            f"失败原因：{'; '.join(errors)[:1200]}"
        )
        return decomposition


def _normalize_decomposition_tree(decomposition: BomDecomposition) -> BomDecomposition:
    if len(decomposition.top_level_nodes) == 1:
        root = decomposition.top_level_nodes[0]
        root_name = root.name.strip().lower()
        subject_name = decomposition.subject_name.strip().lower()
        if root.children and root.node_type == "end_item" and (subject_name in root_name or root_name in subject_name):
            decomposition.top_level_nodes = root.children
            if root.rationale and root.rationale not in decomposition.scope_notes:
                decomposition.scope_notes.insert(0, root.rationale)

    for node in decomposition.top_level_nodes:
        node.children = []
    return decomposition


def run_bom_decomposition(
    product_name: str,
    product_context: str | None = None,
    config: AgentConfig | None = None,
    research_trace_path: str | Path | None = None,
    output_path: str | Path | None = None,
    run_output_path: str | Path | None = None,
) -> BomDecomposition:
    run = run_bom_decomposition_with_trace(
        product_name=product_name,
        product_context=product_context,
        config=config,
        research_trace_path=research_trace_path,
        output_path=output_path,
        run_output_path=run_output_path,
    )
    return run.decomposition


def run_bom_decomposition_with_trace(
    product_name: str,
    product_context: str | None = None,
    config: AgentConfig | None = None,
    research_trace_path: str | Path | None = None,
    output_path: str | Path | None = None,
    run_output_path: str | Path | None = None,
) -> BomDecompositionRun:
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
    except APITimeoutError as exc:
        raise RuntimeError(
            "模型请求超时，错误类型为 APITimeoutError。"
            f"base_url={active_config.base_url!r}, model={active_config.model!r}, "
            f"timeout={active_config.model_timeout_seconds}s, retries={active_config.model_max_retries}. "
            "这通常表示单次模型响应在当前超时阈值内没有返回完成。"
            "请尝试提高 model_timeout_seconds，增加 model_max_retries，"
            "或减少单次请求复杂度，例如降低搜索页数、搜索结果数、工具调用上限。"
        ) from exc
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
    trace = _build_research_trace(messages)
    trace.product_name = product_name
    trace.product_context = product_context
    if research_trace_path is not None:
        save_research_trace(trace, research_trace_path)
    if trace.research_output:
        try:
            decomposition = _structure_bom_output(
                product_name=product_name,
                product_context=product_context,
                trace=trace,
                config=active_config,
            )
            decomposition = _enrich_bom_suppliers(
                decomposition=decomposition,
                product_name=product_name,
                product_context=product_context,
                trace=trace,
                config=active_config,
            )
        except Exception as exc:
            saved_hint = (
                f" 一阶段结果已保存到 {research_trace_path}，可使用 --from-research 重跑二阶段。"
                if research_trace_path is not None
                else " 可使用 --save-research 保存一阶段结果，之后用 --from-research 重跑二阶段。"
            )
            raise RuntimeError(
                f"二阶段结构化输出失败：{type(exc).__name__}: {exc}.{saved_hint}"
            ) from exc
        run = BomDecompositionRun(decomposition=decomposition, research=trace)
        if output_path is not None:
            save_bom_decomposition(decomposition, output_path)
        if run_output_path is not None:
            save_bom_run(run, run_output_path)
        return run
    last_message = messages[-1] if messages else None
    raise RuntimeError(
        "Agent 执行结束后没有返回可用文本结果。"
        f"message_count={len(messages)}, "
        f"last_message_type={type(last_message).__name__ if last_message is not None else 'None'}."
    )


def structure_bom_from_research_trace(
    trace: BomResearchTrace,
    product_name: str | None = None,
    product_context: str | None = None,
    config: AgentConfig | None = None,
    output_path: str | Path | None = None,
    run_output_path: str | Path | None = None,
) -> BomDecompositionRun:
    active_config = config or AgentConfig.from_env()
    effective_product_name = product_name or trace.product_name
    if not effective_product_name:
        raise RuntimeError("缺少 product_name，无法基于一阶段结果重跑二阶段。")
    effective_product_context = product_context if product_context is not None else trace.product_context
    decomposition = _structure_bom_output(
        product_name=effective_product_name,
        product_context=effective_product_context,
        trace=trace,
        config=active_config,
    )
    decomposition = _enrich_bom_suppliers(
        decomposition=decomposition,
        product_name=effective_product_name,
        product_context=effective_product_context,
        trace=trace,
        config=active_config,
    )
    run = BomDecompositionRun(decomposition=decomposition, research=trace)
    if output_path is not None:
        save_bom_decomposition(decomposition, output_path)
    if run_output_path is not None:
        save_bom_run(run, run_output_path)
    return run


def structure_bom_from_research_trace_file(
    path: str | Path,
    product_name: str | None = None,
    product_context: str | None = None,
    config: AgentConfig | None = None,
    output_path: str | Path | None = None,
    run_output_path: str | Path | None = None,
) -> BomDecompositionRun:
    trace = load_research_trace(path)
    return structure_bom_from_research_trace(
        trace=trace,
        product_name=product_name,
        product_context=product_context,
        config=config,
        output_path=output_path,
        run_output_path=run_output_path,
    )
