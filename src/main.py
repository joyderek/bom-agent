from __future__ import annotations

import argparse
import sys

from src.config import AgentConfig
from src.models import BomDecomposition, BomItem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于公开网页研究，对对象进行通用结构分解，并输出结构化结果。"
    )
    parser.add_argument(
        "product_name",
        nargs="?",
        help="要拆解的对象名，例如 'iPhone 15 Pro'、'光伏逆变器'、'锂电产业链'；使用 --from-research 时可省略。",
    )
    parser.add_argument(
        "--context",
        default=None,
        help="可选补充上下文，例如型号、代际或目标配置。",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text", "run-json"),
        default="json",
        help="输出格式：json 为结构化输出，text 为可读文本，run-json 包含研究中间过程。",
    )
    parser.add_argument(
        "--save-research",
        help="保存一阶段研究结果到指定 JSON 文件；即使二阶段失败也会先写入。",
    )
    parser.add_argument(
        "--save-output",
        help="保存最终 BomDecomposition JSON 到指定文件；仅在二阶段成功后写入。",
    )
    parser.add_argument(
        "--save-run",
        help="保存完整运行结果到指定 JSON 文件，包含最终结果和一阶段中间过程；仅在二阶段成功后写入。",
    )
    parser.add_argument(
        "--from-research",
        help="跳过一阶段搜索，直接基于已保存的一阶段研究 JSON 重跑二阶段。",
    )
    return parser


def _render_node(node: BomItem) -> list[str]:
    lines = [f"- 名称：{node.name}"]
    lines.append(f"  描述：{node.description or '未明确'}")
    lines.append(f"  供应商：{node.supplier_market or '主要供应商和份额未明确'}")
    lines.append(f"  成本：{node.cost_share or '未明确'}")
    return lines


def render_text(result: BomDecomposition) -> str:
    lines = [
        f"对象: {result.subject_name}",
        f"类型: {result.subject_kind}",
        f"描述: {result.subject_description}",
    ]
    if result.decomposition_goal:
        lines.append(f"目标: {result.decomposition_goal}")
    if result.decomposition_basis:
        lines.append(f"分解依据: {result.decomposition_basis}")
    if result.depth_policy:
        lines.append(f"深度策略: {result.depth_policy}")
    if result.scope_notes:
        lines.append("范围说明:")
        for note in result.scope_notes:
            lines.append(f"- {note}")
    lines.append("分解结果:")
    for node in result.top_level_nodes:
        lines.extend(_render_node(node))
    return "\n".join(lines)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.product_name and not args.from_research:
        parser.error("缺少 product_name；只有使用 --from-research 时才可省略。")

    from src.agent import (
        run_bom_decomposition,
        run_bom_decomposition_with_trace,
        structure_bom_from_research_trace_file,
    )

    config = AgentConfig.from_env()
    if args.from_research:
        run_result = structure_bom_from_research_trace_file(
            path=args.from_research,
            product_name=args.product_name,
            product_context=args.context,
            config=config,
            output_path=args.save_output,
            run_output_path=args.save_run,
        )
        if args.format == "run-json":
            sys.stdout.write(run_result.model_dump_json(indent=2))
        elif args.format == "json":
            sys.stdout.write(run_result.decomposition.model_dump_json(indent=2))
        else:
            sys.stdout.write(render_text(run_result.decomposition))
    elif args.format == "run-json":
        run_result = run_bom_decomposition_with_trace(
            product_name=args.product_name or "",
            product_context=args.context,
            config=config,
            research_trace_path=args.save_research,
            output_path=args.save_output,
            run_output_path=args.save_run,
        )
        sys.stdout.write(run_result.model_dump_json(indent=2))
    else:
        result = run_bom_decomposition(
            product_name=args.product_name or "",
            product_context=args.context,
            config=config,
            research_trace_path=args.save_research,
            output_path=args.save_output,
            run_output_path=args.save_run,
        )
        if args.format == "json":
            sys.stdout.write(result.model_dump_json(indent=2))
        elif args.format == "text":
            sys.stdout.write(render_text(result))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
