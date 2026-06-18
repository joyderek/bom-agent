from __future__ import annotations

import argparse
import json
import sys

from src.config import AgentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于公开网页研究，将产品 BOM 拆解到尽可能接近原材料的粒度。"
    )
    parser.add_argument("product_name", help="要拆解的产品名，例如 'iPhone 15 Pro'")
    parser.add_argument(
        "--context",
        default=None,
        help="可选补充上下文，例如型号、代际或目标配置。",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="格式化输出结果 JSON。",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    from src.agent import run_bom_decomposition

    result = run_bom_decomposition(
        product_name=args.product_name,
        product_context=args.context,
        config=AgentConfig.from_env(),
    )
    if args.pretty:
        json.dump(result.model_dump(mode="json"), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(result.model_dump_json())
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
