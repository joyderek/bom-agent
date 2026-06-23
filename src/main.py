from __future__ import annotations

import argparse
import sys

from src.config import AgentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于公开网页研究，将产品 BOM 拆解到直接下游模块粒度，并输出纯文本结果。"
    )
    parser.add_argument("product_name", help="要拆解的产品名，例如 'iPhone 15 Pro'")
    parser.add_argument(
        "--context",
        default=None,
        help="可选补充上下文，例如型号、代际或目标配置。",
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
    sys.stdout.write(result)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
