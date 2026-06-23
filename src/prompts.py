SYSTEM_PROMPT = """你是一个结构分解研究智能体。
你的职责是针对用户输入的对象，检索公开网页信息，并产出可追溯的研究材料。

执行要求：
- 优先依据公开证据进行拆解，再结合稳健的行业常识做补充推断。
- 分解对象可以是科技产品、传统工业产品、制造系统、工艺流程、供应链或产业链。
- 分解粒度应与对象类型匹配，不强制拆到原材料，也不强制只停留在一级模块。
- 结构可以按功能模块、物理结构、工艺阶段、供应链环节或价值链节点组织，选择最合理的一种。
- 不要为了凑数量而编造节点；证据不足时要明确说明不确定性。
- 回复内容使用中文。
- 本阶段只输出研究报告和证据摘要，不输出最终 JSON。
"""


def build_user_prompt(product_name: str, product_context: str | None = None) -> str:
    context = product_context.strip() if product_context else ""
    extra = f"\n已知上下文：\n{context}\n" if context else ""
    return f"""请对这个对象做一个尽可能通用、可解释的结构分解研究。

对象：{product_name}
{extra}
输出要求：
- 先判断该对象更适合按什么维度分解，例如功能模块、物理组成、工艺流程、供应链阶段、产业链环节。
- 输出层级要服务于理解对象本身，不必机械限定为一级，也不必强行拆到原材料层。
- 为关键结论附上证据 URL、标题和简短证据摘录。
- 对缺口、歧义、估算成分和不确定项写清楚范围说明。
- 请输出 Markdown 研究报告，至少包含：对象判断、分解维度选择、候选分解树、证据列表、不确定性和边界。
- 不要输出最终 JSON；下一阶段会基于你的研究报告和中间过程生成结构化结果。
"""


def build_structuring_prompt(
    product_name: str,
    product_context: str | None,
    research_output: str,
    intermediate_process: str,
) -> str:
    context = product_context.strip() if product_context else ""
    extra = f"\n已知上下文：\n{context}\n" if context else ""
    return f"""请基于一阶段研究报告和中间过程，生成最终 BomDecomposition 结构化结果。

对象：{product_name}
{extra}
结构化要求：
- 必须严格符合 BomDecomposition schema，由 structured output 机制返回，不要输出额外解释。
- 回复内容使用中文。
- subject_kind 只能使用 product、system、process、supply_chain、industry_chain、generic。
- node_type 只能使用 end_item、assembly、module、component、material、process、service、supplier、stage、other。
- confidence 只能使用 high、medium、low。
- evidence 只能使用一阶段材料中出现过的真实 HTTP/HTTPS URL；没有可靠 URL 时 evidence 使用空数组，并在 rationale 或 scope_notes 说明。
- 不要编造数量；只有一阶段材料明确支持数量或规模时才填写 quantity/unit。
- 如果填写 unit，必须同时填写 quantity。
- children 必须始终是数组，没有子节点时使用空数组。

一阶段研究报告：
{research_output}

一阶段中间过程摘要：
{intermediate_process}
"""
