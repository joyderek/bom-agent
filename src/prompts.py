SYSTEM_PROMPT = """你是一个结构分解研究智能体。
你的职责是针对用户输入的对象，检索公开网页信息，并产出可追溯的研究材料。

执行要求：
- 优先依据公开证据进行拆解，再结合稳健的行业常识做补充推断。
- 分解对象可以是科技产品、传统工业产品、制造系统、工艺流程、供应链或产业链。
- 永远只拆解到直接下游一级，不要继续拆子模块、零件或原材料。
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
- 只输出对象的直接下游组成/环节，不要继续拆解下游节点。
- 同时研究每个直接下游节点的当前市场供应情况、成本占比和关键信息来源（URL），优先包含主要供应商、市占/供货比例、BOM 成本占比范围。
- 为关键结论附上证据 URL、标题和简短证据摘录，并在 sources 中列出信息来源名称和链接。
- 对缺口、歧义、估算成分和不确定项写清楚范围说明。
- 请输出 Markdown 研究报告，至少包含：对象判断、直接下游清单、每项供应商/市占、每项成本占比、每项信息来源（名称+URL）、证据列表、不确定性和边界。
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
- top_level_nodes 只允许包含对象的直接下游一级节点；不要给任何节点填 children。
- 每个 top_level_nodes 节点只需要五个业务字段：name、description、supplier_market、cost_share、sources。
- name 使用直接下游名称，例如「显示屏」。
- description 用一句话描述规格、功能、位置或典型配置，例如「6.1 英寸 OLED 显示屏，支持高刷新率」。
- supplier_market 用一句话描述当前市场供应情况，例如「约 80% 由三星供应，约 20% 由京东方供应」；没有可靠比例时写主要供应商和不确定性。
- cost_share 用短文本描述该直接下游在父对象 BOM 成本中的占比，例如「30-40%」；没有可靠范围时写 null。
- sources 列出关键信息来源，每项包含 name 和 url，例如 [{{"name": "平安证券", "url": "https://..."}}, {{"name": "雪球", "url": "https://..."}}]；没有可靠来源时写空数组或 null。
- 不要输出多层 BOM，不要拆到间接下游，不要列原材料层。
- subject_kind 只能使用 product、system、process、supply_chain、industry_chain、generic。
- 兼容字段如 node_type、confidence、rationale、evidence、market_analysis、suppliers、children 可以留空或使用默认值。

一阶段研究报告：
{research_output}

一阶段中间过程摘要：
{intermediate_process}
"""


def build_supplier_enrichment_prompt(
    product_name: str,
    product_context: str | None,
    research_output: str,
    decomposition_json: str,
) -> str:
    context = product_context.strip() if product_context else ""
    extra = f"\n已知上下文：\n{context}\n" if context else ""
    return f"""请在已有 BomDecomposition 结构化结果基础上，只补充直接下游一级节点的当前市场供应情况和成本占比。

对象：{product_name}
{extra}
任务要求：
- 返回完整 BomDecomposition，不要只返回增量字段。
- top_level_nodes 只保留直接下游一级节点；删除/忽略所有 children。
- 每个 top_level_nodes 节点只需要 name、description、supplier_market、cost_share、sources 五个业务字段。
- supplier_market 用一句中文概括主要供应商、供货比例、市占或行业排名；例如「约 80% 由三星供应，约 20% 由京东方供应」。
- cost_share 用短文本概括该直接下游在父对象 BOM 成本中的占比，例如「30-40%」；只有行业资料或稳健成本拆分支持时填写，缺乏依据时写 null。
- sources 列出关键信息来源，每项包含 name 和 url，例如 [{{"name": "平安证券", "url": "https://..."}}]；没有可靠来源时写空数组或 null。
- 只有公开材料或稳健行业常识支持时才写具体比例；没有可靠比例时写「主要供应商包括……，具体份额未公开」。
- 不要为了凑数量编造供应商；优先列 1-4 个最重要供应商。
- 必须严格符合 BomDecomposition schema，由 structured output 机制返回，不要输出额外解释。

一阶段研究报告：
{research_output}

已有 BomDecomposition JSON：
{decomposition_json}
"""
