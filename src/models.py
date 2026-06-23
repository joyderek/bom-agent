from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


ConfidenceLevel = Literal["high", "medium", "low"]
DecompositionKind = Literal["product", "system", "process", "supply_chain", "industry_chain", "generic"]
NodeType = Literal["end_item", "assembly", "module", "component", "material", "process", "service", "supplier", "stage", "other"]

VALID_NODE_TYPES = {
    "end_item",
    "assembly",
    "module",
    "component",
    "material",
    "process",
    "service",
    "supplier",
    "stage",
    "other",
}

NODE_TYPE_ALIASES = {
    "system": "assembly",
    "subsystem": "module",
    "sub_system": "module",
    "part": "component",
    "raw_material": "material",
    "resource": "material",
    "vendor": "supplier",
    "manufacturer": "supplier",
    "step": "stage",
}


def _is_usable_evidence_input(value: object) -> bool:
    if isinstance(value, str):
        text = value.strip()
        return text.startswith("http://") or text.startswith("https://")
    if isinstance(value, dict):
        url = value.get("url")
        return isinstance(url, str) and (url.startswith("http://") or url.startswith("https://"))
    return isinstance(value, Evidence)


class Evidence(BaseModel):
    url: HttpUrl
    title: str = Field(..., description="Short human-readable source title.")
    snippet: str = Field(
        ...,
        description="Concise evidence summary tied to the source, not a raw page dump.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_evidence(cls, value: object) -> object:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("http://") or text.startswith("https://"):
                return {"url": text, "title": text, "snippet": text}
        return value


class SupplierMarketShare(BaseModel):
    supplier_name: str = Field(..., description="Company or organization name.")
    market_position: Optional[str] = Field(
        default=None,
        description="Qualitative position such as leader, top tier, regional specialist, emerging supplier, or niche player.",
    )
    market_share: Optional[str] = Field(
        default=None,
        description="Market share estimate as reported or inferred from evidence, for example '约 20%' or 'top 5 by shipments'.",
    )
    geography: Optional[str] = Field(
        default=None,
        description="Relevant market geography such as global, China, Europe, North America, or application segment.",
    )
    rationale: str = Field(..., description="Why this supplier is relevant to this module.")
    confidence: ConfidenceLevel = "medium"
    evidence: List[Evidence] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_supplier_aliases(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "supplier_name" not in normalized:
            for alias in ("name", "company", "vendor", "manufacturer"):
                if isinstance(normalized.get(alias), str):
                    normalized["supplier_name"] = normalized[alias]
                    break
        if "market_share" not in normalized:
            for alias in ("share", "market_share_estimate"):
                if isinstance(normalized.get(alias), str):
                    normalized["market_share"] = normalized[alias]
                    break
        if "rationale" not in normalized:
            normalized["rationale"] = "模型基于研究材料和行业常识识别为该模块的代表性供应商。"
        if "confidence" not in normalized:
            normalized["confidence"] = "medium"
        if "evidence" in normalized and isinstance(normalized["evidence"], list):
            normalized["evidence"] = [item for item in normalized["evidence"] if _is_usable_evidence_input(item)]
        return normalized


class BomItem(BaseModel):
    name: str = Field(..., description="Direct downstream item name.")
    description: Optional[str] = Field(
        default=None,
        description="Brief description of the direct downstream item, including key specs when available.",
    )
    supplier_market: Optional[str] = Field(
        default=None,
        description="Current market supply situation in one concise sentence, including major suppliers and shares when available.",
    )
    cost_share: Optional[str] = Field(
        default=None,
        description="Estimated cost share of this direct downstream item in the parent product or system, e.g. '30-40%'.",
    )
    category: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Optional functional or business grouping such as power, structure, packaging, final assembly, upstream wafer fabrication, logistics.",
    )
    node_type: NodeType = Field(
        default="component",
        exclude=True,
        description="Generic node kind so the same schema can represent products, industrial systems, process steps, and supply-chain stages.",
    )
    quantity: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Quantity if discoverable. Keep textual units when needed, e.g. '12 screws'.",
    )
    unit: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Optional unit associated with quantity, capacity, throughput, or scale.",
    )
    role: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Why this node matters within the parent structure, for example energy storage, structural support, distribution, contract manufacturing.",
    )
    stage: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Optional lifecycle or supply-chain stage such as upstream, midstream, downstream, manufacturing, integration, distribution, after-sales.",
    )
    confidence: ConfidenceLevel = Field(default="medium", exclude=True)
    rationale: Optional[str] = Field(default=None, exclude=True, description="Why this item exists in the BOM.")
    evidence: List[Evidence] = Field(default_factory=list, exclude=True)
    market_analysis: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Short market structure summary for this module or component, including concentration and share caveats when available.",
    )
    suppliers: List[SupplierMarketShare] = Field(default_factory=list, exclude=True)
    children: List["BomItem"] = Field(default_factory=list, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_item_defaults(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "confidence" not in normalized:
            normalized["confidence"] = "medium"
        if "rationale" not in normalized:
            normalized["rationale"] = normalized.get("description") or "模型基于研究材料和通用结构知识判断该节点属于 BOM 分解。"
        if "evidence" in normalized and isinstance(normalized["evidence"], list):
            normalized["evidence"] = [item for item in normalized["evidence"] if _is_usable_evidence_input(item)]
        if "suppliers" in normalized and isinstance(normalized["suppliers"], list):
            normalized["suppliers"] = [item for item in normalized["suppliers"] if isinstance(item, dict)]
        if "supplier_market" not in normalized:
            for alias in ("supplier", "suppliers_summary", "market_supply", "current_market_supply"):
                if isinstance(normalized.get(alias), str):
                    normalized["supplier_market"] = normalized[alias]
                    break
        if "cost_share" not in normalized:
            for alias in ("cost", "cost_ratio", "cost_percentage", "cost_percent", "bom_cost_share"):
                if isinstance(normalized.get(alias), str):
                    normalized["cost_share"] = normalized[alias]
                    break
        if "supplier_market" not in normalized:
            market_parts = []
            if isinstance(normalized.get("market_analysis"), str):
                market_parts.append(normalized["market_analysis"])
            suppliers = normalized.get("suppliers")
            if isinstance(suppliers, list) and suppliers:
                supplier_parts = []
                for supplier in suppliers:
                    if not isinstance(supplier, dict):
                        continue
                    name = supplier.get("supplier_name") or supplier.get("name") or supplier.get("company")
                    share = supplier.get("market_share") or supplier.get("share")
                    if isinstance(name, str) and isinstance(share, str):
                        supplier_parts.append(f"{name}{share}")
                    elif isinstance(name, str):
                        supplier_parts.append(name)
                if supplier_parts:
                    market_parts.append("主要供应商：" + "、".join(supplier_parts))
            if market_parts:
                normalized["supplier_market"] = "；".join(market_parts)
        return normalized

    @field_validator("node_type", mode="before")
    @classmethod
    def normalize_node_type(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
            return NODE_TYPE_ALIASES.get(normalized, normalized if normalized in VALID_NODE_TYPES else "other")
        return value

    @model_validator(mode="after")
    def validate_quantity_unit_pair(self) -> "BomItem":
        if self.unit and not self.quantity:
            raise ValueError("unit should only be provided when quantity or scale is also provided.")
        return self


class BomDecomposition(BaseModel):
    subject_name: str = Field(..., description="Name of the thing being decomposed, such as a product, industrial asset, process, or industry chain.")
    subject_kind: DecompositionKind = Field(
        default="generic",
        description="High-level classification of the decomposition target.",
    )
    subject_description: str = Field(
        ...,
        description="Short description of the decomposition target and the framing used for this analysis.",
    )
    decomposition_goal: Optional[str] = Field(
        default=None,
        description="What this decomposition is intended to reveal, for example physical structure, process flow, cost stack, or supply-chain stages.",
    )
    decomposition_basis: Optional[str] = Field(
        default=None,
        description="Primary organizing principle used for the hierarchy, such as functional modules, manufacturing stages, process flow, or value chain.",
    )
    depth_policy: Optional[str] = Field(
        default=None,
        description="How deep the hierarchy goes and where the decomposition intentionally stops.",
    )
    scope_notes: List[str] = Field(
        default_factory=list,
        description="Assumptions, unknowns, and explicit boundary notes.",
    )
    top_level_nodes: List[BomItem] = Field(
        default_factory=list,
        description="Top-level nodes in the decomposition tree.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_top_level_aliases(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        subject = normalized.get("subject")
        if "subject_name" not in normalized and isinstance(subject, str):
            normalized["subject_name"] = subject
        if "subject_description" not in normalized:
            for alias in ("description", "summary", "subject_desc"):
                if isinstance(normalized.get(alias), str):
                    normalized["subject_description"] = normalized[alias]
                    break
        if "subject_description" not in normalized:
            normalized["subject_description"] = "未明确"
        if "subject_description" not in normalized and isinstance(subject, dict):
            for alias in ("description", "summary", "name"):
                if isinstance(subject.get(alias), str):
                    normalized["subject_description"] = subject[alias]
                    break
        if "subject_name" not in normalized and isinstance(subject, dict):
            for alias in ("name", "title"):
                if isinstance(subject.get(alias), str):
                    normalized["subject_name"] = subject[alias]
                    break
        if "top_level_nodes" not in normalized:
            for alias in ("nodes", "top_level_bom", "items", "children"):
                if isinstance(normalized.get(alias), list):
                    normalized["top_level_nodes"] = normalized[alias]
                    break
        return normalized

    @field_validator("subject_description", mode="before")
    @classmethod
    def normalize_subject_description(cls, value: object) -> object:
        if value is None:
            return "未明确"
        if isinstance(value, str) and not value.strip():
            return "未明确"
        return value

    @field_validator("scope_notes", mode="before")
    @classmethod
    def normalize_scope_notes(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [item for item in value if not isinstance(item, str) or item.strip()]
        return value


class ResearchTraceMessage(BaseModel):
    role: str
    message_type: str
    content: str = ""
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: List[dict[str, Any]] = Field(default_factory=list)


class BomResearchTrace(BaseModel):
    product_name: str = ""
    product_context: Optional[str] = None
    research_output: str
    intermediate_messages: List[ResearchTraceMessage] = Field(default_factory=list)


class BomDecompositionRun(BaseModel):
    decomposition: BomDecomposition
    research: BomResearchTrace


BomItem.model_rebuild()
