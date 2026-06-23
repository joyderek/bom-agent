from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


ConfidenceLevel = Literal["high", "medium", "low"]
DecompositionKind = Literal["product", "system", "process", "supply_chain", "industry_chain", "generic"]
NodeType = Literal["end_item", "assembly", "module", "component", "material", "process", "service", "supplier", "stage", "other"]


class Evidence(BaseModel):
    url: HttpUrl
    title: str = Field(..., description="Short human-readable source title.")
    snippet: str = Field(
        ...,
        description="Concise evidence summary tied to the source, not a raw page dump.",
    )


class BomItem(BaseModel):
    name: str
    category: Optional[str] = Field(
        default=None,
        description="Optional functional or business grouping such as power, structure, packaging, final assembly, upstream wafer fabrication, logistics.",
    )
    node_type: NodeType = Field(
        ...,
        description="Generic node kind so the same schema can represent products, industrial systems, process steps, and supply-chain stages.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Short description of what this node is or does within the decomposition.",
    )
    quantity: Optional[str] = Field(
        default=None,
        description="Quantity if discoverable. Keep textual units when needed, e.g. '12 screws'.",
    )
    unit: Optional[str] = Field(
        default=None,
        description="Optional unit associated with quantity, capacity, throughput, or scale.",
    )
    role: Optional[str] = Field(
        default=None,
        description="Why this node matters within the parent structure, for example energy storage, structural support, distribution, contract manufacturing.",
    )
    stage: Optional[str] = Field(
        default=None,
        description="Optional lifecycle or supply-chain stage such as upstream, midstream, downstream, manufacturing, integration, distribution, after-sales.",
    )
    confidence: ConfidenceLevel
    rationale: str = Field(..., description="Why this item exists in the BOM.")
    evidence: List[Evidence] = Field(default_factory=list)
    children: List["BomItem"] = Field(default_factory=list)

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

    @field_validator("scope_notes", mode="before")
    @classmethod
    def normalize_scope_notes(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
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
