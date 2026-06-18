from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


ConfidenceLevel = Literal["high", "medium", "low"]


class Evidence(BaseModel):
    url: HttpUrl
    title: str = Field(..., description="Short human-readable source title.")
    snippet: str = Field(
        ...,
        description="Concise evidence summary tied to the source, not a raw page dump.",
    )


class BomItem(BaseModel):
    name: str
    category: str = Field(
        ...,
        description="Functional grouping such as enclosure, battery, PCB, display, fastener.",
    )
    quantity: Optional[str] = Field(
        default=None,
        description="Quantity if discoverable. Keep textual units when needed, e.g. '12 screws'.",
    )
    material_type: Literal["assembly", "component", "subcomponent", "raw_material"]
    confidence: ConfidenceLevel
    rationale: str = Field(..., description="Why this item exists in the BOM.")
    evidence: List[Evidence] = Field(default_factory=list)
    children: List["BomItem"] = Field(default_factory=list)


class BomDecomposition(BaseModel):
    product_name: str
    product_description: str
    scope_notes: List[str] = Field(
        default_factory=list,
        description="Assumptions, unknowns, and explicit boundary notes.",
    )
    top_level_bom: List[BomItem]


BomItem.model_rebuild()
