from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator


class SiteType(str, Enum):
    CONTROL = "control"
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"
    BASELINE = "baseline"


class Measurement(BaseModel):
    """Schema for a single water quality measurement."""

    timestamp: datetime
    site_id: str
    site_type: SiteType

    pH: float = Field(..., ge=0, le=14, description="Acidity/alkalinity (0-14)")
    conductivity_uScm: float = Field(..., ge=0, description="Conductivity (µS/cm)")

    Fe_mg_L: float = Field(..., ge=0, description="Dissolved iron (mg/L)")
    Mn_mg_L: float = Field(..., ge=0, description="Dissolved manganese (mg/L)")
    Al_mg_L: float = Field(..., ge=0, description="Dissolved aluminum (mg/L)")
    sulfate_mg_L: float = Field(..., ge=0, description="Sulfate (mg/L)")

    @field_validator("site_id")
    @classmethod
    def _strip_site_id(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("site_id cannot be empty")
        return s

    def to_record(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["timestamp"] = self.timestamp.isoformat()
        return d

