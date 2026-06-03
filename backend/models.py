"""Pydantic request schemas for the control-panel API."""
from __future__ import annotations

from pydantic import BaseModel


class StartReq(BaseModel):
    task: str
    start_url: str | None = None
