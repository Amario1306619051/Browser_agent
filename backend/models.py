"""Pydantic request schemas for the control-panel API."""
from __future__ import annotations

from pydantic import BaseModel


class StartReq(BaseModel):
    task: str
    start_url: str | None = None
    thread_id: str | None = None
    unlimited: bool = False
    scroll_speed: str | None = None
    scroll_delay: float | None = None
    scroll_distance: int | None = None
    smart: bool = True
    vision: bool = True


class GotoReq(BaseModel):
    url: str
