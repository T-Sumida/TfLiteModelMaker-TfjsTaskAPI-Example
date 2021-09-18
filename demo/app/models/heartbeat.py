# -*- coding:utf-8 -*-

from pydantic import BaseModel


class HeartbeatResult(BaseModel):
    is_alive: bool
