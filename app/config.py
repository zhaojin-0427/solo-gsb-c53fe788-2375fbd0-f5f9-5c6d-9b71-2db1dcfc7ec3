"""运行配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    db_path: str = os.environ.get("UNCERTAINTY_DB_PATH", "/data/uncertainty.db")


settings = Settings()
