from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v != "" else default


def _getenv_float(name: str, default: float) -> float:
    v = _getenv(name)
    return float(v) if v is not None else default


def _getenv_int(name: str, default: int) -> int:
    v = _getenv(name)
    return int(v) if v is not None else default


def _getenv_bool(name: str, default: bool) -> bool:
    v = _getenv(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class BotConfig:
    app_id: str
    token: Optional[str]
    symbol: str

    stake: float
    currency: str
    duration: int
    duration_unit: str

    adaptive_mode: bool
    min_edge: float
    min_confidence: float
    cooldown_seconds: int
    window_ticks: int

    dry_run: bool
    max_daily_loss: float
    take_profit: float
    max_trades_per_session: int


def load_config() -> BotConfig:
    return BotConfig(
        app_id=_getenv("DERIV_APP_ID", "1089") or "1089",
        token=_getenv("DERIV_TOKEN", None),
        symbol=_getenv("DERIV_SYMBOL", "R_50") or "R_50",
        stake=_getenv_float("STAKE", 0.35),
        currency=_getenv("CURRENCY", "USD") or "USD",
        duration=_getenv_int("DURATION", 1),
        duration_unit=_getenv("DURATION_UNIT", "t") or "t",
        adaptive_mode=_getenv_bool("ADAPTIVE_MODE", True),
        min_edge=_getenv_float("MIN_EDGE", 0.05),
        min_confidence=_getenv_float("MIN_CONFIDENCE", 0.15),
        cooldown_seconds=_getenv_int("COOLDOWN_SECONDS", 6),
        window_ticks=_getenv_int("WINDOW_TICKS", 250),
        dry_run=_getenv_bool("DRY_RUN", False),
        max_daily_loss=_getenv_float("MAX_DAILY_LOSS", 5.0),
        take_profit=_getenv_float("TAKE_PROFIT", 0.0),
        max_trades_per_session=_getenv_int("MAX_TRADES_PER_SESSION", 3),
    )

