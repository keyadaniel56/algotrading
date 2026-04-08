from __future__ import annotations

import asyncio
import os
import time
from typing import Deque, Optional

from ai_logic import DigitsAI, decision_from_scores
from config import load_config
from deriv_ws import DerivAPIError, DerivWS, now_ts


def _load_dotenv_if_present(path: str = ".env") -> None:
    # Minimal dotenv loader to avoid adding dependencies.
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def _barrier_for_action(action: str) -> str:
    # Digits Over / Under barriers are the digit threshold.
    if action == "OVER_5":
        return "5"
    if action == "UNDER_4":
        return "4"
    raise ValueError(f"Unknown action: {action}")


def _contract_type_for_action(action: str) -> str:
    # Deriv: DIGITOVER / DIGITUNDER for digits contracts.
    if action == "OVER_5":
        return "DIGITOVER"
    if action == "UNDER_4":
        return "DIGITUNDER"
    raise ValueError(f"Unknown action: {action}")


async def main() -> None:
    # Always load the real runtime env file (".env") from the project directory.
    # ".env.example" is only a template and is never loaded.
    project_env = os.path.join(os.path.dirname(__file__), ".env")
    _load_dotenv_if_present(project_env)
    cfg = load_config()

    if not cfg.token:
        print("Missing DERIV_TOKEN. Set it in .env (use a DEMO token first).")
        return

    ws = DerivWS(app_id=cfg.app_id, token=cfg.token)
    ai = DigitsAI()

    last_digits: Deque[int]
    from collections import deque

    # Keep a generous history and let the model pick its own effective window.
    # (In adaptive mode we don't want to depend on WINDOW_TICKS.)
    max_hist = max(500, cfg.window_ticks) if not cfg.adaptive_mode else 500
    last_digits = deque(maxlen=max(max_hist, 60))

    last_trade_ts: float = 0.0
    last_trade_epoch: Optional[int] = None
    trades = 0
    realized_pnl = 0.0
    ticks_seen = 0
    trade_lock = asyncio.Lock()

    print(
        "Starting Deriv Digits bot\n"
        f"symbol={cfg.symbol} stake={cfg.stake} {cfg.currency} duration={cfg.duration}{cfg.duration_unit} "
        f"dry_run={cfg.dry_run}\n"
        f"adaptive={cfg.adaptive_mode} "
        f"(MIN_EDGE={cfg.min_edge:+.3%} MIN_CONFIDENCE={cfg.min_confidence:.3f} cooldown={cfg.cooldown_seconds}s)\n"
        f"max_loss={cfg.max_daily_loss} take_profit={cfg.take_profit if cfg.take_profit > 0 else 'off'}"
    )

    backoff_s = 1.0
    try:
        while True:
            try:
                await ws.connect()
                auth = await ws.request({"authorize": cfg.token})
                acct = (auth.get("authorize") or {}).get("loginid", "?")
                print(f"Authorized as {acct}")
                backoff_s = 1.0

                async for tick in ws.ticks(cfg.symbol):
                    ticks_seen += 1
                    d = tick.last_digit
                    last_digits.append(d)

                    # Online update from realized next digit (needs at least 2 samples).
                    ai.update(list(last_digits))

                    # Warm-up: need enough history for stable features + online updates.
                    min_history = 80 if cfg.adaptive_mode else 60
                    if len(last_digits) < min_history:
                        if ticks_seen % 10 == 0:
                            print(f"warmup: ticks={ticks_seen} buffer={len(last_digits)}/{min_history}")
                        continue

                    # Risk gates.
                    if trades >= cfg.max_trades_per_session:
                        print("Session trade limit reached. Exiting.")
                        return
                    if realized_pnl <= -abs(cfg.max_daily_loss):
                        print(f"Max loss reached (pnl={realized_pnl:.2f}). Exiting.")
                        return
                    if cfg.take_profit > 0 and realized_pnl >= cfg.take_profit:
                        print(f"Take profit reached (pnl={realized_pnl:.2f} >= {cfg.take_profit:.2f}). Exiting.")
                        return
                    # Cooldown: fixed in legacy mode, dynamic in adaptive mode (set after decision).
                    if not cfg.adaptive_mode and (now_ts() - last_trade_ts) < cfg.cooldown_seconds:
                        continue

                    p_over, p_under, diag = ai.score(list(last_digits))

                    # Get current payout quotes for both sides.
                    prop_over = await ws.proposal(
                        symbol=cfg.symbol,
                        contract_type=_contract_type_for_action("OVER_5"),
                        amount=cfg.stake,
                        currency=cfg.currency,
                        duration=cfg.duration,
                        duration_unit=cfg.duration_unit,
                        barrier=_barrier_for_action("OVER_5"),
                    )
                    prop_under = await ws.proposal(
                        symbol=cfg.symbol,
                        contract_type=_contract_type_for_action("UNDER_4"),
                        amount=cfg.stake,
                        currency=cfg.currency,
                        duration=cfg.duration,
                        duration_unit=cfg.duration_unit,
                        barrier=_barrier_for_action("UNDER_4"),
                    )

                    po = prop_over["proposal"]
                    pu = prop_under["proposal"]
                    payout_over = float(po.get("payout", 0.0))
                    payout_under = float(pu.get("payout", 0.0))

                    decision = decision_from_scores(
                        p_over=p_over,
                        p_under=p_under,
                        payout_over=payout_over,
                        payout_under=payout_under,
                        stake=cfg.stake,
                        min_edge=cfg.min_edge,
                        min_confidence=cfg.min_confidence,
                        adaptive=cfg.adaptive_mode,
                        n_samples=len(last_digits),
                        diag=diag,
                    )

                    ts = time.strftime("%H:%M:%S")
                    print(f"\n[{ts}] tick last_digit={d} quote={tick.quote}")
                    for r in decision.reasons:
                        print(f"- {r}")

                    if decision.action == "SKIP":
                        continue

                    # Hard safety: never place two trades at once, and never twice on same tick.
                    if trade_lock.locked():
                        print("- skip: trade already in-flight")
                        continue
                    if last_trade_epoch is not None and tick.epoch == last_trade_epoch:
                        print("- skip: already traded on this tick epoch")
                        continue

                    if cfg.adaptive_mode and (now_ts() - last_trade_ts) < decision.cooldown_seconds:
                        print(f"- skip: adaptive cooldown {decision.cooldown_seconds}s not elapsed")
                        continue

                    # Execute (or dry-run).
                    async with trade_lock:
                        action = decision.action
                        prop = po if action == "OVER_5" else pu
                        proposal_id = prop.get("id")
                        ask_price = float(prop.get("ask_price", cfg.stake))
                        if not proposal_id:
                            print("- skip: missing proposal id")
                            continue

                        # Reserve cooldown immediately so we don't double-fire while awaiting network.
                        last_trade_ts = now_ts()
                        last_trade_epoch = tick.epoch

                        if cfg.dry_run:
                            print(
                                f"- DRY_RUN: would BUY {action} for price={ask_price:.2f}, "
                                f"payout≈{float(prop.get('payout', 0.0)):.2f}"
                            )
                            trades += 1
                            continue

                        buy = await ws.buy(str(proposal_id), price=ask_price)
                        buy_info = buy.get("buy") or {}
                        contract_id: Optional[str] = buy_info.get("contract_id")
                        print(
                            f"- BUY placed: contract_id={contract_id} "
                            f"buy_price={float(buy_info.get('buy_price', 0.0)):.2f}"
                        )

                        profit = buy_info.get("profit")
                        if profit is not None:
                            realized_pnl += float(profit)
                            print(f"- pnl update: realized_pnl={realized_pnl:.2f}")

                        trades += 1
                        if cfg.take_profit > 0 and realized_pnl >= cfg.take_profit:
                            print(
                                f"Take profit reached (pnl={realized_pnl:.2f} >= {cfg.take_profit:.2f}). Exiting."
                            )
                            return

            except KeyboardInterrupt:
                print("Stopped by user.")
                return
            except Exception as e:  # noqa: BLE001
                # Any connection hiccup: close + retry with backoff.
                msg = str(e)
                if isinstance(e, DerivAPIError) or "ConnectionClosed" in type(e).__name__ or "close frame" in msg:
                    print(f"Connection error, reconnecting in {backoff_s:.1f}s: {e}")
                    try:
                        await ws.close()
                    except Exception:  # noqa: BLE001
                        pass
                    await asyncio.sleep(backoff_s)
                    backoff_s = min(15.0, backoff_s * 1.7)
                    continue
                raise
    finally:
        await ws.close()


if __name__ == "__main__":
    asyncio.run(main())

