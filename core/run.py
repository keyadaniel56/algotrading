from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from typing import Deque, Optional

from ai_logic import DigitsAI, decision_from_scores
from config import load_config
from deriv_ws import DerivAPIError, DerivWS


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
    if action in ("OVER_5", "OVER_1"):
        return "5" if action == "OVER_5" else "1"
    if action in ("UNDER_4", "UNDER_8"):
        return "4" if action == "UNDER_4" else "8"
    raise ValueError(f"Unknown action: {action}")


def _contract_type_for_action(action: str) -> str:
    # Deriv: DIGITOVER / DIGITUNDER for digits contracts.
    if action.startswith("OVER"):
        return "DIGITOVER"
    if action.startswith("UNDER"):
        return "DIGITUNDER"
    raise ValueError(f"Unknown action: {action}")


async def _settle_contract(
    ws: DerivWS,
    contract_id: str,
    state: dict,
) -> None:
    """Wait for 1-tick contract to settle and update bot state."""
    try:
        for attempt in range(15):  # poll up to 15s
            await asyncio.sleep(1.0)
            try:
                resp = await ws.request({"proposal_open_contract": 1, "contract_id": int(contract_id)})
            except DerivAPIError:
                # Contract may already be closed — try profit_table instead
                break
            poc = resp.get("proposal_open_contract") or {}
            status = poc.get("status", "")
            if status in ("won", "lost") or poc.get("is_sold"):
                profit_val = float(poc.get("profit", 0.0))
                state["realized_pnl"] += profit_val
                won = profit_val > 0
                state["recovery_mode"] = not won
                state["pending"] = False
                mode_str = "RECOVERY (OVER_5/UNDER_4)" if state["recovery_mode"] else "NORMAL (OVER_1/UNDER_8)"
                print(
                    f"- {'WIN' if won else 'LOSS'}: contract_id={contract_id} "
                    f"profit={profit_val:+.2f} pnl={state['realized_pnl']:.2f} -> mode={mode_str}"
                )
                return
        # Fallback: check profit_table for the contract
        try:
            pt = await ws.request({"profit_table": 1, "contract_type": ["DIGITOVER", "DIGITUNDER"],
                                   "limit": 5, "sort": "DESC"})
            for txn in (pt.get("profit_table") or {}).get("transactions", []):
                if str(txn.get("contract_id")) == str(contract_id):
                    profit_val = float(txn.get("profit", 0.0))
                    state["realized_pnl"] += profit_val
                    won = profit_val > 0
                    state["recovery_mode"] = not won
                    state["pending"] = False
                    mode_str = "RECOVERY (OVER_5/UNDER_4)" if state["recovery_mode"] else "NORMAL (OVER_1/UNDER_8)"
                    print(
                        f"- {'WIN' if won else 'LOSS'} (table): contract_id={contract_id} "
                        f"profit={profit_val:+.2f} pnl={state['realized_pnl']:.2f} -> mode={mode_str}"
                    )
                    return
        except Exception:  # noqa: BLE001
            pass
        # Could not determine result — reset pending so bot can continue
        print(f"- settle timeout for {contract_id}, resetting to NORMAL mode")
        state["recovery_mode"] = False
        state["pending"] = False
    except Exception as e:  # noqa: BLE001
        print(f"- settle error for {contract_id}: {e}")
        state["recovery_mode"] = False
        state["pending"] = False


async def main() -> None:
    # Always load the real runtime env file (".env") from the project directory.
    # ".env.example" is only a template and is never loaded.
    project_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    _load_dotenv_if_present(project_env)
    cfg = load_config()

    if not cfg.token:
        print("Missing DERIV_TOKEN. Set it in .env (use a DEMO token first).")
        return

    ws = DerivWS(app_id=cfg.app_id, token=cfg.token)
    ai = DigitsAI()

    last_digits: Deque[int]

    # Keep a generous history and let the model pick its own effective window.
    # (In adaptive mode we don't want to depend on WINDOW_TICKS.)
    max_hist = max(500, cfg.window_ticks) if not cfg.adaptive_mode else 500
    last_digits = deque(maxlen=max(max_hist, 60))

    last_trade_epoch: Optional[int] = None
    trades = 0
    ticks_seen = 0
    trade_lock = asyncio.Lock()
    _state: dict = {
        "realized_pnl": 0.0,
        "recovery_mode": False,
        "pending": False,
        "trades_in_cycle": 0,   # counts trades placed in current analysis cycle
        "pause_until_tick": 0,  # skip trading until ticks_seen reaches this value
    }

    print(
        "Starting Deriv Digits bot\n"
        f"symbol={cfg.symbol} stake={cfg.stake} {cfg.currency} duration={cfg.duration}{cfg.duration_unit} "
        f"dry_run={cfg.dry_run}\n"
        f"adaptive={cfg.adaptive_mode} "
        f"(MIN_EDGE={cfg.min_edge:+.3%} MIN_CONFIDENCE={cfg.min_confidence:.3f})\n"
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

                    # Warm-up: need at least 20 ticks for analysis.
                    min_history = 20
                    if len(last_digits) < min_history:
                        if ticks_seen % 5 == 0:
                            print(f"warmup: ticks={ticks_seen} buffer={len(last_digits)}/{min_history}")
                        continue

                    # Risk gates.
                    if trades >= cfg.max_trades_per_session:
                        print("Session trade limit reached. Exiting.")
                        return
                    if _state["realized_pnl"] <= -abs(cfg.max_daily_loss):
                        print(f"Max loss reached (pnl={_state['realized_pnl']:.2f}). Exiting.")
                        return
                    if cfg.take_profit > 0 and _state["realized_pnl"] >= cfg.take_profit:
                        print(f"Take profit reached (pnl={_state['realized_pnl']:.2f} >= {cfg.take_profit:.2f}). Exiting.")
                        return

                    recovery_mode = _state["recovery_mode"]

                    # After every 2-trade cycle, pause for 5 ticks before re-analyzing.
                    if ticks_seen <= _state["pause_until_tick"]:
                        remaining = _state["pause_until_tick"] - ticks_seen
                        if remaining % 2 == 0:
                            print(f"- re-analyzing in {remaining} ticks...")
                        continue

                    p_over, p_under, diag = ai.score(list(last_digits), recovery=recovery_mode)

                    # Primary: OVER_1/UNDER_8. Recovery (after a loss): OVER_5/UNDER_4.
                    over_action = "OVER_5" if recovery_mode else "OVER_1"
                    under_action = "UNDER_4" if recovery_mode else "UNDER_8"

                    # Get current payout quotes for both sides.
                    prop_over = await ws.proposal(
                        symbol=cfg.symbol,
                        contract_type=_contract_type_for_action(over_action),
                        amount=cfg.stake,
                        currency=cfg.currency,
                        duration=cfg.duration,
                        duration_unit=cfg.duration_unit,
                        barrier=_barrier_for_action(over_action),
                    )
                    prop_under = await ws.proposal(
                        symbol=cfg.symbol,
                        contract_type=_contract_type_for_action(under_action),
                        amount=cfg.stake,
                        currency=cfg.currency,
                        duration=cfg.duration,
                        duration_unit=cfg.duration_unit,
                        barrier=_barrier_for_action(under_action),
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
                        is_recovery=recovery_mode,
                    )

                    ts = time.strftime("%H:%M:%S")
                    print(f"\n[{ts}] tick last_digit={d} quote={tick.quote}")
                    for r in decision.reasons:
                        print(f"- {r}")

                    if decision.action == "SKIP":
                        continue

                    # Block new trades until previous contract settles.
                    if _state["pending"]:
                        print("- skip: waiting for previous contract to settle")
                        continue

                    # Hard safety: never place two trades at once, and never twice on same tick.
                    if trade_lock.locked():
                        print("- skip: trade already in-flight")
                        continue
                    if last_trade_epoch is not None and tick.epoch == last_trade_epoch:
                        print("- skip: already traded on this tick epoch")
                        continue

                    # Execute (or dry-run).
                    async with trade_lock:
                        action = decision.action
                        prop = po if action.startswith("OVER") else pu
                        proposal_id = prop.get("id")
                        ask_price = float(prop.get("ask_price", cfg.stake))
                        if not proposal_id:
                            print("- skip: missing proposal id")
                            continue

                        last_trade_epoch = tick.epoch

                        if cfg.dry_run:
                            print(
                                f"- DRY_RUN: would BUY {action} for price={ask_price:.2f}, "
                                f"payout≈{float(prop.get('payout', 0.0)):.2f} "
                                f"[{'RECOVERY' if recovery_mode else 'NORMAL'}]"
                            )
                            trades += 1
                            _state["trades_in_cycle"] += 1
                            if _state["trades_in_cycle"] >= 2:
                                _state["trades_in_cycle"] = 0
                                _state["pause_until_tick"] = ticks_seen + 5
                                print("- cycle complete: pausing 5 ticks to re-analyze market")
                            continue

                        buy = await ws.buy(str(proposal_id), price=ask_price)
                        buy_info = buy.get("buy") or {}
                        contract_id: Optional[str] = str(buy_info.get("contract_id", ""))
                        buy_price = float(buy_info.get("buy_price", 0.0))
                        print(
                            f"- BUY placed: contract_id={contract_id} "
                            f"buy_price={buy_price:.2f} "
                            f"[{'RECOVERY' if recovery_mode else 'NORMAL'}]"
                        )
                        trades += 1
                        _state["pending"] = True
                        _state["trades_in_cycle"] += 1
                        if _state["trades_in_cycle"] >= 2:
                            _state["trades_in_cycle"] = 0
                            _state["pause_until_tick"] = ticks_seen + 5
                            print("- cycle complete: pausing 5 ticks to re-analyze market")
                        else:
                            print("- cycle complete: re-analyzing market")

                        # Fetch settlement result to determine win/loss and update mode.
                        if contract_id:
                            asyncio.create_task(
                                _settle_contract(ws, contract_id, _state)
                            )

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

