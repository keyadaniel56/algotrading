from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from typing import Deque, Optional

from ai_logic import DigitsAI
from config import load_config
from deriv_ws import DerivAPIError, DerivWS

SESSION_MAX_TRADES = 3  # hard cap: 3 trades per session
MIN_CONFIDENCE = 0.008   # min gap below uniform (0.1) to trade


def _load_dotenv_if_present(path: str = ".env") -> None:
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


def _pick_differ_digit(last_digits: list[int], ai: DigitsAI) -> tuple[int, float]:
    """
    Pick the DIGITDIFF barrier using a blend of:
    - AI model probability (60%) — predicted likelihood of each digit
    - Recent 20-tick frequency (40%) — hot digits are riskier barriers

    Lowest blended score = safest barrier (least likely to appear).
    """
    p_digits = ai.digit_probs(last_digits)
    recent = last_digits[-20:] if len(last_digits) >= 20 else last_digits
    freq = [recent.count(i) / len(recent) for i in range(10)]
    blended = [0.6 * p_digits[i] + 0.4 * freq[i] for i in range(10)]

    best = min(range(10), key=lambda i: blended[i])
    confidence = 0.1 - p_digits[best]

    scores_str = " ".join(
        f"{i}:{blended[i]:.3f}(p={p_digits[i]:.3f},f={freq[i]:.2f})"
        for i in range(10)
    )
    print(f"- digit scores: {scores_str}")
    print(f"- selected barrier={best} (model_p={p_digits[best]:.3f}, recent_freq={freq[best]:.2f})")
    return best, confidence


async def _reconnect(ws: DerivWS, cfg) -> None:
    """Close and reconnect + re-authorize."""
    try:
        await ws.close()
    except Exception:
        pass
    backoff = 1.0
    while True:
        try:
            await ws.connect()
            await ws.request({"authorize": cfg.token})
            print("- reconnected.")
            return
        except Exception as e:
            print(f"- reconnect failed ({e}), retrying in {backoff:.1f}s...")
            await asyncio.sleep(backoff)
            backoff = min(15.0, backoff * 2)


async def collect_ticks(
    ws: DerivWS, cfg, n: int, label: str,
    last_digits: Deque[int], ai: DigitsAI
) -> None:
    """
    Collect exactly n ticks, reconnecting transparently on any error.
    Opens a fresh subscription each call and ensures it's closed when done.
    """
    print(f"{label} ({n} ticks)...")
    collected = 0
    while collected < n:
        gen = ws.ticks(cfg.symbol)
        try:
            async for tick in gen:
                d = tick.last_digit
                last_digits.append(d)
                ai.update(list(last_digits))
                collected += 1
                if collected >= n:
                    break
        except (DerivAPIError, Exception) as e:
            print(f"- tick stream error ({e}), reconnecting... ({collected}/{n} collected)")
            await _reconnect(ws, cfg)
        finally:
            # Always close the generator so the symbol is unregistered
            await gen.aclose()


async def _get_differ_proposal(ws: DerivWS, cfg, digit: int) -> dict:
    return await ws.request({
        "proposal": 1,
        "symbol": cfg.symbol,
        "contract_type": "DIGITDIFF",
        "amount": cfg.stake,
        "basis": "stake",
        "currency": cfg.currency,
        "duration": cfg.duration,
        "duration_unit": cfg.duration_unit,
        "barrier": str(digit),
    })


async def _settle_and_return(ws: DerivWS, cfg, contract_id: str) -> tuple[bool, float]:
    """Poll until contract settles. Returns (won, profit)."""
    for _ in range(20):
        await asyncio.sleep(1.0)
        try:
            resp = await ws.request({"proposal_open_contract": 1, "contract_id": int(contract_id)})
        except DerivAPIError as e:
            print(f"- settle poll error: {e}, reconnecting...")
            await _reconnect(ws, cfg)
            continue
        poc = resp.get("proposal_open_contract") or {}
        status = poc.get("status", "")
        if status in ("won", "lost") or poc.get("is_sold"):
            profit = float(poc.get("profit", 0.0))
            return profit > 0, profit

    # Fallback: profit_table
    try:
        pt = await ws.request({
            "profit_table": 1,
            "contract_type": ["DIGITDIFF"],
            "limit": 5,
            "sort": "DESC",
        })
        for txn in (pt.get("profit_table") or {}).get("transactions", []):
            if str(txn.get("contract_id")) == str(contract_id):
                profit = float(txn.get("profit", 0.0))
                return profit > 0, profit
    except Exception:
        pass

    print("- settle timeout, assuming loss for safety")
    return False, 0.0


async def main() -> None:
    project_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    _load_dotenv_if_present(project_env)
    cfg = load_config()

    if not cfg.token:
        print("Missing DERIV_TOKEN. Set it in .env")
        return

    ws = DerivWS(app_id=cfg.app_id, token=cfg.token)
    ai = DigitsAI()
    last_digits: Deque[int] = deque(maxlen=500)

    session_trades = 0
    session_pnl = 0.0

    print(
        f"Digit Differs Bot | symbol={cfg.symbol} stake={cfg.stake} {cfg.currency} "
        f"dry_run={cfg.dry_run} | max {SESSION_MAX_TRADES} trades/session, stop on loss"
    )

    try:
        await ws.connect()
        auth = await ws.request({"authorize": cfg.token})
        acct = (auth.get("authorize") or {}).get("loginid", "?")
        print(f"Authorized as {acct}\n")

        # Warm-up: 100 ticks for the model to build signal
        await collect_ticks(ws, cfg, 100, "Warming up", last_digits, ai)
        print(f"Warm-up done. Buffer={len(last_digits)} ticks.\n")

        while session_trades < SESSION_MAX_TRADES:
            trade_num = session_trades + 1
            print(f"--- Trade {trade_num}/{SESSION_MAX_TRADES} ---")

            # Fresh re-analysis before each trade
            await collect_ticks(ws, cfg, 5, "Re-analyzing market", last_digits, ai)

            digit, confidence = _pick_differ_digit(list(last_digits), ai)
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] DIGITDIFF barrier={digit} confidence={confidence:.4f}")

            # Wait for a clear signal if model is still near-uniform
            while confidence < MIN_CONFIDENCE:
                print(f"- low confidence ({confidence:.4f}), collecting 5 more ticks...")
                await collect_ticks(ws, cfg, 5, "Extra analysis", last_digits, ai)
                digit, confidence = _pick_differ_digit(list(last_digits), ai)
                print(f"- updated: barrier={digit} confidence={confidence:.4f}")

            if cfg.dry_run:
                print(f"- DRY_RUN: would BUY DIGITDIFF barrier={digit} stake={cfg.stake:.2f}")
                session_trades += 1
                print(f"- DRY_RUN: simulated WIN | session_trades={session_trades}\n")
                continue

            # Get proposal
            try:
                prop_resp = await _get_differ_proposal(ws, cfg, digit)
            except DerivAPIError as e:
                print(f"- proposal error: {e}, skipping")
                continue

            prop = prop_resp.get("proposal") or {}
            proposal_id = prop.get("id")
            ask_price = float(prop.get("ask_price", cfg.stake))
            payout = float(prop.get("payout", 0.0))

            if not proposal_id:
                print("- no proposal id, skipping")
                continue

            print(f"- proposal: id={proposal_id} ask={ask_price:.2f} payout≈{payout:.2f}")

            # Place trade
            try:
                buy_resp = await ws.buy(str(proposal_id), price=ask_price)
            except DerivAPIError as e:
                print(f"- buy error: {e}")
                continue

            buy_info = buy_resp.get("buy") or {}
            contract_id = str(buy_info.get("contract_id", ""))
            buy_price = float(buy_info.get("buy_price", 0.0))
            print(f"- BUY placed: contract_id={contract_id} buy_price={buy_price:.2f}")

            session_trades += 1

            print("- waiting for settlement...")
            won, profit = await _settle_and_return(ws, cfg, contract_id)
            session_pnl += profit

            print(f"- {'WIN' if won else 'LOSS'}: profit={profit:+.2f} | session_pnl={session_pnl:+.2f}\n")

            if not won:
                print("Loss detected — stopping session immediately.")
                break

        print(f"Session complete: {session_trades} trade(s) | pnl={session_pnl:+.2f} {cfg.currency}")

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        await ws.close()


if __name__ == "__main__":
    asyncio.run(main())
