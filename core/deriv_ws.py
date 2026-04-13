from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

import websockets


DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3"


class DerivAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tick:
    symbol: str
    epoch: int
    quote: float

    @property
    def last_digit(self) -> int:
        # Deriv digits contracts use the last digit of the quote.
        s = f"{self.quote:.5f}".rstrip("0").rstrip(".")
        last_char = s[-1]
        if last_char == ".":
            return 0
        return int(last_char)


class DerivWS:
    def __init__(self, app_id: str, token: Optional[str], *, ping_interval: float = 20.0):
        self._app_id = app_id
        self._token = token
        self._ping_interval = ping_interval
        # websockets v12+ returns a ClientConnection object (no .closed attr),
        # older versions returned WebSocketClientProtocol (.closed attr).
        self._ws: Optional[Any] = None
        self._req_id = 0
        self._pending: Dict[int, asyncio.Future[dict]] = {}
        self._tick_queues: Dict[str, asyncio.Queue[Tick]] = {}
        self._reader_task: Optional[asyncio.Task[None]] = None

    @property
    def connected(self) -> bool:
        ws = self._ws
        if ws is None:
            return False
        # websockets<=11: .closed (bool)
        closed = getattr(ws, "closed", None)
        if isinstance(closed, bool):
            return not closed
        # websockets v12+: .state (enum), and/or .close_code (None while open)
        state = getattr(ws, "state", None)
        if state is not None:
            try:
                from websockets.protocol import State  # type: ignore

                return state == State.OPEN
            except Exception:  # noqa: BLE001
                # If State import fails, fall back to close_code heuristic.
                pass
        close_code = getattr(ws, "close_code", None)
        if close_code is None:
            return True
        # Some implementations use .open (bool).
        open_attr = getattr(ws, "open", None)
        if isinstance(open_attr, bool):
            return open_attr
        return False

    async def connect(self) -> None:
        if self.connected:
            return
        url = f"{DERIV_WS_URL}?app_id={self._app_id}"
        self._ws = await websockets.connect(url, ping_interval=self._ping_interval)
        self._reader_task = asyncio.create_task(self._reader_loop())
        if self._token:
            await self.authorize(self._token)

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
        self._ws = None
        if self._reader_task is not None:
            self._reader_task.cancel()
        self._reader_task = None
        self._pending.clear()
        self._tick_queues.clear()

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                data = json.loads(msg)
                req_id = data.get("req_id")
                if req_id is not None and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        fut.set_result(data)
                    continue

                # Streamed ticks (subscribe: 1) don't carry req_id reliably.
                if data.get("msg_type") == "tick":
                    tick = data.get("tick") or {}
                    try:
                        t = Tick(symbol=tick["symbol"], epoch=int(tick["epoch"]), quote=float(tick["quote"]))
                    except Exception:  # noqa: BLE001
                        continue
                    q = self._tick_queues.get(t.symbol)
                    if q is not None:
                        # Avoid blocking reader loop; drop if queue is full.
                        try:
                            q.put_nowait(t)
                        except asyncio.QueueFull:
                            pass
        except Exception as e:  # noqa: BLE001
            # Unblock any waiters.
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(e)
            self._pending.clear()
            for q in self._tick_queues.values():
                try:
                    q.put_nowait(Tick(symbol="__error__", epoch=int(now_ts()), quote=float("nan")))
                except Exception:  # noqa: BLE001
                    pass

    async def request(self, payload: dict) -> dict:
        if not self.connected:
            await self.connect()
        assert self._ws is not None
        self._req_id += 1
        req_id = self._req_id
        payload = dict(payload)
        payload["req_id"] = req_id
        fut: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        await self._ws.send(json.dumps(payload))
        resp = await fut
        if resp.get("error"):
            msg = resp["error"].get("message", "Deriv API error")
            code = resp["error"].get("code")
            details = resp["error"].get("details")
            raise DerivAPIError(
                f"{msg} (code={code}, details={details}, request_keys={sorted(payload.keys())})"
            )
        return resp

    async def authorize(self, token: str) -> dict:
        return await self.request({"authorize": token})

    async def proposal(
        self,
        *,
        symbol: str,
        contract_type: str,
        amount: float,
        currency: str,
        duration: int,
        duration_unit: str,
        barrier: str,
    ) -> dict:
        return await self.request(
            {
                "proposal": 1,
                "symbol": symbol,
                "contract_type": contract_type,
                "amount": amount,
                "basis": "stake",
                "currency": currency,
                "duration": duration,
                "duration_unit": duration_unit,
                "barrier": barrier,
            }
        )

    async def buy(self, proposal_id: str, price: float) -> dict:
        return await self.request({"buy": proposal_id, "price": price})

    async def ticks(self, symbol: str) -> AsyncIterator[Tick]:
        """
        Subscribe to ticks and yield Tick objects.
        """
        if not self.connected:
            await self.connect()
        assert self._ws is not None
        if symbol in self._tick_queues:
            # Only one consumer per symbol.
            raise RuntimeError(f"Already subscribed to ticks for {symbol}")

        q: asyncio.Queue[Tick] = asyncio.Queue(maxsize=200)
        self._tick_queues[symbol] = q
        try:
            await self._ws.send(json.dumps({"ticks": symbol, "subscribe": 1}))
            while True:
                t = await q.get()
                # Sentinel from reader failure.
                if t.symbol == "__error__":
                    raise DerivAPIError("Websocket reader loop stopped")
                yield t
        finally:
            self._tick_queues.pop(symbol, None)


def now_ts() -> float:
    return time.time()

