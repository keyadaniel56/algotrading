from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class Decision:
    action: str  # "OVER_1" | "UNDER_8" | "OVER_5" | "UNDER_4" | "SKIP"
    p_win: float
    confidence: float
    edge: float
    reasons: list[str]


class OnlineLogReg:
    def __init__(self, n_features: int, *, lr: float = 0.15, l2: float = 1e-4):
        self.w = [0.0] * n_features
        self.b = 0.0
        self.lr = lr
        self.l2 = l2

    def predict_proba(self, x: list[float]) -> float:
        z = self.b
        for wi, xi in zip(self.w, x, strict=False):
            z += wi * xi
        return sigmoid(z)

    def update(self, x: list[float], y: int) -> None:
        p = self.predict_proba(x)
        g = p - float(y)
        m = min(len(self.w), len(x))
        for i in range(m):
            self.w[i] -= self.lr * (g * x[i] + self.l2 * self.w[i])
        self.b -= self.lr * g


def _softmax(z: list[float]) -> list[float]:
    if not z:
        return []
    m = max(z)
    exps = [math.exp(v - m) for v in z]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(z)] * len(z)
    return [e / s for e in exps]


class OnlineSoftmaxReg:
    def __init__(self, n_features: int, n_classes: int = 10, *, lr: float = 0.08, l2: float = 5e-5):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.l2 = l2
        self.W = [[0.0] * n_features for _ in range(n_classes)]
        self.b = [0.0] * n_classes

    def predict_proba(self, x: list[float]) -> list[float]:
        m = min(self.n_features, len(x))
        logits = [self.b[c] for c in range(self.n_classes)]
        for c in range(self.n_classes):
            wc = self.W[c]
            zc = logits[c]
            for i in range(m):
                zc += wc[i] * x[i]
            logits[c] = zc
        return _softmax(logits)

    def update(self, x: list[float], y_class: int) -> None:
        if y_class < 0 or y_class >= self.n_classes:
            return
        p = self.predict_proba(x)
        m = min(self.n_features, len(x))
        for c in range(self.n_classes):
            g = p[c] - (1.0 if c == y_class else 0.0)
            wc = self.W[c]
            for i in range(m):
                wc[i] -= self.lr * (g * x[i] + self.l2 * wc[i])
            self.b[c] -= self.lr * g


def _digit_one_hot(d: int) -> list[float]:
    v = [0.0] * 10
    v[d] = 1.0
    return v


def _adaptive_feature_window_len(n: int) -> int:
    if n <= 0:
        return 20
    return max(20, min(120, int(n * 0.33)))


def build_features(last_digits: list[int]) -> tuple[list[float], dict[str, float]]:
    data = list(last_digits) if last_digits else [0]
    d0 = data[-1]
    wlen = _adaptive_feature_window_len(len(data))
    window = data[max(0, len(data) - wlen):]

    over5_rate = sum(1 for d in window if d >= 6) / len(window)
    under4_rate = sum(1 for d in window if d <= 3) / len(window)
    even_rate = sum(1 for d in window if d % 2 == 0) / len(window)

    streak = 1
    for i in range(len(data) - 2, -1, -1):
        if data[i] == d0:
            streak += 1
        else:
            break
    streak = min(streak, 10)

    d1 = data[-2] if len(data) >= 2 else d0
    d2 = data[-3] if len(data) >= 3 else d1

    x: list[float] = []
    x.extend(_digit_one_hot(d0))
    x.extend(_digit_one_hot(d1))
    x.extend(_digit_one_hot(d2))
    x.append(over5_rate)
    x.append(under4_rate)
    x.append(even_rate)
    x.append(streak / 10.0)
    x.append(1.0)

    diag = {
        "last_digit": float(d0),
        "prev_digit": float(d1),
        "prev2_digit": float(d2),
        "feature_window_len": float(wlen),
        "over5_rate_w": over5_rate,
        "under4_rate_w": under4_rate,
        "even_rate_w": even_rate,
        "same_digit_streak": float(streak),
    }
    return x, diag


class DigitsAI:
    def __init__(self):
        self._n = len(build_features([0])[0])
        self._m_digit = OnlineSoftmaxReg(self._n, 10)
        self._m_over = OnlineLogReg(self._n)
        self._m_under = OnlineLogReg(self._n)

    def update(self, last_digits: list[int]) -> None:
        if len(last_digits) < 2:
            return
        x, _ = build_features(last_digits[:-1])
        d = last_digits[-1]
        self._m_digit.update(x, int(d))
        # Balanced binary labels (~40% positive each) to avoid base-rate bias.
        self._m_over.update(x, 1 if d >= 6 else 0)
        self._m_under.update(x, 1 if d <= 3 else 0)

    def score(self, last_digits: list[int], recovery: bool = False) -> tuple[float, float, dict[str, float]]:
        x, diag = build_features(last_digits)
        p_digits = self._m_digit.predict_proba(x)

        if recovery:
            # OVER_5: digits 6-9 (~40% base), UNDER_4: digits 0-3 (~40% base)
            p_over_raw = sum(p_digits[6:10])
            p_under_raw = sum(p_digits[0:4])
            base_over, base_under = 0.4, 0.4
        else:
            # OVER_1: digits 2-9 (~80% base), UNDER_8: digits 0-7 (~80% base)
            # Use true base rates so rescaling produces meaningful confidence
            p_over_raw = sum(p_digits[2:10])
            p_under_raw = sum(p_digits[0:8])
            base_over, base_under = 0.8, 0.8

        def _rescale(p: float, base: float) -> float:
            if p >= base:
                return 0.5 + 0.5 * (p - base) / max(1e-9, 1.0 - base)
            return 0.5 * p / max(1e-9, base)

        p_over = _rescale(p_over_raw, base_over)
        p_under = _rescale(p_under_raw, base_under)

        # Blend with binary models for early stability.
        p_over_bin = self._m_over.predict_proba(x)
        p_under_bin = self._m_under.predict_proba(x)
        alpha = 0.65
        p_over = alpha * p_over + (1.0 - alpha) * p_over_bin
        p_under = alpha * p_under + (1.0 - alpha) * p_under_bin

        top = sorted(range(10), key=lambda i: p_digits[i], reverse=True)[:3]
        diag["p_digit_top1"] = float(top[0])
        diag["p_digit_top1_p"] = float(p_digits[top[0]])
        diag["p_digit_top2"] = float(top[1])
        diag["p_digit_top2_p"] = float(p_digits[top[1]])
        diag["p_digit_top3"] = float(top[2])
        diag["p_digit_top3_p"] = float(p_digits[top[2]])
        diag["recovery_mode"] = float(recovery)
        return p_over, p_under, diag


def expected_edge(p_win: float, payout: float, stake: float) -> float:
    if stake <= 0:
        return 0.0
    ev = p_win * (payout - stake) - (1.0 - p_win) * stake
    return ev / stake


def decision_from_scores(
    *,
    p_over: float,
    p_under: float,
    payout_over: float,
    payout_under: float,
    stake: float,
    min_edge: float,
    min_confidence: float,
    adaptive: bool,
    n_samples: int,
    diag: dict[str, float],
    is_recovery: bool = False,
) -> Decision:
    conf_over = abs(p_over - 0.5) * 2.0
    conf_under = abs(p_under - 0.5) * 2.0

    edge_over = expected_edge(p_over, payout_over, stake)
    edge_under = expected_edge(p_under, payout_under, stake)

    reasons: List[str] = []
    reasons.append(
        f"p_over={p_over:.3f} (conf={conf_over:.3f}, edge={edge_over:+.3%}), "
        f"p_under={p_under:.3f} (conf={conf_under:.3f}, edge={edge_under:+.3%})"
    )
    reasons.append(
        f"signals: last_digit={int(diag.get('last_digit', -1))}, "
        f"window={int(diag.get('feature_window_len', 0))}, "
        f"over5_rate={diag.get('over5_rate_w', 0.0):.2f}, "
        f"under4_rate={diag.get('under4_rate_w', 0.0):.2f}, "
        f"streak={int(diag.get('same_digit_streak', 0))}, "
        f"n={n_samples}"
    )
    if "p_digit_top1_p" in diag:
        reasons.append(
            "digit_pred: "
            f"{int(diag.get('p_digit_top1', 0))}:{diag.get('p_digit_top1_p', 0.0):.2f} "
            f"{int(diag.get('p_digit_top2', 0))}:{diag.get('p_digit_top2_p', 0.0):.2f} "
            f"{int(diag.get('p_digit_top3', 0))}:{diag.get('p_digit_top3_p', 0.0):.2f}"
        )

    best = ("OVER_5" if edge_over >= edge_under else "UNDER_4") if is_recovery \
        else ("OVER_1" if edge_over >= edge_under else "UNDER_8")

    if best.startswith("OVER"):
        p_win, conf, edge = p_over, conf_over, edge_over
    else:
        p_win, conf, edge = p_under, conf_under, edge_under

    if adaptive:
        if conf < min_confidence:
            reasons.append(f"skip: confidence {conf:.3f} < min_conf {min_confidence:.3f}")
            return Decision(action="SKIP", p_win=p_win, confidence=conf, edge=edge, reasons=reasons)
        # Require p_win to be meaningfully above 0.5 (model must favor this side).
        if p_win < 0.52:
            reasons.append(f"skip: p_win {p_win:.3f} not sufficiently above 0.5")
            return Decision(action="SKIP", p_win=p_win, confidence=conf, edge=edge, reasons=reasons)
        reasons.append(f"trade: {best} (conf={conf:.3f}, p_win={p_win:.3f})")
        return Decision(action=best, p_win=p_win, confidence=conf, edge=edge, reasons=reasons)

    # Legacy fixed-threshold policy.
    if conf < min_confidence:
        reasons.append(f"skip: confidence {conf:.3f} < MIN_CONFIDENCE {min_confidence:.3f}")
        return Decision(action="SKIP", p_win=p_win, confidence=conf, edge=edge, reasons=reasons)
    if p_win < 0.52:
        reasons.append(f"skip: p_win {p_win:.3f} not sufficiently above 0.5")
        return Decision(action="SKIP", p_win=p_win, confidence=conf, edge=edge, reasons=reasons)
    reasons.append(f"trade: {best} (conf {conf:.3f}, p_win={p_win:.3f})")
    return Decision(action=best, p_win=p_win, confidence=conf, edge=edge, reasons=reasons)
