from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class Decision:
    action: str  # "OVER_5" | "UNDER_4" | "SKIP"
    p_win: float
    confidence: float
    edge: float
    cooldown_seconds: int
    reasons: List[str]


class OnlineLogReg:
    """
    Tiny online logistic regression with SGD.
    We keep this dependency-free (no numpy/sklearn).
    """

    def __init__(self, n_features: int, *, lr: float = 0.15, l2: float = 1e-4):
        self.w = [0.0] * n_features
        self.b = 0.0
        self.lr = lr
        self.l2 = l2

    def predict_proba(self, x: List[float]) -> float:
        z = self.b
        for wi, xi in zip(self.w, x, strict=False):
            z += wi * xi
        return sigmoid(z)

    def update(self, x: List[float], y: int) -> None:
        p = self.predict_proba(x)
        g = (p - float(y))
        # L2 on weights
        # Be robust to feature vector changes.
        m = min(len(self.w), len(x))
        for i in range(m):
            self.w[i] -= self.lr * (g * x[i] + self.l2 * self.w[i])
        self.b -= self.lr * g


def _softmax(z: List[float]) -> List[float]:
    if not z:
        return []
    m = max(z)
    exps = [math.exp(v - m) for v in z]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(z)] * len(z)
    return [e / s for e in exps]


class OnlineSoftmaxReg:
    """
    Tiny online multinomial logistic regression (softmax) with SGD.
    Dependency-free (no numpy).
    """

    def __init__(self, n_features: int, n_classes: int = 10, *, lr: float = 0.08, l2: float = 5e-5):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.l2 = l2
        # W[c][i]
        self.W = [[0.0] * n_features for _ in range(n_classes)]
        self.b = [0.0] * n_classes

    def predict_proba(self, x: List[float]) -> List[float]:
        m = min(self.n_features, len(x))
        logits = [self.b[c] for c in range(self.n_classes)]
        for c in range(self.n_classes):
            wc = self.W[c]
            zc = logits[c]
            for i in range(m):
                zc += wc[i] * x[i]
            logits[c] = zc
        return _softmax(logits)

    def update(self, x: List[float], y_class: int) -> None:
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


def _digit_one_hot(d: int) -> List[float]:
    v = [0.0] * 10
    v[d] = 1.0
    return v


def _adaptive_feature_window_len(n: int) -> int:
    """
    Pick a window length based on how much history we have.
    This avoids hard-coding a single window size via config.
    """
    if n <= 0:
        return 30
    # Use ~1/3 of available history, clamped.
    return max(30, min(120, int(n * 0.33)))


def build_features(last_digits: List[int]) -> Tuple[List[float], Dict[str, float]]:
    """
    Feature vector + a small named diagnostics dict for reasoning.
    """
    if not last_digits:
        last_digits = [0]
    d0 = last_digits[-1]
    wlen = _adaptive_feature_window_len(len(last_digits))
    window = last_digits[-wlen:]
    over5_rate = sum(1 for d in window if d >= 6) / len(window)
    under4_rate = sum(1 for d in window if d <= 3) / len(window)
    even_rate = sum(1 for d in window if (d % 2 == 0)) / len(window)

    # Simple “streak” feature: last run length of the current digit.
    streak = 1
    for i in range(len(last_digits) - 2, -1, -1):
        if last_digits[i] == d0:
            streak += 1
        else:
            break
    streak = min(streak, 10)

    # Sequence features: last 3 digits (captures short dependencies if any).
    d1 = last_digits[-2] if len(last_digits) >= 2 else d0
    d2 = last_digits[-3] if len(last_digits) >= 3 else d1

    x: List[float] = []
    x.extend(_digit_one_hot(d0))
    x.extend(_digit_one_hot(d1))
    x.extend(_digit_one_hot(d2))
    x.append(over5_rate)
    x.append(under4_rate)
    x.append(even_rate)
    x.append(streak / 10.0)
    x.append(1.0)  # bias-like constant feature

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
    """
    Hybrid:
    - a 10-class next-digit model (softmax) to capture sequence patterns
    - plus the legacy binary models as a fallback/regularizer
    """

    def __init__(self):
        # Infer feature size from builder to avoid mismatch bugs.
        self._n = len(build_features([0])[0])
        self._m_digit = OnlineSoftmaxReg(self._n, 10)
        self._m_over = OnlineLogReg(self._n)
        self._m_under = OnlineLogReg(self._n)

    def update(self, last_digits: List[int]) -> None:
        if len(last_digits) < 2:
            return
        x, _ = build_features(last_digits[:-1])
        d = last_digits[-1]
        self._m_digit.update(x, int(d))
        y_over = 1 if d >= 6 else 0
        y_under = 1 if d <= 3 else 0
        self._m_over.update(x, y_over)
        self._m_under.update(x, y_under)

    def score(self, last_digits: List[int]) -> Tuple[float, float, Dict[str, float]]:
        x, diag = build_features(last_digits)
        p_digits = self._m_digit.predict_proba(x)
        p_over = sum(p_digits[6:10])
        p_under = sum(p_digits[0:4])
        # Blend with legacy binaries for a bit of stability early on.
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
        return p_over, p_under, diag


def expected_edge(p_win: float, payout: float, stake: float) -> float:
    """
    Edge estimate per unit stake:
    EV = p_win*(payout - stake) + (1-p_win)*(-stake)
    edge = EV / stake
    """
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
    diag: Dict[str, float],
) -> Decision:
    # Convert to “confidence” as distance from 0.5.
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

    best = "OVER_5" if edge_over >= edge_under else "UNDER_4"
    if best == "OVER_5":
        p_win, conf, edge = p_over, conf_over, edge_over
    else:
        p_win, conf, edge = p_under, conf_under, edge_under

    # Adaptive policy: require statistical separation from break-even probability.
    #
    # For a 1-tick digits contract bought at ask_price=stake with payout quoted,
    # break-even probability is p_break = stake / payout (if payout > 0).
    # We then require p_win to exceed p_break by a margin that shrinks as we get more data.
    cooldown_seconds = 6
    if adaptive:
        payout = payout_over if best == "OVER_5" else payout_under
        p_break = (stake / payout) if payout and payout > 0 and stake > 0 else 1.0
        # Approximate standard error of p_hat (binomial), with a small floor to avoid zero.
        n_eff = max(30, min(int(n_samples), 400))
        se = math.sqrt(max(1e-9, p_win * (1.0 - p_win) / float(n_eff)))
        # Conservative margin: about 1.65σ (~90% one-sided) + small fixed cushion.
        margin = 1.65 * se + 0.005

        # Convert old config knobs into soft floors (still honored if user sets adaptive=0).
        required_edge = max(0.0, min_edge * 0.35)
        required_conf = max(0.0, min_confidence * 0.35)

        # Translate p-break separation into an "edge-like" requirement.
        sep = p_win - p_break
        reasons.append(
            f"adaptive: p_break={p_break:.3f}, sep={sep:+.3f}, margin={margin:.3f}, "
            f"soft_min_edge={required_edge:+.3%}, soft_min_conf={required_conf:.3f}"
        )

        # Cooldown adapts with signal strength: higher edge/conf => shorter cooldown.
        strength = max(0.0, min(1.0, (edge * 8.0) + (conf - 0.5)))
        cooldown_seconds = int(round(10 - 8 * strength))
        cooldown_seconds = max(1, min(12, cooldown_seconds))

        if sep < margin:
            reasons.append("skip: insufficient separation from break-even (adaptive gate)")
            return Decision(
                action="SKIP",
                p_win=p_win,
                confidence=conf,
                edge=edge,
                cooldown_seconds=cooldown_seconds,
                reasons=reasons,
            )
        if edge < required_edge:
            reasons.append("skip: edge below soft floor (adaptive gate)")
            return Decision(
                action="SKIP",
                p_win=p_win,
                confidence=conf,
                edge=edge,
                cooldown_seconds=cooldown_seconds,
                reasons=reasons,
            )
        if conf < required_conf:
            reasons.append("skip: confidence below soft floor (adaptive gate)")
            return Decision(
                action="SKIP",
                p_win=p_win,
                confidence=conf,
                edge=edge,
                cooldown_seconds=cooldown_seconds,
                reasons=reasons,
            )

        reasons.append(f"trade: {best} (adaptive ok, cooldown={cooldown_seconds}s)")
        return Decision(
            action=best,
            p_win=p_win,
            confidence=conf,
            edge=edge,
            cooldown_seconds=cooldown_seconds,
            reasons=reasons,
        )

    # Legacy fixed-threshold policy.
    if edge < min_edge:
        reasons.append(f"skip: edge {edge:+.3%} < MIN_EDGE {min_edge:+.3%}")
        return Decision(
            action="SKIP",
            p_win=p_win,
            confidence=conf,
            edge=edge,
            cooldown_seconds=cooldown_seconds,
            reasons=reasons,
        )
    if conf < min_confidence:
        reasons.append(f"skip: confidence {conf:.3f} < MIN_CONFIDENCE {min_confidence:.3f}")
        return Decision(
            action="SKIP",
            p_win=p_win,
            confidence=conf,
            edge=edge,
            cooldown_seconds=cooldown_seconds,
            reasons=reasons,
        )

    reasons.append(f"trade: {best} (edge {edge:+.3%}, conf {conf:.3f}, cooldown={cooldown_seconds}s)")
    return Decision(
        action=best,
        p_win=p_win,
        confidence=conf,
        edge=edge,
        cooldown_seconds=cooldown_seconds,
        reasons=reasons,
    )

