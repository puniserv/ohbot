import numpy as np

LIP_CLOSED = 5.0
LIP_SPAN = 2.5  # motor units each lip travels at full opening
# speech RMS sits around 0.1-0.3 of full scale, so scale up before mapping to lips
LIP_GAIN = 4.0


def _samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def level(pcm: bytes) -> float:
    x = _samples(pcm)
    if x.size == 0:
        return 0.0
    return float(min(1.0, np.sqrt(np.mean(x * x))))


def envelope(pcm: bytes, rate: int, frame_ms: int = 40) -> list[float]:
    """Smoothed per-frame loudness 0..1 (instant attack, decaying release)."""
    step = rate * frame_ms // 1000
    out: list[float] = []
    prev = 0.0
    for i in range(0, len(pcm) // 2, step):
        raw = level(pcm[i * 2 : (i + step) * 2])
        prev = raw if raw > prev else prev * 0.6 + raw * 0.4
        out.append(prev if prev > 1e-4 else 0.0)
    return out


def lip_positions(env: list[float]) -> list[tuple[float, float]]:
    """(top, bottom) lip motor positions per frame; like the ohbot lib, both rise from 5 as the mouth opens."""
    res = []
    for v in env:
        p = LIP_CLOSED + LIP_SPAN * min(1.0, v * LIP_GAIN)
        res.append((p, p))
    return res
