"""DCT-QIM Watermark — Two variants.

Variant 1 (uniform): fixed Δ for all blocks, blind extraction.
Variant 2 (content-adaptive): per-block effective Δ modulated by the
semantic strength map (embedding key). Extraction is key-assisted:
each block is demodulated with its own Δ_eff and votes are weighted by
strength. This makes the strength map genuinely affect the embedding —
instance regions carry stronger, more robust watermark signals.

Δ_eff(b) = Δ * (0.5 + S_block),  S_block ∈ [0.1, 1.0]
         → Δ_eff ∈ [0.6Δ, 1.5Δ]
Instance regions (S≈0.8) → Δ_eff ≈ 1.3Δ  (strong embedding)
Background    (S≈0.3) → Δ_eff ≈ 0.8Δ  (light embedding)
"""
import numpy as np
import cv2


# ── Shared helpers ─────────────────────────────────────────────
def _block_grid(H, W):
    H -= H % 8
    W -= W % 8
    return H, W, H // 8, W // 8


def _prep_y(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)
    return y, ycrcb


def _restore(y, ycrcb, H, W):
    out = ycrcb.copy()
    out[:H, :W, 0] = np.clip(y, 0, 255)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# Variant 1: UNIFORM (fixed Δ, blind extraction)
# ═══════════════════════════════════════════════════════════════
def embed_uniform(cover, message, delta=15):
    """Fixed-Δ QIM embedding. Strength map NOT used."""
    H, W, n_h, n_w = _block_grid(*cover.shape[:2])
    y, ycrcb = _prep_y(cover)
    y = y[:H, :W]
    for i in range(n_h):
        for j in range(n_w):
            bi = (i * n_w + j) % len(message)
            b = y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            d = cv2.dct(b.astype(np.float32))
            c = d[4, 1]
            if message[bi] == 1:
                d[4, 1] = np.floor(c / delta) * delta + delta / 2
            else:
                d[4, 1] = np.round(c / delta) * delta
            y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = cv2.idct(d)
    return _restore(y, ycrcb, H, W)


def extract_uniform(watermarked, msg_len=64, delta=15):
    """Blind extraction with uniform weights."""
    H, W, n_h, n_w = _block_grid(*watermarked.shape[:2])
    y, _ = _prep_y(watermarked)
    y = y[:H, :W]
    votes = [[] for _ in range(msg_len)]
    for i in range(n_h):
        for j in range(n_w):
            bi = (i * n_w + j) % msg_len
            d = cv2.dct(y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].astype(np.float32))
            bit = _demodulate_bit(d[4, 1], delta)
            votes[bi].append((bit, 1.0))
    return _majority_vote(votes, msg_len)


# ═══════════════════════════════════════════════════════════════
# Variant 2: CONTENT-ADAPTIVE (per-block Δ_eff from strength map)
# ═══════════════════════════════════════════════════════════════
def _delta_eff_map(strength_map, H, W, delta):
    """Per-block effective Δ: Δ_eff = Δ*(1.0 + 0.5*S).
    Background (S=0.3) → 1.15Δ (never weaker than uniform),
    instance (S=0.8) → 1.40Δ (stronger). Returns (n_h, n_w) array."""
    sm = cv2.resize(strength_map, (W, H))
    n_h, n_w = H // 8, W // 8
    deff = np.zeros((n_h, n_w), dtype=np.float32)
    for i in range(n_h):
        for j in range(n_w):
            s = sm[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].mean()
            deff[i, j] = delta * (1.0 + 0.5 * s)
    return deff


def embed_adaptive(cover, message, strength_map, delta=15):
    """Content-adaptive embedding: per-block Δ_eff = Δ*(0.5+S_block).
    The strength map IS the embedding key — instance regions get
    stronger watermark signals (larger effective step)."""
    H, W, n_h, n_w = _block_grid(*cover.shape[:2])
    y, ycrcb = _prep_y(cover)
    y = y[:H, :W]
    deff = _delta_eff_map(strength_map, H, W, delta)
    for i in range(n_h):
        for j in range(n_w):
            bi = (i * n_w + j) % len(message)
            d = cv2.dct(y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].astype(np.float32))
            c = d[4, 1]
            dd = deff[i, j]
            if message[bi] == 1:
                d[4, 1] = np.floor(c / dd) * dd + dd / 2
            else:
                d[4, 1] = np.round(c / dd) * dd
            y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = cv2.idct(d)
    return _restore(y, ycrcb, H, W)


def extract_adaptive(watermarked, msg_len=64, delta=15, strength_map=None):
    """Key-assisted extraction: demodulate each block with its Δ_eff
    (needs the strength map / embedding key) and weight votes by S.
    If strength_map is None → fall back to uniform-Δ demodulation
    with uniform weights (blind, degraded)."""
    H, W, n_h, n_w = _block_grid(*watermarked.shape[:2])
    y, _ = _prep_y(watermarked)
    y = y[:H, :W]
    if strength_map is not None:
        deff = _delta_eff_map(strength_map, H, W, delta)
        sm = cv2.resize(strength_map, (W, H))
    votes = [[] for _ in range(msg_len)]
    for i in range(n_h):
        for j in range(n_w):
            bi = (i * n_w + j) % msg_len
            d = cv2.dct(y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].astype(np.float32))
            if strength_map is not None:
                bit = _demodulate_bit(d[4, 1], deff[i, j])
                w = sm[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].mean()
            else:
                bit = _demodulate_bit(d[4, 1], delta)
                w = 1.0
            votes[bi].append((bit, w))
    return _majority_vote(votes, msg_len)


# ── Helpers ────────────────────────────────────────────────────
def _demodulate_bit(c, delta):
    q_floor = np.floor(c / delta)
    q_ceil = np.ceil(c / delta)
    q0_candidates = [q_floor * delta, q_ceil * delta]
    q1_candidates = [q * delta + delta / 2 for q in [q_floor, q_floor - 1, q_floor + 1]]
    d0 = min(abs(c - x) for x in q0_candidates)
    d1 = min(abs(c - x) for x in q1_candidates)
    return 1 if d1 < d0 else 0


def _majority_vote(votes, msg_len):
    result = np.zeros(msg_len, dtype=np.uint8)
    for k in range(msg_len):
        if not votes[k]:
            continue
        w_sum = sum(w for _, w in votes[k])
        w1_sum = sum(w for b, w in votes[k] if b == 1)
        result[k] = 1 if w1_sum > w_sum / 2 else 0
    return result


def bit_accuracy(msg1, msg2):
    return np.mean(msg1 == msg2)


# Backward-compatible aliases
def embed_watermark(cover, message, strength_map=None, delta=15):
    if strength_map is None:
        return embed_uniform(cover, message, delta=delta)
    return embed_adaptive(cover, message, strength_map, delta=delta)


def extract_watermark(watermarked, msg_len=64, delta=15, strength_map=None):
    if strength_map is None:
        return extract_uniform(watermarked, msg_len=msg_len, delta=delta)
    return extract_adaptive(watermarked, msg_len=msg_len, delta=delta,
                            strength_map=strength_map)
