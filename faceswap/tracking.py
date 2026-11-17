"""Rastreamento de rostos entre frames e suavização temporal dos pontos (sem tremido).

Como o vídeo é processado offline, a suavização é bidirecional (usa frames passados e futuros),
então não há atraso no movimento. Falhas curtas de detecção são preenchidas por interpolação,
o que evita o rosto original "piscar" de volta por 1 ou 2 frames.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    frames: dict = field(default_factory=dict)   # frame -> Face
    last_frame: int = -1

    @property
    def last(self):
        return self.frames[self.last_frame]

    def similarity(self, ref: np.ndarray) -> float:
        embs = [f.embedding for f in self.frames.values() if f.embedding is not None]
        return float(np.mean([e @ ref for e in embs])) if embs else -1.0

    def weight(self) -> float:
        return sum(f.area for f in self.frames.values())


class Tracker:
    def __init__(self, max_gap: int = 8, iou_threshold: float = 0.25, emb_threshold: float = 0.3):
        self.max_gap = max_gap
        self.iou_threshold = iou_threshold
        self.emb_threshold = emb_threshold
        self.tracks = []

    def update(self, frame_idx: int, faces: list) -> None:
        alive = [t for t in self.tracks if frame_idx - t.last_frame <= self.max_gap + 1]
        pairs = []
        for ti, t in enumerate(alive):
            for fi, f in enumerate(faces):
                score = iou(t.last.bbox, f.bbox)
                if t.last.embedding is not None and f.embedding is not None:
                    sim = float(t.last.embedding @ f.embedding)
                    if sim < self.emb_threshold and score < 0.6:
                        continue  # outra pessoa ocupando o mesmo lugar
                    score += 0.5 * sim
                if score >= self.iou_threshold:
                    pairs.append((score, ti, fi))
        used_t, used_f = set(), set()
        for _, ti, fi in sorted(pairs, reverse=True):
            if ti in used_t or fi in used_f:
                continue
            used_t.add(ti)
            used_f.add(fi)
            alive[ti].frames[frame_idx] = faces[fi]
            alive[ti].last_frame = frame_idx
        for fi, f in enumerate(faces):
            if fi not in used_f:
                t = Track(len(self.tracks), {frame_idx: f}, frame_idx)
                self.tracks.append(t)


def smooth_track(track: Track, sigma: float, max_gap: int) -> dict:
    """Retorna {frame: kps suavizados} preenchendo lacunas de até `max_gap` frames."""
    idx = np.array(sorted(track.frames))
    kps = np.stack([track.frames[i].kps for i in idx]).reshape(len(idx), -1)
    result = {}
    # divide em segmentos onde a lacuna é grande demais (ex.: rosto saiu de cena)
    breaks = np.where(np.diff(idx) > max_gap + 1)[0] + 1
    for seg_idx, seg_kps in zip(np.split(idx, breaks), np.split(kps, breaks)):
        full = np.arange(seg_idx[0], seg_idx[-1] + 1)
        filled = np.stack([np.interp(full, seg_idx, seg_kps[:, c]) for c in range(seg_kps.shape[1])], axis=1)
        if len(full) >= 5:
            filled = median_filter(filled, size=(3, 1), mode="nearest")  # remove picos isolados
        if sigma > 0 and len(full) > 1:
            filled = gaussian_filter1d(filled, sigma=sigma, axis=0, mode="nearest")
        for f, k in zip(full, filled):
            result[int(f)] = k.reshape(5, 2).astype(np.float32)
    return result
