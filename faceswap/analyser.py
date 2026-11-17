"""Detecção (SCRFD det_10g) e reconhecimento facial (ArcFace w600k_r50)."""
from dataclasses import dataclass

import cv2
import numpy as np

from . import models
from .geometry import ARCFACE_112, warp_face


@dataclass
class Face:
    bbox: np.ndarray            # x1, y1, x2, y2
    kps: np.ndarray             # (5, 2)
    score: float
    embedding: np.ndarray = None

    @property
    def area(self) -> float:
        return float((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))


class FaceDetector:
    STRIDES = (8, 16, 32)
    ANCHORS = 2

    def __init__(self, size: int = 640):
        self.size = size
        self.sess = models.session("det_10g")
        self.input = self.sess.get_inputs()[0].name
        self._centers = {}

    def _anchor_centers(self, stride):
        if stride not in self._centers:
            n = self.size // stride
            grid = np.stack(np.mgrid[:n, :n][::-1], axis=-1).astype(np.float32) * stride
            self._centers[stride] = np.repeat(grid.reshape(-1, 2), self.ANCHORS, axis=0)
        return self._centers[stride]

    def detect(self, frame: np.ndarray, threshold: float = 0.5, nms: float = 0.4) -> list:
        h, w = frame.shape[:2]
        scale = min(self.size / h, self.size / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128.0, (self.size, self.size), (127.5, 127.5, 127.5), swapRB=True)
        outs = [o[0] if o.ndim == 3 else o for o in self.sess.run(None, {self.input: blob})]

        boxes, kpss, scores = [], [], []
        for i, stride in enumerate(self.STRIDES):
            sc = outs[i].reshape(-1)
            keep = np.where(sc >= threshold)[0]
            if not len(keep):
                continue
            c = self._anchor_centers(stride)[keep]
            d = outs[i + 3][keep] * stride
            k = outs[i + 6][keep].reshape(-1, 5, 2) * stride
            boxes.append(np.concatenate([c - d[:, :2], c + d[:, 2:]], axis=1))
            kpss.append(k + c[:, None, :])
            scores.append(sc[keep])
        if not boxes:
            return []
        boxes = np.concatenate(boxes) / scale
        kpss = np.concatenate(kpss) / scale
        scores = np.concatenate(scores)
        xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1)
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), threshold, nms)
        idx = np.array(idx).reshape(-1)
        return [Face(boxes[i], kpss[i], float(scores[i])) for i in idx]


class FaceRecognizer:
    def __init__(self):
        self.sess = models.session("w600k_r50")
        self.input = self.sess.get_inputs()[0].name

    def embed(self, frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
        crop, _ = warp_face(frame, kps, ARCFACE_112, 112, cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(crop, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        e = self.sess.run(None, {self.input: blob})[0][0]
        return e / (np.linalg.norm(e) + 1e-8)
