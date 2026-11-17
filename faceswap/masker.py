"""Máscaras de mesclagem: caixa com borda suave + segmentação facial (BiSeNet) para oclusões."""
import cv2
import numpy as np

from . import models

# Regiões do BiSeNet consideradas "rosto": pele, sobrancelhas, olhos, óculos, nariz, boca, lábios.
FACE_REGIONS = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def box_mask(size: int, blur: float, padding=(0, 0, 0, 0)) -> np.ndarray:
    """padding = (cima, direita, baixo, esquerda) em % do tamanho."""
    blur_amount = int(size * 0.5 * blur)
    blur_area = max(blur_amount // 2, 1)
    m = np.ones((size, size), np.float32)
    top, right, bottom, left = (int(size * p / 100) for p in padding)
    m[:max(blur_area, top), :] = 0
    m[-max(blur_area, bottom):, :] = 0
    m[:, :max(blur_area, left)] = 0
    m[:, -max(blur_area, right):] = 0
    if blur_amount > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur_amount * 0.25)
    return m


class FaceParser:
    SIZE = 512

    def __init__(self):
        self.sess = models.session("bisenet_resnet_34")
        self.input = self.sess.get_inputs()[0].name

    def mask(self, crop: np.ndarray, size: int, dilate: int = 0) -> np.ndarray:
        x = cv2.resize(crop.astype(np.uint8), (self.SIZE, self.SIZE), interpolation=cv2.INTER_LINEAR)
        x = (x[:, :, ::-1].astype(np.float32) / 255.0 - _MEAN) / _STD
        logits = self.sess.run(None, {self.input: x.transpose(2, 0, 1)[None]})[0][0]
        m = np.isin(logits.argmax(0), FACE_REGIONS).astype(np.float32)
        if dilate > 0:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1,) * 2))
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR)
        m = cv2.GaussianBlur(m, (0, 0), max(size / 128.0, 1.0))
        return ((m.clip(0.5, 1) - 0.5) * 2).astype(np.float32)
