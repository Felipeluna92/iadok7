"""Troca de rosto (InSwapper 128) com "pixel boost" para gerar 256/512 px de detalhe real."""
import numpy as np
import onnx
from onnx import numpy_helper

from . import models

MODEL_SIZE = 128


class FaceSwapper:
    def __init__(self):
        path = models.model_path("inswapper_128")
        graph = onnx.load(str(path)).graph
        self.emap = numpy_helper.to_array(graph.initializer[-1]).astype(np.float32)
        self.sess = models.session("inswapper_128")
        ins = self.sess.get_inputs()
        self.target_name = next(i.name for i in ins if len(i.shape) == 4)
        self.source_name = next(i.name for i in ins if len(i.shape) == 2)

    def latent(self, embedding: np.ndarray) -> np.ndarray:
        lat = embedding.reshape(1, -1).astype(np.float32) @ self.emap
        return (lat / np.linalg.norm(lat)).astype(np.float32)

    def _swap(self, crop: np.ndarray, latent: np.ndarray) -> np.ndarray:
        blob = (crop[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.target_name: blob, self.source_name: latent})[0][0]
        return np.clip(out.transpose(1, 2, 0)[:, :, ::-1] * 255.0, 0, 255)

    def swap(self, crop: np.ndarray, latent: np.ndarray, boost: int = 1) -> np.ndarray:
        """`crop` tem tamanho 128*boost. Cada sub-amostra de pixels vira um rosto 128 completo,
        é trocada separadamente e remontada, preservando a resolução maior."""
        if boost == 1:
            return self._swap(crop, latent)
        n, s = boost, MODEL_SIZE
        tiles = crop.reshape(s, n, s, n, 3).transpose(1, 3, 0, 2, 4).reshape(n * n, s, s, 3)
        outs = np.stack([self._swap(t, latent) for t in tiles])
        return outs.reshape(n, n, s, s, 3).transpose(2, 0, 3, 1, 4).reshape(s * n, s * n, 3)
