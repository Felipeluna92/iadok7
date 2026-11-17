"""Troca de rosto: HyperSwap (256 px nativo) ou InSwapper (128 px), com "pixel boost".

Pixel boost: o recorte é feito em resolução maior (ex.: 512) e dividido em sub-amostras de pixels;
cada uma é um rosto completo no tamanho do modelo, trocada separadamente e remontada.
"""
import numpy as np

from . import models

SWAPPERS = {
    "hyperswap_1a_256": {"size": 256, "kind": "hyperswap"},
    "hyperswap_1b_256": {"size": 256, "kind": "hyperswap"},
    "hyperswap_1c_256": {"size": 256, "kind": "hyperswap"},
    "inswapper_128": {"size": 128, "kind": "inswapper"},
}


class FaceSwapper:
    def __init__(self, name: str = "hyperswap_1a_256"):
        cfg = SWAPPERS[name]
        self.name, self.size, self.kind = name, cfg["size"], cfg["kind"]
        self.emap = None
        if self.kind == "inswapper":
            import onnx
            from onnx import numpy_helper
            graph = onnx.load(str(models.model_path(name))).graph
            self.emap = numpy_helper.to_array(graph.initializer[-1]).astype(np.float32)
        self.sess = models.session(name)
        ins = self.sess.get_inputs()
        self.target_name = next(i.name for i in ins if len(i.shape) == 4)
        self.source_name = next(i.name for i in ins if len(i.shape) == 2)

    def latent(self, source: np.ndarray, target: np.ndarray = None, strength: float = 0.0) -> np.ndarray:
        """`strength` > 0 afasta a identidade do rosto original do vídeo (reforça a da referência)."""
        e = source.astype(np.float32).reshape(-1)
        if target is not None and strength:
            e = e * (1 + strength) - target.reshape(-1) * strength
        e = e / np.linalg.norm(e)
        if self.emap is not None:
            e = e @ self.emap
            e = e / np.linalg.norm(e)
        return e.reshape(1, -1).astype(np.float32)

    def _swap(self, crop: np.ndarray, latent: np.ndarray):
        x = crop[:, :, ::-1].astype(np.float32) / 255.0
        if self.kind == "hyperswap":
            x = (x - 0.5) / 0.5
        outs = self.sess.run(None, {self.target_name: x.transpose(2, 0, 1)[None], self.source_name: latent})
        y = outs[0][0].transpose(1, 2, 0)
        if self.kind == "hyperswap":
            y = y * 0.5 + 0.5
        img = np.clip(y, 0, 1)[:, :, ::-1] * 255.0
        mask = np.clip(outs[1][0, 0], 0, 1) if len(outs) > 1 else None
        return img, mask

    def swap(self, crop: np.ndarray, latent: np.ndarray, boost: int = 1):
        """Retorna (rosto trocado float BGR, máscara do modelo ou None) no tamanho de `crop`."""
        if boost == 1:
            return self._swap(crop, latent)
        n, s = boost, self.size
        tiles = crop.reshape(s, n, s, n, 3).transpose(1, 3, 0, 2, 4).reshape(n * n, s, s, 3)
        results = [self._swap(t, latent) for t in tiles]
        img = np.stack([r[0] for r in results]).reshape(n, n, s, s, 3).transpose(2, 0, 3, 1, 4).reshape(s * n, s * n, 3)
        mask = None
        if results[0][1] is not None:
            mask = np.stack([r[1] for r in results]).reshape(n, n, s, s).transpose(2, 0, 3, 1).reshape(s * n, s * n)
        return img, mask
