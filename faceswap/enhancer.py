"""Restauração de rosto em 512 px (GFPGAN 1.4, GPEN 512 ou CodeFormer)."""
import numpy as np

from . import models

ENHANCERS = ("gfpgan_1.4", "gpen_bfr_512", "codeformer")


class FaceEnhancer:
    SIZE = 512

    def __init__(self, name: str, codeformer_fidelity: float = 0.7):
        self.sess = models.session(name)
        self.inputs = self.sess.get_inputs()
        self.fidelity = codeformer_fidelity

    def enhance(self, crop: np.ndarray) -> np.ndarray:
        x = crop[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        feed = {}
        for i in self.inputs:
            if len(i.shape) == 4:
                feed[i.name] = x
            else:  # CodeFormer: peso de fidelidade (0 = mais detalhe, 1 = mais fiel)
                feed[i.name] = np.array(self.fidelity, dtype=np.float64 if "double" in i.type else np.float32)
        out = self.sess.run(None, feed)[0][0]
        out = (np.clip(out, -1, 1) + 1) / 2 * 255.0
        return out.transpose(1, 2, 0)[:, :, ::-1]
