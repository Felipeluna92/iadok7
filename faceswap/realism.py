"""Integração do rosto ao vídeo: o que separa um resultado "com cara de IA" de um profissional.

Um rosto gerado costuma ser mais nítido, mais limpo (sem granulação) e "ferver" de um frame para o
outro. Aqui o rosto final é degradado/estabilizado para casar com a câmera original:
  * nitidez igual à do rosto original no mesmo frame;
  * granulação (ruído do sensor/compressão) igual à do vídeo, nova a cada frame como a real;
  * estabilização temporal adaptativa ao movimento (sem rastro em movimentos rápidos);
  * transferência opcional do tom de pele da referência, preservando a luz da cena.
"""
import cv2
import numpy as np

from .geometry import mask_to_frame


def _gray(x: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(x, 0, 255).astype(np.float32), cv2.COLOR_BGR2GRAY)


def _detail_energy(gray: np.ndarray, m: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F)[m].var())


def noise_sigma(gray: np.ndarray, m: np.ndarray) -> float:
    """Estimativa robusta do ruído (MAD do resíduo de um filtro de mediana)."""
    residual = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    return float(1.4826 * np.median(np.abs(residual[m]))) if m.any() else 0.0


def match_sharpness(img: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask > 0.5
    if m.sum() < 200:
        return img
    target = _detail_energy(_gray(ref), m)
    if _detail_energy(_gray(img), m) <= target * 1.05:
        return img
    out = img
    for sigma in np.arange(0.3, 3.01, 0.15):
        out = cv2.GaussianBlur(img, (0, 0), float(sigma))
        if _detail_energy(_gray(out), m) <= target:
            break
    return out


_NOISE_CAL = 0.87  # estimador MAD/mediana-3x3 retorna ~0.87x o desvio real do ruído gaussiano


def match_grain(out: np.ndarray, original: np.ndarray, mask: np.ndarray, m: np.ndarray,
                rng: np.random.Generator, amount: float = 1.0) -> np.ndarray:
    """Aplica granulação direto na resolução do frame (depois da colagem), igualando o ruído
    do rosto novo ao ruído que o rosto original tinha nesse mesmo frame."""
    if amount <= 0:
        return out
    r = mask_to_frame(mask, m, out.shape)
    if r is None:
        return out
    (x1, y1, x2, y2), wmask = r
    sel = wmask > 0.5
    if sel.sum() < 200:
        return out
    s_ref = noise_sigma(_gray(original[y1:y2, x1:x2]), sel) / _NOISE_CAL * amount
    h, w = sel.shape
    region = out[y1:y2, x1:x2].astype(np.float32)
    for _ in range(3):  # textura de pele confunde o estimador: corrige em até 3 passos
        s_cur = noise_sigma(_gray(region), sel) / _NOISE_CAL
        add = np.sqrt(max(s_ref ** 2 - s_cur ** 2, 0.0))
        if add < 0.3:
            break
        noise = rng.standard_normal((h, w, 1)).astype(np.float32) * add
        noise = noise + rng.standard_normal((h, w, 3)).astype(np.float32) * (add * 0.3)
        region = region + noise * wmask[..., None]
    out[y1:y2, x1:x2] = np.clip(region, 0, 255).astype(np.uint8)
    return out


def temporal_stabilize(cur: np.ndarray, prev: np.ndarray, cur_orig: np.ndarray, prev_orig: np.ndarray,
                       strength: float, tau: float = 10.0) -> np.ndarray:
    """Mistura com o frame anterior onde o rosto original quase não mudou (pele parada),
    e não mistura onde houve movimento (boca falando, piscadas) — evita rastros."""
    if prev is None or strength <= 0:
        return cur
    diff = np.abs(_gray(cur_orig) - _gray(prev_orig))
    diff = cv2.GaussianBlur(diff, (0, 0), 3)
    w = (strength * np.exp(-diff / tau))[..., None]
    return cur + w * (prev - cur)


def transfer_skin_tone(img: np.ndarray, mask: np.ndarray, source_ab: np.ndarray, amount: float) -> np.ndarray:
    """Move a cor (canais a/b do LAB) da pele na direção da referência; a luminância (luz e
    sombras da cena) fica intacta."""
    m = mask > 0.5
    if amount <= 0 or source_ab is None or m.sum() < 200:
        return img
    lab = cv2.cvtColor((np.clip(img, 0, 255) / 255.0).astype(np.float32), cv2.COLOR_BGR2LAB)
    shift = (source_ab - lab[..., 1:][m].mean(0)) * amount
    lab[..., 1:] += shift * mask[..., None]
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) * 255.0


def skin_ab(img: np.ndarray, skin_mask: np.ndarray):
    m = skin_mask > 0.5
    if m.sum() < 200:
        return None
    lab = cv2.cvtColor((img / 255.0).astype(np.float32), cv2.COLOR_BGR2LAB)
    return lab[..., 1:][m].mean(0)
