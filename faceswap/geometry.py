"""Alinhamento de rostos (templates de 5 pontos), recorte e colagem de volta no frame."""
import cv2
import numpy as np

# Templates normalizados (0..1) de 5 pontos: olho esq., olho dir., nariz, canto boca esq., canto boca dir.
ARCFACE_112 = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], np.float32) / 112.0
ARCFACE_128 = np.array([
    [0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
    [0.38710391, 0.72160547], [0.61507734, 0.72034453]], np.float32)
FFHQ_512 = np.array([
    [0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
    [0.39308822, 0.72541100], [0.61150205, 0.72490465]], np.float32)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Transformação de similaridade por mínimos quadrados (Umeyama). Retorna matriz 2x3."""
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    u, sig, vt = np.linalg.svd(cov)
    e = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        e[1, 1] = -1
    r = u @ e @ vt
    scale = (sig * np.diag(e)).sum() / s.var(0).sum()
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]]).astype(np.float32)


def warp_face(frame, kps, template, size, interp=cv2.INTER_CUBIC):
    m = similarity_transform(kps, template * size)
    crop = cv2.warpAffine(frame, m, (size, size), flags=interp, borderMode=cv2.BORDER_REPLICATE)
    return crop, m


def paste_back(frame: np.ndarray, crop: np.ndarray, mask: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Cola `crop` (float BGR) em `frame` usando a inversa de `m`, só na região afetada."""
    inv = cv2.invertAffineTransform(m)
    h, w = crop.shape[:2]
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
    pts = corners @ inv[:, :2].T + inv[:, 2]
    fh, fw = frame.shape[:2]
    x1, y1 = np.floor(pts.min(0)).astype(int)
    x2, y2 = np.ceil(pts.max(0)).astype(int)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, fw), min(y2, fh)
    if x2 <= x1 or y2 <= y1:
        return frame
    inv_roi = inv.copy()
    inv_roi[:, 2] -= (x1, y1)
    size = (x2 - x1, y2 - y1)
    warped = cv2.warpAffine(crop, inv_roi, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wmask = cv2.warpAffine(mask, inv_roi, size, flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = np.clip(wmask * warped + (1.0 - wmask) * roi, 0, 255).astype(np.uint8)
    return frame
