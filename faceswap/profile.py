"""Análise da referência e do vídeo: extrai as características do rosto e avalia a compatibilidade.

Para cada foto de referência: qualidade (nitidez, resolução, ângulo, confiança), idade, gênero e tom
de pele. Fotos que parecem de outra pessoa são descartadas, e a identidade final é uma média
ponderada pela qualidade — as melhores fotos pesam mais.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import models
from .geometry import FFHQ_512, warp_face


class AgeGender:
    SIZE = 96

    def __init__(self):
        self.sess = models.session("genderage")
        self.input = self.sess.get_inputs()[0].name

    def predict(self, frame: np.ndarray, bbox: np.ndarray):
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        sc = self.SIZE / (max(w, h) * 1.5)
        m = np.float32([[sc, 0, self.SIZE / 2 - cx * sc], [0, sc, self.SIZE / 2 - cy * sc]])
        crop = cv2.warpAffine(frame, m, (self.SIZE, self.SIZE), borderValue=0)
        blob = cv2.dnn.blobFromImage(crop, 1.0, (self.SIZE, self.SIZE), (0, 0, 0), swapRB=True)
        pred = self.sess.run(None, {self.input: blob})[0][0]
        return ("masculino" if np.argmax(pred[:2]) == 1 else "feminino"), int(round(pred[2] * 100))


def pose(kps: np.ndarray):
    """Estimativa simples de ângulo a partir dos 5 pontos: (giro lateral, inclinação vertical)."""
    eye_mid = (kps[0] + kps[1]) / 2
    mouth_mid = (kps[3] + kps[4]) / 2
    eye_dist = np.linalg.norm(kps[1] - kps[0]) + 1e-6
    yaw = float((kps[2, 0] - eye_mid[0]) / eye_dist)
    pitch = float((kps[2, 1] - eye_mid[1]) / (mouth_mid[1] - eye_mid[1] + 1e-6))
    return yaw, pitch


def sharpness(crop: np.ndarray) -> float:
    g = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


@dataclass
class FaceReport:
    name: str
    ok: bool
    quality: float = 0.0
    eye_dist: float = 0.0
    sharp: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.5
    gender: str = ""
    age: int = 0
    similarity: float = 1.0
    notes: list = field(default_factory=list)


@dataclass
class SourceProfile:
    embedding: np.ndarray
    skin_ab: np.ndarray
    reports: list
    gender: str
    age: int
    eye_dist: float


def _quality(eye_dist, sharp, yaw, pitch, score):
    q_res = np.clip(eye_dist / 90.0, 0, 1)               # ~90 px entre os olhos = ótimo
    q_sharp = np.clip(sharp / 250.0, 0, 1)
    q_pose = np.clip(1 - abs(yaw) / 0.45, 0, 1) * np.clip(1 - abs(pitch - 0.5) / 0.35, 0, 1)
    return float(100 * (0.3 * q_res + 0.3 * q_sharp + 0.3 * q_pose + 0.1 * score))


def analyse_face(pipe, image: np.ndarray, name: str):
    """Retorna (FaceReport, embedding, skin_ab) do maior rosto da imagem."""
    face, image = pipe.largest_face(image)
    if face is None:
        return FaceReport(name, False, notes=["nenhum rosto encontrado"]), None, None
    emb = pipe.recognizer.embed(image, face.kps)
    crop, _ = warp_face(image, face.kps, FFHQ_512, 512)
    eye_dist = float(np.linalg.norm(face.kps[1] - face.kps[0]))
    sharp = sharpness(cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA))
    yaw, pitch = pose(face.kps)
    gender, age = pipe.agegender.predict(image, face.bbox)
    labels = pipe.parser.labels(crop)
    from .realism import skin_ab
    ab = skin_ab(crop.astype(np.float32), (labels == 1).astype(np.float32))
    r = FaceReport(name, True, _quality(eye_dist, sharp, yaw, pitch, face.score), eye_dist, sharp,
                   yaw, pitch, gender, age)
    if eye_dist < 45:
        r.notes.append("rosto pequeno/baixa resolução")
    if sharp < 80:
        r.notes.append("foto desfocada")
    if abs(yaw) > 0.3:
        r.notes.append("rosto muito de lado")
    if abs(pitch - 0.5) > 0.25:
        r.notes.append("cabeça muito inclinada")
    if np.isin(labels, [6]).mean() > 0.01:
        r.notes.append("óculos detectados (podem ser transferidos)")
    return r, emb, ab


def build_source_profile(pipe, images: list, names: list = None) -> SourceProfile:
    names = names or [f"foto {i + 1}" for i in range(len(images))]
    items = [analyse_face(pipe, img, n) for img, n in zip(images, names)]
    valid = [(r, e, ab) for r, e, ab in items if e is not None]
    if not valid:
        raise ValueError("Nenhum rosto encontrado nas fotos de referência.")

    # descarta fotos que parecem de outra pessoa (comparadas com a média das demais)
    embs = np.stack([e for _, e, _ in valid])
    keep = np.ones(len(valid), bool)
    if len(valid) >= 3:
        for i in range(len(valid)):
            others = np.delete(embs, i, axis=0).mean(0)
            sim = float(embs[i] @ (others / np.linalg.norm(others)))
            valid[i][0].similarity = sim
            if sim < 0.45:
                keep[i] = False
                valid[i][0].ok = False
                valid[i][0].notes.append("parece ser outra pessoa — ignorada")

    chosen = [v for v, k in zip(valid, keep) if k] or valid
    w = np.array([max(r.quality, 5.0) for r, _, _ in chosen]) ** 2   # as melhores fotos pesam mais
    emb = (np.stack([e for _, e, _ in chosen]) * w[:, None]).sum(0)
    emb /= np.linalg.norm(emb)
    abs_ = [(ab, wi) for (_, _, ab), wi in zip(chosen, w) if ab is not None]
    skin = np.average([a for a, _ in abs_], axis=0, weights=[wi for _, wi in abs_]) if abs_ else None
    best = max(chosen, key=lambda v: v[0].quality)[0]
    genders = [r.gender for r, _, _ in chosen]
    return SourceProfile(emb, skin, [i[0] for i in items], max(set(genders), key=genders.count),
                         int(np.median([r.age for r, _, _ in chosen])), best.eye_dist)


def analyse_video(pipe, video: str, samples: int = 12) -> dict:
    from .video_io import prepare_input, probe, read_frames
    video = prepare_input(video)
    info = probe(video)
    n = max(info["frames"], 1)
    picks = set(np.linspace(0, n - 1, min(samples, n)).astype(int).tolist())
    data = []
    for idx, frame in read_frames(video):
        if idx not in picks:
            continue
        faces = pipe.detector.detect(frame, 0.5)
        if not faces:
            continue
        f = max(faces, key=lambda x: x.area)
        crop, _ = warp_face(frame, f.kps, FFHQ_512, 512)
        labels = pipe.parser.labels(crop)
        from .realism import skin_ab
        g, a = pipe.agegender.predict(frame, f.bbox)
        data.append({"eye": float(np.linalg.norm(f.kps[1] - f.kps[0])), "yaw": pose(f.kps)[0], "gender": g,
                     "age": a, "ab": skin_ab(crop.astype(np.float32), (labels == 1).astype(np.float32)),
                     "faces": len(faces)})
        if idx >= max(picks):
            break
    if not data:
        return {"ok": False}
    genders = [d["gender"] for d in data]
    abs_ = [d["ab"] for d in data if d["ab"] is not None]
    return {"ok": True, "frames_with_face": len(data), "samples": len(picks), "info": info,
            "eye": float(np.median([d["eye"] for d in data])), "eye_max": float(max(d["eye"] for d in data)),
            "yaw_max": float(max(abs(d["yaw"]) for d in data)),
            "gender": max(set(genders), key=genders.count), "age": int(np.median([d["age"] for d in data])),
            "ab": np.mean(abs_, axis=0) if abs_ else None, "multi": max(d["faces"] for d in data) > 1}


def compatibility_markdown(src: SourceProfile, vid: dict) -> str:
    lines = ["_Idade é uma estimativa aproximada; vale mais a diferença entre referência e vídeo._", "",
             "### Fotos de referência", "| Foto | Qualidade | Resolução | Nitidez | Ângulo | Idade/Gênero | Observações |",
             "|---|---|---|---|---|---|---|"]
    for r in src.reports:
        if not r.eye_dist:
            lines.append(f"| {r.name} | — | — | — | — | — | {'; '.join(r.notes)} |")
            continue
        ang = "frontal" if abs(r.yaw) < 0.12 else ("leve" if abs(r.yaw) < 0.3 else "perfil")
        lines.append(f"| {r.name} | **{r.quality:.0f}/100** | {r.eye_dist:.0f}px | {r.sharp:.0f} | {ang} | "
                     f"~{r.age} / {r.gender} | {'; '.join(r.notes) or 'ok'} |")
    lines += ["", f"Identidade final: média ponderada de {sum(r.ok for r in src.reports)} foto(s), "
                  f"~{src.age} anos, {src.gender}."]
    if not vid.get("ok"):
        return "\n".join(lines + ["", "**Nenhum rosto encontrado no vídeo.**"])

    lines += ["", "### Vídeo", f"- Rosto encontrado em {vid['frames_with_face']}/{vid['samples']} amostras; "
                               f"~{vid['age']} anos, {vid['gender']}; distância entre olhos ~{vid['eye']:.0f}px "
                               f"(máx. {vid['eye_max']:.0f}px)."]
    lines += ["", "### Compatibilidade"]
    tips = []
    if vid["gender"] != src.gender:
        tips.append("⚠️ Gênero diferente entre referência e vídeo: formato do rosto, cabelo e pescoço do vídeo "
                    "vão denunciar a troca. Prefira um vídeo de alguém do mesmo gênero.")
    if abs(vid["age"] - src.age) > 15:
        tips.append(f"⚠️ Diferença de idade grande (~{abs(vid['age'] - src.age)} anos): a pele/rugas vêm do vídeo.")
    if vid["eye_max"] > src.eye_dist * 1.6:
        tips.append(f"⚠️ O rosto no vídeo chega a {vid['eye_max']:.0f}px entre os olhos, mas a melhor foto tem "
                    f"{src.eye_dist:.0f}px. Use uma foto de resolução maior para não perder detalhes.")
    if vid["ab"] is not None and src.skin_ab is not None:
        d = float(np.linalg.norm(vid["ab"] - src.skin_ab))
        if d > 8:
            tips.append(f"⚠️ Tom de pele bem diferente (Δ={d:.0f}). A 'Cor da pele igual à do vídeo' já "
                        "corrige o rosto para combinar com orelhas e pescoço; se preferir a cor da referência, "
                        "use 'Tom de pele da referência'.")
    if vid["yaw_max"] > 0.35:
        tips.append("O vídeo tem momentos de perfil; inclua uma foto de referência mais de lado para esses trechos.")
    if vid["multi"]:
        tips.append("Há mais de um rosto em alguns frames; o modo 'rosto principal' troca só o maior.")
    if not tips:
        tips.append("✅ Boa compatibilidade entre referência e vídeo.")
    return "\n".join(lines + [f"- {t}" for t in tips])
