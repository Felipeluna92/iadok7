"""Pipeline completo: análise (detecção + rastreamento + suavização) e renderização."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .analyser import FaceDetector, FaceRecognizer
from .enhancer import FaceEnhancer
from .geometry import ARCFACE_128, FFHQ_512, paste_back, warp_face
from .masker import FaceParser, box_mask
from .swapper import MODEL_SIZE, FaceSwapper
from .tracking import Tracker, smooth_track
from .video_io import VideoWriter, probe, read_frames


@dataclass
class Settings:
    mode: str = "largest"              # largest | reference | all
    reference_threshold: float = 0.35  # similaridade mínima com a foto de referência
    detect_threshold: float = 0.5
    smoothing: float = 1.5             # sigma (em frames) da suavização dos pontos do rosto
    max_gap: int = 8                   # frames sem detecção que são preenchidos por interpolação
    pixel_boost: int = 2               # 1 = 128px, 2 = 256px, 4 = 512px
    enhancer: str = "gfpgan_1.4"       # "none" | gfpgan_1.4 | gpen_bfr_512 | codeformer
    enhancer_blend: float = 0.8
    codeformer_fidelity: float = 0.7
    mask_blur: float = 0.3
    occlusion: bool = True             # usa segmentação facial (mãos/objetos na frente do rosto)
    mask_padding: tuple = (0, 0, 0, 0)  # cima, direita, baixo, esquerda (%)
    mask_smoothing: float = 0.5        # estabilidade da máscara entre frames (0 = desligado)
    crf: int = 16                      # qualidade do H.264 (menor = melhor)
    preset: str = "slow"
    watermark: bool = True
    keep_audio: bool = True


def add_watermark(frame: np.ndarray, text: str = "Gerado por IA") -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max(h / 1080 * 0.8, 0.4)
    thick = max(int(scale * 2), 1)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    org = (w - tw - int(20 * scale), h - int(20 * scale))
    overlay = frame.copy()
    cv2.putText(overlay, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(overlay, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)


class FaceSwapPipeline:
    def __init__(self):
        self._detector = self._recognizer = self._swapper = self._parser = None
        self._enhancers = {}

    # ---- modelos carregados sob demanda ----
    @property
    def detector(self):
        self._detector = self._detector or FaceDetector()
        return self._detector

    @property
    def recognizer(self):
        self._recognizer = self._recognizer or FaceRecognizer()
        return self._recognizer

    @property
    def swapper(self):
        self._swapper = self._swapper or FaceSwapper()
        return self._swapper

    @property
    def parser(self):
        self._parser = self._parser or FaceParser()
        return self._parser

    def enhancer(self, name: str, fidelity: float):
        if name not in self._enhancers:
            self._enhancers[name] = FaceEnhancer(name)
        self._enhancers[name].fidelity = fidelity
        return self._enhancers[name]

    # ---- identidade ----
    def largest_face_embedding(self, image: np.ndarray, threshold: float = 0.5):
        faces = self.detector.detect(image, threshold)
        if not faces:  # fotos recortadas muito justas: adiciona margem e tenta de novo
            pad = max(image.shape[:2]) // 2
            image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            faces = self.detector.detect(image, threshold * 0.8)
        if not faces:
            return None
        face = max(faces, key=lambda f: f.area)
        return self.recognizer.embed(image, face.kps)

    def source_embedding(self, images: list) -> np.ndarray:
        """Média das identidades de várias fotos = rosto trocado mais fiel e estável."""
        embs = [e for e in (self.largest_face_embedding(img) for img in images) if e is not None]
        if not embs:
            raise ValueError("Nenhum rosto encontrado nas fotos de origem.")
        e = np.mean(embs, axis=0)
        return e / np.linalg.norm(e)

    # ---- passo 1: análise ----
    def analyse(self, video: str, s: Settings, reference=None, start=0, end=None, progress=None) -> dict:
        info = probe(video)
        total = max((end or info["frames"]) - start, 1)
        tracker = Tracker(max_gap=s.max_gap)
        for idx, frame in read_frames(video, start, end):
            faces = self.detector.detect(frame, s.detect_threshold)
            for f in faces:
                f.embedding = self.recognizer.embed(frame, f.kps)
            tracker.update(idx, faces)
            if progress and idx % 5 == 0:
                progress(min((idx - start) / total, 1.0), "Analisando rostos")

        tracks = [t for t in tracker.tracks if len(t.frames) >= min(3, total)]
        if not tracks:
            return {}
        if s.mode == "reference":
            if reference is None:
                raise ValueError("Envie uma foto de referência para o modo 'pessoa da referência'.")
            tracks = [t for t in tracks if t.similarity(reference) >= s.reference_threshold]
        elif s.mode == "largest":
            main = max(tracks, key=lambda t: t.weight())
            main_emb = np.mean([f.embedding for f in main.frames.values()], axis=0)
            main_emb /= np.linalg.norm(main_emb)
            # a mesma pessoa pode virar outra "trilha" depois de um corte de cena
            tracks = [t for t in tracks if t is main or t.similarity(main_emb) >= 0.45]

        plan = {}
        for t in tracks:
            for f, kps in smooth_track(t, s.smoothing, s.max_gap).items():
                if start <= f < (end if end is not None else f + 1):
                    plan.setdefault(f, []).append((t.id, kps))
        return plan

    # ---- passo 2: renderização ----
    def render(self, frame: np.ndarray, faces: list, latent: np.ndarray, s: Settings, state: dict) -> np.ndarray:
        out = frame.copy()
        size = MODEL_SIZE * s.pixel_boost
        bmask = box_mask(size, s.mask_blur, s.mask_padding)
        seen = set()
        for tid, kps in faces:
            crop, m = warp_face(frame, kps, ARCFACE_128, size)
            swapped = self.swapper.swap(crop, latent, s.pixel_boost)
            mask = bmask
            if s.occlusion:
                region = self.parser.mask(crop, size)
                prev = state.get(tid)
                if prev is not None and s.mask_smoothing > 0:
                    region = s.mask_smoothing * prev + (1 - s.mask_smoothing) * region
                state[tid] = region
                seen.add(tid)
                mask = np.minimum(bmask, region)
            out = paste_back(out, swapped, mask, m)

        if s.enhancer and s.enhancer != "none":
            enh = self.enhancer(s.enhancer, s.codeformer_fidelity)
            emask = box_mask(enh.SIZE, max(s.mask_blur, 0.3)) * s.enhancer_blend
            for _, kps in faces:
                crop, m = warp_face(out, kps, FFHQ_512, enh.SIZE)
                out = paste_back(out, enh.enhance(crop), emask, m)

        for tid in list(state):
            if tid not in seen:
                del state[tid]
        return add_watermark(out) if s.watermark else out

    # ---- APIs de alto nível ----
    def process_video(self, sources: list, video: str, output: str, s: Settings,
                      reference_image=None, progress=None) -> Path:
        progress = progress or (lambda p, d="": None)
        latent = self.swapper.latent(self.source_embedding(sources))
        reference = self.largest_face_embedding(reference_image) if reference_image is not None else None
        if s.mode == "reference" and reference is None:
            raise ValueError("Nenhum rosto encontrado na foto de referência.")

        plan = self.analyse(video, s, reference, progress=lambda p, d: progress(p * 0.25, d))
        if not plan:
            raise ValueError("Nenhum rosto correspondente foi encontrado no vídeo.")

        info = probe(video)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        writer = VideoWriter(output, info["width"], info["height"], info["fps"], s.crf, s.preset,
                             audio_from=video if s.keep_audio else None)
        state, total = {}, max(info["frames"], 1)
        try:
            for idx, frame in read_frames(video):
                faces = plan.get(idx)
                if faces:
                    frame = self.render(frame, faces, latent, s, state)
                elif s.watermark:
                    frame = add_watermark(frame)
                    state.clear()
                writer.write(frame)
                if idx % 2 == 0:
                    progress(0.25 + 0.75 * min(idx / total, 1.0), f"Renderizando frame {idx}/{total}")
        finally:
            result = writer.close()
        progress(1.0, "Concluído")
        return result

    def preview(self, sources: list, video: str, time_sec: float, s: Settings, reference_image=None):
        """Processa um único frame (com contexto de ±15 frames para a suavização)."""
        latent = self.swapper.latent(self.source_embedding(sources))
        reference = self.largest_face_embedding(reference_image) if reference_image is not None else None
        info = probe(video)
        target = min(int(time_sec * info["fps"]), max(info["frames"] - 1, 0))
        start, end = max(target - 15, 0), target + 16
        plan = self.analyse(video, s, reference, start, end)
        frame = next(read_frames(video, target, target + 1))[1]
        faces = plan.get(target)
        if not faces:
            return frame, False
        return self.render(frame, faces, latent, s, {}), True
