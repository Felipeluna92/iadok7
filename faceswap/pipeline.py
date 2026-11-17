"""Pipeline completo: análise (detecção + rastreamento + suavização), troca e acabamento realista."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .analyser import FaceDetector, FaceRecognizer
from .enhancer import FaceEnhancer
from .geometry import ARCFACE_128, FFHQ_512, paste_back, warp_face
from .masker import FaceParser, box_mask
from .profile import AgeGender, SourceProfile, build_source_profile
from .realism import match_grain, match_sharpness, temporal_stabilize, transfer_skin_tone
from .swapper import FaceSwapper
from .tracking import Tracker, smooth_track
from .video_io import VideoWriter, probe, read_frames

FINISH = 512  # resolução do acabamento (alinhamento FFHQ)


@dataclass
class Settings:
    # troca
    swapper: str = "hyperswap_1c_256"
    swap_resolution: int = 512         # 256 ou 512 (128 só para inswapper)
    identity_strength: float = 0.2     # afasta a identidade do rosto original do vídeo
    # quem trocar
    mode: str = "largest"              # largest | reference | all
    reference_threshold: float = 0.35
    detect_threshold: float = 0.5
    # movimento
    smoothing: float = 1.5
    max_gap: int = 8
    # máscara
    mask_blur: float = 0.3
    occlusion: bool = True
    model_mask: bool = False           # máscara própria do HyperSwap
    mask_padding: tuple = (0, 0, 0, 0)
    mask_smoothing: float = 0.5
    # acabamento / realismo
    enhancer: str = "none"             # none | gfpgan_1.4 | gpen_bfr_512 | codeformer
    enhancer_blend: float = 0.5
    codeformer_fidelity: float = 0.7
    skin_tone: float = 0.0             # 0..1: aproxima o tom de pele da referência
    temporal: float = 0.5              # 0..0.9: estabilidade entre frames
    match_sharpness: bool = True
    grain: float = 1.0                 # 0..1.5: granulação igual à do vídeo
    # saída
    crf: int = 14
    preset: str = "slow"
    watermark: bool = True
    keep_audio: bool = True


def add_watermark(frame: np.ndarray, text: str = "Gerado por IA") -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max(h / 1080 * 0.8, 0.4)
    thick = max(int(scale * 2), 1)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    org = (w - tw - int(20 * scale), h - int(20 * scale))
    overlay = frame.copy()
    cv2.putText(overlay, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(overlay, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)


def _compose(m_outer: np.ndarray, m_inner: np.ndarray) -> np.ndarray:
    """Transformação do espaço do recorte `m_inner` para o espaço do recorte `m_outer`."""
    a = np.vstack([m_outer, [0, 0, 1]])
    b = np.vstack([cv2.invertAffineTransform(m_inner), [0, 0, 1]])
    return (a @ b)[:2].astype(np.float32)


class FaceSwapPipeline:
    def __init__(self):
        self._detector = self._recognizer = self._parser = self._agegender = None
        self._swappers, self._enhancers = {}, {}
        self._profile_cache = {}
        self.rng = np.random.default_rng()

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
    def parser(self):
        self._parser = self._parser or FaceParser()
        return self._parser

    @property
    def agegender(self):
        self._agegender = self._agegender or AgeGender()
        return self._agegender

    def swapper(self, name: str) -> FaceSwapper:
        if name not in self._swappers:
            self._swappers.clear()  # libera VRAM do modelo anterior (GPU de 6 GB)
            self._swappers[name] = FaceSwapper(name)
        return self._swappers[name]

    def enhancer(self, name: str, fidelity: float):
        if name not in self._enhancers:
            self._enhancers.clear()
            self._enhancers[name] = FaceEnhancer(name)
        self._enhancers[name].fidelity = fidelity
        return self._enhancers[name]

    # ---- identidade ----
    def largest_face(self, image: np.ndarray, threshold: float = 0.5):
        faces = self.detector.detect(image, threshold)
        if not faces:  # fotos recortadas muito justas: adiciona margem e tenta de novo
            pad = max(image.shape[:2]) // 2
            image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            faces = self.detector.detect(image, threshold * 0.8)
        if not faces:
            return None, image
        return max(faces, key=lambda f: f.area), image

    def largest_face_embedding(self, image: np.ndarray):
        face, image = self.largest_face(image)
        return None if face is None else self.recognizer.embed(image, face.kps)

    def source_profile(self, images: list, names: list = None) -> SourceProfile:
        key = tuple(hash(i.tobytes()) for i in images)
        if key not in self._profile_cache:
            self._profile_cache = {key: build_source_profile(self, images, names)}
        return self._profile_cache[key]

    # ---- passo 1: análise ----
    def analyse(self, video: str, s: Settings, reference=None, start=0, end=None, progress=None):
        """Retorna (plano {frame: [(track, kps)]}, {track: identidade média do rosto original})."""
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
            return {}, {}
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

        plan, embs = {}, {}
        for t in tracks:
            e = np.mean([f.embedding for f in t.frames.values()], axis=0)
            embs[t.id] = e / np.linalg.norm(e)
            for f, kps in smooth_track(t, s.smoothing, s.max_gap).items():
                if start <= f < (end if end is not None else f + 1):
                    plan.setdefault(f, []).append((t.id, kps))
        return plan, embs

    # ---- passo 2: renderização ----
    def render(self, frame: np.ndarray, idx: int, faces: list, src: SourceProfile, track_embs: dict,
               s: Settings, state: dict) -> np.ndarray:
        out = frame.copy()
        sw = self.swapper(s.swapper)
        boost = max(s.swap_resolution // sw.size, 1)
        res = sw.size * boost
        bmask = box_mask(res, s.mask_blur, s.mask_padding)
        enh = self.enhancer(s.enhancer, s.codeformer_fidelity) if s.enhancer not in (None, "none") else None

        for tid, kps in faces:
            st = state.setdefault(tid, {})
            if st.get("last") != idx - 1:      # trilha nova ou interrompida: zera o histórico
                st.clear()
            st["last"] = idx

            # 1) troca no espaço ArcFace
            crop, m = warp_face(frame, kps, ARCFACE_128, res)
            latent = sw.latent(src.embedding, track_embs.get(tid), s.identity_strength)
            swapped, model_mask = sw.swap(crop, latent, boost)
            mask = bmask
            if s.occlusion:
                region = self.parser.mask(crop, res)
                if "region" in st and s.mask_smoothing > 0:
                    region = s.mask_smoothing * st["region"] + (1 - s.mask_smoothing) * region
                st["region"] = region
                mask = np.minimum(mask, region)
            if s.model_mask and model_mask is not None:
                mask = np.minimum(mask, cv2.GaussianBlur(model_mask, (0, 0), res / 128))
            out = paste_back(out, swapped, mask, m)

            # 2) acabamento em 512 px (alinhamento FFHQ), comparando com o rosto original
            fin, m2 = warp_face(out, kps, FFHQ_512, FINISH)
            orig, _ = warp_face(frame, kps, FFHQ_512, FINISH)
            fin = fin.astype(np.float32)
            orig = orig.astype(np.float32)
            fmask = cv2.warpAffine(mask, _compose(m2, m), (FINISH, FINISH), flags=cv2.INTER_LINEAR, borderValue=0)
            if enh is not None:
                fin = fin * (1 - s.enhancer_blend) + enh.enhance(fin) * s.enhancer_blend
            if s.skin_tone > 0:
                fin = transfer_skin_tone(fin, fmask, src.skin_ab, s.skin_tone)
            fin = temporal_stabilize(fin, st.get("prev"), orig, st.get("prev_orig"), s.temporal)
            st["prev"], st["prev_orig"] = fin, orig
            if s.match_sharpness:
                fin = match_sharpness(fin, orig, fmask)
            out = paste_back(out, fin, fmask, m2)
            out = match_grain(out, frame, fmask, m2, self.rng, s.grain)

        for tid in [t for t in state if t not in {f[0] for f in faces}]:
            del state[tid]
        return add_watermark(out) if s.watermark else out

    def fidelity(self, frame: np.ndarray, kps: np.ndarray, src: SourceProfile) -> float:
        """Semelhança (ArcFace) entre o rosto renderizado e a referência."""
        return float(self.recognizer.embed(frame, kps) @ src.embedding)

    # ---- APIs de alto nível ----
    def process_video(self, sources: list, video: str, output: str, s: Settings,
                      reference_image=None, progress=None):
        progress = progress or (lambda p, d="": None)
        src = self.source_profile(sources)
        reference = self.largest_face_embedding(reference_image) if reference_image is not None else None
        if s.mode == "reference" and reference is None:
            raise ValueError("Nenhum rosto encontrado na foto de referência.")

        plan, embs = self.analyse(video, s, reference, progress=lambda p, d: progress(p * 0.2, d))
        if not plan:
            raise ValueError("Nenhum rosto correspondente foi encontrado no vídeo.")

        info = probe(video)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        writer = VideoWriter(output, info["width"], info["height"], info["fps"], s.crf, s.preset,
                             audio_from=video if s.keep_audio else None)
        state, total, scores = {}, max(info["frames"], 1), []
        try:
            for idx, frame in read_frames(video):
                faces = plan.get(idx)
                if faces:
                    rendered = self.render(frame, idx, faces, src, embs, s, state)
                    if idx % 10 == 0:
                        scores.append(self.fidelity(rendered, faces[0][1], src))
                    frame = rendered
                else:
                    state.clear()
                    if s.watermark:
                        frame = add_watermark(frame)
                writer.write(frame)
                if idx % 2 == 0:
                    progress(0.2 + 0.8 * min(idx / total, 1.0), f"Renderizando frame {idx}/{total}")
        finally:
            result = writer.close()
        progress(1.0, "Concluído")
        stats = {"fidelity_mean": float(np.mean(scores)) if scores else None,
                 "fidelity_min": float(np.min(scores)) if scores else None,
                 "frames_swapped": len(plan), "frames": total}
        return result, stats

    def preview(self, sources: list, video: str, time_sec: float, s: Settings, reference_image=None):
        """Processa um frame com ±15 frames de contexto (suavização e estabilidade temporal reais)."""
        src = self.source_profile(sources)
        reference = self.largest_face_embedding(reference_image) if reference_image is not None else None
        info = probe(video)
        target = min(int(time_sec * info["fps"]), max(info["frames"] - 1, 0))
        start, end = max(target - 15, 0), target + 16
        plan, embs = self.analyse(video, s, reference, start, end)
        state, result, original = {}, None, None
        for idx, frame in read_frames(video, max(target - 5, start), target + 1):
            faces = plan.get(idx)
            if idx == target:
                original = frame
            result = self.render(frame, idx, faces, src, embs, s, state) if faces else frame
        faces = plan.get(target)
        score = self.fidelity(result, faces[0][1], src) if faces else None
        return original, result, score
