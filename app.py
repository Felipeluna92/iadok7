"""Interface web (Gradio) do FaceSwap Studio. Rode: python app.py"""
import time
from pathlib import Path

import cv2
import gradio as gr

from faceswap import models
from faceswap.pipeline import FaceSwapPipeline, Settings
from faceswap.video_io import probe

OUTPUTS = Path(__file__).resolve().parent / "outputs"
pipeline = FaceSwapPipeline()

CONSENT = ("Confirmo que tenho autorização das pessoas cujos rostos aparecem na foto e no vídeo, "
           "e que não vou usar o resultado para enganar, difamar ou criar conteúdo íntimo sem consentimento.")


def _load_sources(files):
    if not files:
        raise gr.Error("Envie pelo menos uma foto do rosto de origem.")
    imgs = [cv2.imread(str(f)) for f in files]
    return [i for i in imgs if i is not None]


def _settings(mode, ref_thr, boost, enhancer, blend, fidelity, smoothing, max_gap, det_thr,
              mask_blur, occlusion, pad_top, pad_bottom, crf, watermark, keep_audio):
    return Settings(mode=mode, reference_threshold=ref_thr, pixel_boost=int(boost), enhancer=enhancer,
                    enhancer_blend=blend, codeformer_fidelity=fidelity, smoothing=smoothing,
                    max_gap=int(max_gap), detect_threshold=det_thr, mask_blur=mask_blur, occlusion=occlusion,
                    mask_padding=(pad_top, 0, pad_bottom, 0), crf=int(crf), watermark=watermark,
                    keep_audio=keep_audio)


def _reference(img):
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def on_video(video):
    if not video:
        return gr.Slider(maximum=10, value=0)
    info = probe(video)
    dur = max(info["frames"] / info["fps"], 0.1)
    return gr.Slider(maximum=round(dur, 2), value=0)


def run_preview(files, video, t, ref_img, consent, *opts):
    if not consent:
        raise gr.Error("Marque a confirmação de autorização para continuar.")
    if not video:
        raise gr.Error("Envie o vídeo alvo.")
    try:
        frame, ok = pipeline.preview(_load_sources(files), video, t, _settings(*opts), _reference(ref_img))
    except ValueError as e:
        raise gr.Error(str(e))
    msg = "Prévia gerada." if ok else "Nenhum rosto correspondente neste momento do vídeo."
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), msg


def run_video(files, video, ref_img, consent, *opts, progress=gr.Progress()):
    if not consent:
        raise gr.Error("Marque a confirmação de autorização para continuar.")
    if not video:
        raise gr.Error("Envie o vídeo alvo.")
    out = OUTPUTS / f"faceswap_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    start = time.time()
    try:
        result = pipeline.process_video(_load_sources(files), video, str(out), _settings(*opts),
                                        _reference(ref_img), progress=lambda p, d="": progress(p, desc=d))
    except ValueError as e:
        raise gr.Error(str(e))
    return str(result), f"Pronto em {time.time() - start:.0f}s → `{result}`"


with gr.Blocks(title="FaceSwap Studio") as demo:
    gr.Markdown(f"# FaceSwap Studio\nDispositivo: **{models.active_device()}**")
    with gr.Row():
        with gr.Column(scale=1):
            sources = gr.File(label="Rosto de origem (1 a 10 fotos da MESMA pessoa: mais fotos = mais fiel)",
                              file_count="multiple", file_types=["image"], type="filepath")
            video = gr.Video(label="Vídeo alvo", sources=["upload"])

            with gr.Accordion("Qual rosto trocar", open=True):
                mode = gr.Radio([("Rosto principal (maior)", "largest"),
                                 ("Pessoa da foto de referência", "reference"),
                                 ("Todos os rostos", "all")], value="largest", label="Modo")
                ref_img = gr.Image(label="Foto de referência (só no modo 'pessoa da referência')", type="numpy")
                ref_thr = gr.Slider(0.15, 0.7, 0.35, step=0.01, label="Rigor da referência")

            with gr.Accordion("Qualidade", open=True):
                boost = gr.Radio([("Rápido (128px)", 1), ("Alta (256px)", 2), ("Máxima (512px)", 4)],
                                 value=2, label="Resolução da troca")
                enhancer = gr.Dropdown([("Nenhum", "none"), ("GFPGAN 1.4 (natural)", "gfpgan_1.4"),
                                        ("GPEN 512 (mais nítido)", "gpen_bfr_512"),
                                        ("CodeFormer (robusto)", "codeformer")],
                                       value="gfpgan_1.4", label="Melhorador de rosto")
                blend = gr.Slider(0, 1, 0.8, step=0.05, label="Intensidade do melhorador")
                fidelity = gr.Slider(0, 1, 0.7, step=0.05, label="Fidelidade CodeFormer")
                crf = gr.Slider(10, 28, 16, step=1, label="Compressão do vídeo (menor = melhor)")

            with gr.Accordion("Movimento e máscara", open=False):
                smoothing = gr.Slider(0, 5, 1.5, step=0.1, label="Suavização do movimento")
                max_gap = gr.Slider(0, 30, 8, step=1, label="Preencher falhas de detecção (frames)")
                det_thr = gr.Slider(0.2, 0.9, 0.5, step=0.05, label="Sensibilidade da detecção")
                mask_blur = gr.Slider(0, 1, 0.3, step=0.05, label="Suavidade da borda")
                occlusion = gr.Checkbox(True, label="Respeitar oclusões (mãos, cabelo, objetos na frente)")
                pad_top = gr.Slider(0, 40, 0, step=1, label="Recuo superior da máscara (%)")
                pad_bottom = gr.Slider(0, 40, 0, step=1, label="Recuo inferior da máscara (%)")

            watermark = gr.Checkbox(True, label="Marcar o vídeo como 'Gerado por IA'")
            keep_audio = gr.Checkbox(True, label="Manter áudio original")
            consent = gr.Checkbox(False, label=CONSENT)

        with gr.Column(scale=1):
            t = gr.Slider(0, 10, 0, step=0.05, label="Momento da prévia (segundos)")
            preview_btn = gr.Button("Gerar prévia do frame", variant="secondary")
            preview_img = gr.Image(label="Prévia", interactive=False)
            run_btn = gr.Button("Processar vídeo completo", variant="primary")
            output = gr.Video(label="Resultado", interactive=False)
            status = gr.Markdown()

    opts = [mode, ref_thr, boost, enhancer, blend, fidelity, smoothing, max_gap, det_thr,
            mask_blur, occlusion, pad_top, pad_bottom, crf, watermark, keep_audio]
    video.change(on_video, video, t)
    preview_btn.click(run_preview, [sources, video, t, ref_img, consent, *opts], [preview_img, status])
    run_btn.click(run_video, [sources, video, ref_img, consent, *opts], [output, status])


if __name__ == "__main__":
    OUTPUTS.mkdir(exist_ok=True)
    demo.queue(default_concurrency_limit=1).launch(inbrowser=True, allowed_paths=[str(OUTPUTS)])
