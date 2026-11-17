"""Interface web (Gradio) do FaceSwap Studio. Rode: python app.py"""
import time
from pathlib import Path

import cv2
import gradio as gr

from faceswap import models
from faceswap.pipeline import FaceSwapPipeline, Settings
from faceswap.profile import analyse_video, compatibility_markdown
from faceswap.video_io import probe

OUTPUTS = Path(__file__).resolve().parent / "outputs"
pipeline = FaceSwapPipeline()

CONSENT = ("Confirmo que tenho autorização das pessoas cujos rostos aparecem nas fotos e no vídeo, "
           "e que não vou usar o resultado para enganar, difamar ou criar conteúdo íntimo sem consentimento.")


def _load_sources(files):
    if not files:
        raise gr.Error("Envie pelo menos uma foto de referência.")
    imgs = [(cv2.imread(str(f)), Path(str(f)).name) for f in files]
    imgs = [(i, n) for i, n in imgs if i is not None]
    return [i for i, _ in imgs], [n for _, n in imgs]


def _settings(swapper, resolution, strength, sharp, grain, temporal, skin, enhancer, blend, fidelity,
              smoothing, max_gap, det_thr, mask_blur, occlusion, model_mask, pad_top, pad_bottom,
              mode, ref_thr, crf, watermark, keep_audio):
    return Settings(swapper=swapper, swap_resolution=int(resolution), identity_strength=strength,
                    match_sharpness=sharp, grain=grain, temporal=temporal, skin_tone=skin, enhancer=enhancer,
                    enhancer_blend=blend, codeformer_fidelity=fidelity, smoothing=smoothing, max_gap=int(max_gap),
                    detect_threshold=det_thr, mask_blur=mask_blur, occlusion=occlusion, model_mask=model_mask,
                    mask_padding=(pad_top, 0, pad_bottom, 0), mode=mode, reference_threshold=ref_thr,
                    crf=int(crf), watermark=watermark, keep_audio=keep_audio)


def _reference(img):
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _check(consent, video):
    if not consent:
        raise gr.Error("Marque a confirmação de autorização para continuar.")
    if not video:
        raise gr.Error("Envie o vídeo alvo.")


def on_video(video):
    if not video:
        return gr.Slider(maximum=10, value=0)
    info = probe(video)
    return gr.Slider(maximum=round(max(info["frames"] / info["fps"], 0.1), 2), value=0)


def run_analysis(files, video, progress=gr.Progress()):
    if not video:
        raise gr.Error("Envie o vídeo alvo.")
    imgs, names = _load_sources(files)
    progress(0.1, desc="Analisando fotos de referência")
    try:
        src = pipeline.source_profile(imgs, names)
    except ValueError as e:
        raise gr.Error(str(e))
    progress(0.5, desc="Analisando o vídeo")
    return compatibility_markdown(src, analyse_video(pipeline, video))


def run_preview(files, video, t, ref_img, consent, *opts):
    _check(consent, video)
    imgs, names = _load_sources(files)
    try:
        pipeline.source_profile(imgs, names)
        original, result, score = pipeline.preview(imgs, video, t, _settings(*opts), _reference(ref_img))
    except ValueError as e:
        raise gr.Error(str(e))
    if score is None:
        msg = "Nenhum rosto correspondente neste momento do vídeo."
    else:
        msg = f"Fidelidade à referência neste frame: **{score:.2f}** (acima de 0.65 = muito parecido)"
    rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return (rgb(original), rgb(result)), msg


def run_video(files, video, ref_img, consent, *opts, progress=gr.Progress()):
    _check(consent, video)
    imgs, names = _load_sources(files)
    out = OUTPUTS / f"faceswap_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    start = time.time()
    try:
        pipeline.source_profile(imgs, names)
        result, stats = pipeline.process_video(imgs, video, str(out), _settings(*opts), _reference(ref_img),
                                               progress=lambda p, d="": progress(p, desc=d))
    except ValueError as e:
        raise gr.Error(str(e))
    fid = (f" · fidelidade média **{stats['fidelity_mean']:.2f}** (mín. {stats['fidelity_min']:.2f})"
           if stats["fidelity_mean"] is not None else "")
    return str(result), (f"Pronto em {time.time() - start:.0f}s · rosto trocado em "
                         f"{stats['frames_swapped']}/{stats['frames']} frames{fid}\n\n`{result}`")


with gr.Blocks(title="FaceSwap Studio") as demo:
    gr.Markdown(f"# FaceSwap Studio\nDispositivo: **{models.active_device()}** · "
                "Fluxo: envie as fotos e o vídeo → **Analisar** → ajuste com a **Prévia** → **Processar**")
    with gr.Row():
        with gr.Column(scale=1):
            sources = gr.File(label="Fotos de referência: 3 a 10 fotos nítidas da MESMA pessoa (ângulos e "
                                    "expressões variados)", file_count="multiple", file_types=["image"],
                              type="filepath")
            video = gr.Video(label="Vídeo alvo", sources=["upload"])
            analyse_btn = gr.Button("1. Analisar referência e vídeo")
            report = gr.Markdown()

            with gr.Accordion("Troca de rosto", open=True):
                swapper = gr.Dropdown([("HyperSwap 1C (recomendado)", "hyperswap_1c_256"),
                                       ("HyperSwap 1B", "hyperswap_1b_256"), ("HyperSwap 1A", "hyperswap_1a_256"),
                                       ("InSwapper 128 (antigo)", "inswapper_128")],
                                      value="hyperswap_1c_256", label="Modelo")
                resolution = gr.Radio([("256 px (rápido)", 256), ("512 px (máxima)", 512)], value=512,
                                      label="Resolução da troca")
                strength = gr.Slider(0, 0.5, 0.2, step=0.05, label="Força da identidade (remove traços do rosto original)")

            with gr.Accordion("Realismo (integração com a câmera)", open=True):
                sharp = gr.Checkbox(True, label="Igualar nitidez à do vídeo")
                grain = gr.Slider(0, 1.5, 1.0, step=0.05, label="Granulação igual à do vídeo")
                temporal = gr.Slider(0, 0.9, 0.5, step=0.05, label="Estabilidade entre frames")
                skin = gr.Slider(0, 1, 0, step=0.05, label="Tom de pele da referência (veja a análise)")
                enhancer = gr.Dropdown([("Nenhum (mais natural)", "none"), ("GFPGAN 1.4", "gfpgan_1.4"),
                                        ("GPEN 512", "gpen_bfr_512"), ("CodeFormer", "codeformer")],
                                       value="none", label="Melhorador de rosto (para vídeos de baixa qualidade)")
                blend = gr.Slider(0, 1, 0.5, step=0.05, label="Intensidade do melhorador")
                fidelity = gr.Slider(0, 1, 0.7, step=0.05, label="Fidelidade CodeFormer")

            with gr.Accordion("Movimento e máscara", open=False):
                smoothing = gr.Slider(0, 5, 1.5, step=0.1, label="Suavização do movimento")
                max_gap = gr.Slider(0, 30, 8, step=1, label="Preencher falhas de detecção (frames)")
                det_thr = gr.Slider(0.2, 0.9, 0.5, step=0.05, label="Sensibilidade da detecção")
                mask_blur = gr.Slider(0, 1, 0.3, step=0.05, label="Suavidade da borda")
                occlusion = gr.Checkbox(True, label="Respeitar oclusões (mãos, cabelo, objetos na frente)")
                model_mask = gr.Checkbox(False, label="Usar também a máscara do HyperSwap")
                pad_top = gr.Slider(0, 40, 0, step=1, label="Recuo superior da máscara (%)")
                pad_bottom = gr.Slider(0, 40, 0, step=1, label="Recuo inferior da máscara (%)")

            with gr.Accordion("Quem trocar (vídeos com várias pessoas)", open=False):
                mode = gr.Radio([("Rosto principal (maior)", "largest"),
                                 ("Pessoa da foto abaixo", "reference"), ("Todos os rostos", "all")],
                                value="largest", label="Modo")
                ref_img = gr.Image(label="Foto da pessoa do vídeo a ser trocada", type="numpy")
                ref_thr = gr.Slider(0.15, 0.7, 0.35, step=0.01, label="Rigor")

            with gr.Accordion("Saída", open=False):
                crf = gr.Slider(10, 28, 14, step=1, label="Compressão do vídeo (menor = melhor)")
                keep_audio = gr.Checkbox(True, label="Manter áudio original")
            watermark = gr.Checkbox(True, label="Marcar o vídeo como 'Gerado por IA'")
            consent = gr.Checkbox(False, label=CONSENT)

        with gr.Column(scale=1):
            t = gr.Slider(0, 10, 0, step=0.05, label="Momento da prévia (segundos)")
            preview_btn = gr.Button("2. Gerar prévia (antes / depois)")
            preview_img = gr.ImageSlider(label="Antes / depois — arraste a barra", interactive=False)
            run_btn = gr.Button("3. Processar vídeo completo", variant="primary")
            output = gr.Video(label="Resultado", interactive=False)
            status = gr.Markdown()

    opts = [swapper, resolution, strength, sharp, grain, temporal, skin, enhancer, blend, fidelity,
            smoothing, max_gap, det_thr, mask_blur, occlusion, model_mask, pad_top, pad_bottom,
            mode, ref_thr, crf, watermark, keep_audio]
    video.change(on_video, video, t)
    analyse_btn.click(run_analysis, [sources, video], report)
    preview_btn.click(run_preview, [sources, video, t, ref_img, consent, *opts], [preview_img, status])
    run_btn.click(run_video, [sources, video, ref_img, consent, *opts], [output, status])


if __name__ == "__main__":
    OUTPUTS.mkdir(exist_ok=True)
    demo.queue(default_concurrency_limit=1).launch(inbrowser=True, allowed_paths=[str(OUTPUTS)])
