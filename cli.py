"""Linha de comando. Ex.: python cli.py -s foto1.jpg foto2.jpg -t video.mp4 -o saida.mp4"""
import argparse

import cv2
from tqdm import tqdm

from faceswap.pipeline import FaceSwapPipeline, Settings
from faceswap.profile import analyse_video, compatibility_markdown


def main():
    ap = argparse.ArgumentParser(description="FaceSwap Studio (CLI)")
    ap.add_argument("-s", "--source", nargs="+", required=True, help="fotos de referência (mesma pessoa)")
    ap.add_argument("-t", "--target", required=True, help="vídeo alvo")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--swapper", default="hyperswap_1c_256",
                    choices=["hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256", "inswapper_128"])
    ap.add_argument("--resolution", type=int, choices=[128, 256, 512], default=512)
    ap.add_argument("--strength", type=float, default=0.2, help="força da identidade (0 a 0.5)")
    ap.add_argument("--color", type=float, default=0.85, help="cor da pele igual à do vídeo (0 a 1)")
    ap.add_argument("--grain", type=float, default=1.0)
    ap.add_argument("--temporal", type=float, default=0.5)
    ap.add_argument("--skin", type=float, default=0.0, help="tom de pele da referência (0 a 1)")
    ap.add_argument("--enhancer", default="none", choices=["none", "gfpgan_1.4", "gpen_bfr_512", "codeformer"])
    ap.add_argument("--blend", type=float, default=0.5)
    ap.add_argument("--smoothing", type=float, default=1.5)
    ap.add_argument("--crf", type=int, default=14)
    ap.add_argument("--mode", choices=["largest", "reference", "all"], default="largest")
    ap.add_argument("--reference", help="foto da pessoa do vídeo a ser trocada (modo reference)")
    ap.add_argument("--no-sharpness-match", action="store_true")
    ap.add_argument("--no-occlusion", action="store_true")
    ap.add_argument("--no-watermark", action="store_true")
    ap.add_argument("--analyse-only", action="store_true", help="só mostra a análise de compatibilidade")
    args = ap.parse_args()

    pipe = FaceSwapPipeline()
    sources = [cv2.imread(p) for p in args.source]
    profile = pipe.source_profile(sources, args.source)
    print(compatibility_markdown(profile, analyse_video(pipe, args.target)), "\n")
    if args.analyse_only:
        return

    s = Settings(swapper=args.swapper, swap_resolution=args.resolution, identity_strength=args.strength, color_match=args.color,
                 grain=args.grain, temporal=args.temporal, skin_tone=args.skin, enhancer=args.enhancer,
                 enhancer_blend=args.blend, smoothing=args.smoothing, crf=args.crf, mode=args.mode,
                 match_sharpness=not args.no_sharpness_match, occlusion=not args.no_occlusion,
                 watermark=not args.no_watermark)
    ref = cv2.imread(args.reference) if args.reference else None
    bar = tqdm(total=100, unit="%")

    def progress(p, desc=""):
        bar.set_description(desc)
        bar.n = int(p * 100)
        bar.refresh()

    out, stats = pipe.process_video(sources, args.target, args.output, s, ref, progress)
    bar.close()
    print(f"Salvo em {out}")
    if stats["fidelity_mean"] is not None:
        print(f"Fidelidade à referência: média {stats['fidelity_mean']:.2f}, mínima {stats['fidelity_min']:.2f}")


if __name__ == "__main__":
    main()
