"""Linha de comando. Ex.: python cli.py -s rosto.jpg -t video.mp4 -o saida.mp4 --boost 4"""
import argparse

import cv2
from tqdm import tqdm

from faceswap.pipeline import FaceSwapPipeline, Settings


def main():
    ap = argparse.ArgumentParser(description="FaceSwap Studio (CLI)")
    ap.add_argument("-s", "--source", nargs="+", required=True, help="foto(s) do rosto de origem")
    ap.add_argument("-t", "--target", required=True, help="vídeo alvo")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--mode", choices=["largest", "reference", "all"], default="largest")
    ap.add_argument("--reference", help="foto da pessoa a ser trocada (modo reference)")
    ap.add_argument("--boost", type=int, choices=[1, 2, 4], default=2)
    ap.add_argument("--enhancer", default="gfpgan_1.4", choices=["none", "gfpgan_1.4", "gpen_bfr_512", "codeformer"])
    ap.add_argument("--blend", type=float, default=0.8)
    ap.add_argument("--smoothing", type=float, default=1.5)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--no-occlusion", action="store_true")
    ap.add_argument("--no-watermark", action="store_true")
    args = ap.parse_args()

    s = Settings(mode=args.mode, pixel_boost=args.boost, enhancer=args.enhancer, enhancer_blend=args.blend,
                 smoothing=args.smoothing, crf=args.crf, occlusion=not args.no_occlusion,
                 watermark=not args.no_watermark)
    ref = cv2.imread(args.reference) if args.reference else None
    bar = tqdm(total=100, unit="%")

    def progress(p, desc=""):
        bar.set_description(desc)
        bar.n = int(p * 100)
        bar.refresh()

    out = FaceSwapPipeline().process_video([cv2.imread(p) for p in args.source], args.target,
                                           args.output, s, ref, progress)
    bar.close()
    print(f"Salvo em {out}")


if __name__ == "__main__":
    main()
