#!/usr/bin/env python3
"""
단일 CSV 파일의 지정 컬럼으로 RoBERTa 추론.

Example:
  cd /home/dsail/2026-AINews-Detection/model
  python infer.py --csv ../notebook/newsgpt_ai_news.csv --DEVICE cuda:0
  python infer.py --csv data.csv --text_col body --label_col label
  python infer.py --csv data.csv --run_dir results/2026-05-17/MultiDomainXSum-... --DEVICE cuda:0
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(MODEL_DIR)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

RESULTS_ROOT = os.path.join(MODEL_DIR, "results")


# ---------------------------------------------------------------------------
# Setup helpers (eval.py와 동일)
# ---------------------------------------------------------------------------


def patch_parse_args(infer_structure_weight=None, no_amp=False):
    import parse as parse_mod

    parse_mod.args.article_level = True
    parse_mod.args.article_max_units = 50
    parse_mod.args.reg = getattr(parse_mod.args, "reg", 8.0)
    parse_mod.args.iter = getattr(parse_mod.args, "iter", 0)
    parse_mod.args.no_amp = no_amp
    if infer_structure_weight is not None:
        parse_mod.args.infer_structure_weight = infer_structure_weight


def setup_cuda_fast(device: str) -> None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True


def resolve_run_dir(explicit: str) -> str:
    if explicit:
        path = os.path.abspath(explicit)
        if path.endswith(".pth") and os.path.isfile(path):
            return os.path.dirname(path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"--run_dir not found: {path}")
        return path

    if not os.path.isdir(RESULTS_ROOT):
        raise FileNotFoundError(
            f"No --run_dir given and {RESULTS_ROOT} does not exist. "
            "Train with train.py first or pass --run_dir."
        )

    best_mtime, best_dir = -1.0, None
    for root, _dirs, files in os.walk(RESULTS_ROOT):
        if any(f.endswith(".pth") for f in files):
            mtime = os.path.getmtime(root)
            if mtime > best_mtime:
                best_mtime, best_dir = mtime, root

    if best_dir is None:
        raise FileNotFoundError(f"No .pth checkpoints under {RESULTS_ROOT}")
    return best_dir


def resolve_checkpoint(run_dir: str, checkpoint_name: str) -> str:
    if checkpoint_name:
        path = os.path.join(run_dir, checkpoint_name)
        if os.path.isfile(path):
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    files = glob.glob(os.path.join(run_dir, "*.pth"))
    if not files:
        raise FileNotFoundError(f"No .pth files in {run_dir}")
    return max(files, key=os.path.getmtime)


def load_detector(run_dir, device, pos_bit, checkpoint_path):
    from methods.RoBERTa import RoBERTa

    detector = RoBERTa(device, run_dir, run_dir, pos_bit=pos_bit)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    detector.load_state_dict(state)
    detector.eval()
    return detector


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_texts_from_csv(csv_path: str, text_col: str, max_samples: int = 0):
    """
    Returns:
        texts  : list[str]
        df_sub : pd.DataFrame  (원본 행, 인덱스 유지)
    """
    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found. Available: {list(df.columns)}"
        )

    texts, idxs = [], []

    for i, row in df.iterrows():
        if max_samples > 0 and len(texts) >= max_samples:
            break

        raw = row[text_col]
        text = "" if pd.isna(raw) else str(raw).strip()
        if not text:
            continue

        texts.append(text)
        idxs.append(i)

    return texts, df.loc[idxs].reset_index(drop=False)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_scores(detector, texts, batch_size, embed_batch_size, pos_bit):
    detector.eval()
    n = len(texts)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    chunks = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, n, batch_size),
            desc="infer",
            leave=True,
            total=max(1, (n + batch_size - 1) // batch_size),
        ):
            chunk = texts[start : start + batch_size]
            probs = detector.forward(chunk, embed_batch_size=embed_batch_size)
            chunks.append(probs[:, pos_bit].detach())
    return torch.cat(chunks).cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="RoBERTa inference on a single CSV file."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="입력 CSV 파일 경로",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="ai_article",
        help="추론할 텍스트 컬럼명 (default: ai_article)",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="체크포인트 디렉터리. 비우면 results/ 최신.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help=".pth 파일명. 비우면 run_dir 내 최신.",
    )
    parser.add_argument("--DEVICE", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_batch_size", type=int, default=32)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--infer_structure_weight",
        type=float,
        default=1.0,
        help="w*structure + (1-w)*unit_mean. 1=structure only.",
    )
    parser.add_argument("--pos_bit", type=int, default=1)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="최대 기사 수. 0이면 전체.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="결과 CSV 저장 경로. 비우면 입력 파일명_pred.csv.",
    )
    cli = parser.parse_args()

    patch_parse_args(
        infer_structure_weight=cli.infer_structure_weight, no_amp=cli.no_amp
    )

    device = cli.DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device = "cpu"

    explicit = os.path.abspath(cli.run_dir) if cli.run_dir else ""
    if explicit.endswith(".pth") and os.path.isfile(explicit):
        run_dir = os.path.dirname(explicit)
        ckpt_path = explicit
    else:
        run_dir = resolve_run_dir(cli.run_dir)
        ckpt_path = resolve_checkpoint(run_dir, cli.checkpoint)
    print(f"Checkpoint : {ckpt_path}")
    print(f"Device     : {device}")

    detector = load_detector(run_dir, device, cli.pos_bit, ckpt_path)
    setup_cuda_fast(device)

    csv_path = os.path.abspath(cli.csv)
    print(f"Input CSV  : {csv_path}  (col='{cli.text_col}')")

    texts, df_sub = load_texts_from_csv(
        csv_path,
        text_col=cli.text_col,
        max_samples=cli.max_samples,
    )
    print(f"Loaded     : {len(texts)} articles")

    if len(texts) == 0:
        print("No valid texts found.")
        return

    scores = predict_scores(
        detector, texts, cli.batch_size, cli.embed_batch_size, cli.pos_bit
    )

    df_sub["ai_score"] = scores
    df_sub["pred"] = (scores >= 0.5).astype(int)
    df_sub["pred_label"] = df_sub["pred"].map({1: "ai", 0: "human"})

    out_path = (
        cli.output
        or os.path.splitext(csv_path)[0]
        + f"_pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    df_sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df_sub)} rows)")


if __name__ == "__main__":
    main()
