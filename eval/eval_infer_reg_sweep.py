#!/usr/bin/env python3
"""
infer_reg 스윕: CSV별 SWEEP 실행 (파일 루프 tqdm).

  cd model
  python eval_infer_reg_sweep.py --run_dir results/... --checkpoint file.pth
  python eval_infer_reg_sweep.py --file GPT_5.4_Eng_merged.csv
  python eval_infer_reg_sweep.py --file /path/to/one.csv --max_samples 500
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# 스윕 목록: infer_structure_weight (w*structure + (1-w)*unit)
SWEEP = [1.0, 0.5, 0.8, 0.95, 0.3]

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(MODEL_DIR)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from eval import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    INCLUDE_CSV_PREFIXES,
    compute_metrics,
    load_articles_from_csv,
    load_detector,
    patch_parse_args,
    resolve_checkpoint,
    resolve_run_dir,
)


def resolve_csv_files(data_root: str, pattern: str, file_arg: str) -> list[str]:
    if file_arg.strip():
        farg = file_arg.strip()
        if os.path.isfile(farg):
            return [os.path.abspath(farg)]
        path = os.path.join(data_root, os.path.basename(farg))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV not found: {farg}")
        return [path]

    matched = sorted(glob.glob(os.path.join(data_root, pattern)))
    matched = [
        p for p in matched if not os.path.basename(p).startswith(INCLUDE_CSV_PREFIXES)
    ]
    if not matched:
        raise FileNotFoundError(f"No files matching {pattern!r} under {data_root}")
    return matched


def sweep_labels() -> list[str]:
    return [_weight_label(w) for w in SWEEP]


def _weight_label(w: float) -> str:
    if w >= 1.0:
        return "structure-only"
    if w <= 0.0:
        return "unit-only"
    return f"w={w}"


def predict_scores(
    detector,
    texts,
    structure_weight: float,
    fc_batch_size,
    embed_batch_size,
    pos_bit,
    desc: str = "predict",
):
    import parse as parse_mod

    parse_mod.args.infer_structure_weight = float(structure_weight)

    detector.eval()
    scores = []
    n = len(texts)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    n_batches = max(1, (n + fc_batch_size - 1) // fc_batch_size)
    with torch.inference_mode():
        for start in tqdm(
            range(0, n, fc_batch_size),
            desc=desc,
            leave=False,
            total=n_batches,
        ):
            chunk = texts[start : start + fc_batch_size]
            probs = detector.forward(chunk, embed_batch_size=embed_batch_size)
            scores.extend(probs[:, pos_bit].cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def _format_result_row(row: dict) -> str:
    return (
        f"  {row['infer_structure_weight']} (w={row['weight']:.2f}): "
        f"ROC-AUC={row['roc_auc']:.4f}  PR-AUC={row['pr_auc']:.4f}  "
        f"acc={row['acc']:.4f}  n={row['n_samples']}"
    )


def _print_compare_row(df: pd.DataFrame, metric: str, title: str) -> None:
    cols = sweep_labels()
    line = {
        lbl: df.loc[df["infer_structure_weight"] == lbl, metric].iloc[0] for lbl in cols
    }
    print(f"\n{title}: " + "  ".join(f"{c}={line[c]:.4f}" for c in cols))


def print_sweep_summary(df: pd.DataFrame, csv_name: str) -> None:
    print("\n" + "=" * 72)
    print(f"[요약] {csv_name}  |  SWEEP: {sweep_labels()}")
    print("=" * 72)
    _print_compare_row(df, "roc_auc", "ROC-AUC")
    _print_compare_row(df, "pr_auc", "PR-AUC")
    _print_compare_row(df, "acc", "acc@0.5")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep RoBERTa infer_structure_weight on ai_gen_news CSVs."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--run_dir",
        type=str,
        default="/home/dsail/2026-AINews-Detection/model/results/2026-05-23/RAID-news-18-25-07-344926",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="RAID-news_30_8_8.0_epoch10_20260523-182507.pth",
    )
    parser.add_argument("--DEVICE", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_batch_size", type=int, default=32)
    parser.add_argument("--pos_bit", type=int, default=1)
    parser.add_argument("--pattern", type=str, default="*_Eng_merged.csv")
    parser.add_argument(
        "--file",
        type=str,
        default="home/dsail/yjlee/police/ai_gen/dataset/ai_gen_news_dataset/all_final/GPT_5.4_nano_Eng_merged.csv",
        help="평가할 CSV 하나 (전체 경로 또는 data_root 아래 파일명). "
        "비우면 pattern에 맞는 전체 파일(eval.py 제외 목록 적용).",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    cli = parser.parse_args()

    patch_parse_args()

    run_dir = resolve_run_dir(cli.run_dir)
    ckpt_path = resolve_checkpoint(run_dir, cli.checkpoint)
    data_root = os.path.abspath(cli.data_root)
    output_dir = (
        os.path.abspath(cli.output_dir)
        if cli.output_dir
        else os.path.join(run_dir, "eval_infer_reg_sweep")
    )
    os.makedirs(output_dir, exist_ok=True)

    csv_files = resolve_csv_files(data_root, cli.pattern, cli.file)

    device = cli.DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device = "cpu"

    print(f"checkpoint: {ckpt_path}")
    print(f"CSV files: {len(csv_files)}  |  SWEEP: {SWEEP} ({sweep_labels()})")
    detector = load_detector(run_dir, device, cli.pos_bit, ckpt_path)

    all_rows = []
    for fi, csv_path in enumerate(tqdm(csv_files, desc="CSV files", unit="file"), 1):
        csv_name = os.path.basename(csv_path)
        texts, labels, _ = load_articles_from_csv(csv_path, max_samples=cli.max_samples)
        if not texts:
            tqdm.write(f"  skip {csv_name}: no valid articles")
            continue

        tqdm.write(f"\n>>> [{fi}/{len(csv_files)}] {csv_name}  ({len(texts)} articles)")
        file_rows = []
        for wi, w in enumerate(SWEEP, 1):
            label = _weight_label(w)
            tqdm.write(f"  [{wi}/{len(SWEEP)}] {label} (w={w}) predicting...")
            scores = predict_scores(
                detector,
                texts,
                w,
                cli.batch_size,
                cli.embed_batch_size,
                cli.pos_bit,
                desc=f"{csv_name} w={w}",
            )
            m = compute_metrics(labels, scores)
            row = {
                "infer_structure_weight": label,
                "weight": w,
                "file": csv_name,
                "csv_path": csv_path,
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "acc": m["acc"],
                "n_samples": m["n_samples"],
            }
            file_rows.append(row)
            all_rows.append(row)
            tqdm.write(_format_result_row(row))

        if len(SWEEP) > 1:
            file_df = pd.DataFrame(file_rows)
            print_sweep_summary(file_df, csv_name)

    if not all_rows:
        print("No results.")
        return

    df = pd.DataFrame(all_rows)

    if len(csv_files) == 1 and len(SWEEP) > 1:
        one_name = os.path.basename(csv_files[0])
        sub = (
            df[df["file"] == one_name]
            .set_index("infer_structure_weight")
            .reindex(sweep_labels())
        )
        print("\n=== final summary ===")
        print_sweep_summary(sub.reset_index(), one_name)

    print("\n=== all files ===")
    print_cols = [
        "file",
        "infer_structure_weight",
        "weight",
        "roc_auc",
        "pr_auc",
        "acc",
        "n_samples",
    ]
    print(df[print_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if len(SWEEP) == 1:
        print(f"\nMean ROC-AUC: {df['roc_auc'].mean():.4f}")
        print(f"Mean PR-AUC: {df['pr_auc'].mean():.4f}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_n{cli.max_samples}" if cli.max_samples > 0 else ""
    out_csv = os.path.join(output_dir, f"sweep{suffix}_{stamp}.csv")
    out_json = os.path.join(output_dir, f"sweep{suffix}_{stamp}.json")
    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sweep": SWEEP,
                "sweep_labels": sweep_labels(),
                "csv_files": csv_files,
                "results": all_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nWrote:\n  {out_csv}\n  {out_json}")


if __name__ == "__main__":
    main()
