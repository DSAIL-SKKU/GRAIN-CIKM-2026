#!/usr/bin/env python3
"""
newsgpt_ai_news_human.csv (또는 동일 스키마 CSV) 평가 스크립트.

CSV 스키마:
  - body : 기사 본문 (human/AI 모두 이 컬럼에 저장)
  - label      : "ai" 또는 "human"

Example:
  cd /home/dsail/2026-AINews-Detection/model
  python eval_newsgpt.py \
    --csv /home/dsail/2026-AINews-Detection/notebook/newsgpt_ai_news_human.csv \
    --run_dir results/2026-06-01/RAID-news-22-28-26-322680 \
    --DEVICE cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(MODEL_DIR)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from eval import (
    build_prediction_rows,
    compute_metrics,
    flush_eval_results,
    load_detector,
    patch_parse_args,
    predict_scores_fast,
    resolve_checkpoint,
    resolve_run_dir,
    setup_cuda_fast,
)


def load_articles_from_labeled_csv(
    csv_path: str, max_samples: int = 0
) -> tuple[list[str], list[int], list[dict]]:
    """
    body + label("ai"/"human") 컬럼 CSV 로더.
    label: "ai" → 1, "human" → 0
    """
    df = pd.read_csv(csv_path)
    # df = df[df["source"] == "kminb"]  # kminb, incheontoday, chosun, newspim

    if "body" not in df.columns:
        raise ValueError(f"'body' column not found in {csv_path}")
    if "label" not in df.columns:
        raise ValueError(f"'label' column not found in {csv_path}")

    texts, labels, meta = [], [], []
    for row_idx, row in df.iterrows():
        if max_samples > 0 and len(texts) >= max_samples:
            break

        text = "" if pd.isna(row["body"]) else str(row["body"]).strip()
        if not text:
            continue

        raw_label = str(row["label"]).strip().lower()
        if raw_label == "ai":
            label_int = 1
        elif raw_label == "human":
            label_int = 0
        else:
            continue  # 알 수 없는 라벨 건너뜀

        texts.append(text)
        labels.append(label_int)
        meta.append(
            {
                "csv_row": int(row_idx),
                "label_name": raw_label,
                "source": str(row.get("source", "")) if "source" in df.columns else "",
            }
        )

    return texts, labels, meta


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RoBERTa on a labeled CSV (body + label columns)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="/home/dsail/2026-AINews-Detection/notebook/real-world/Chosun/chosun_ai_news_human.csv",
        help="평가할 CSV 파일 경로.",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="train.py save dir. 비우면 results/ 최신 .pth 폴더.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help=".pth filename under run_dir. 비우면 최신 파일.",
    )
    parser.add_argument("--DEVICE", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_batch_size", type=int, default=128)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--infer_structure_weight",
        type=float,
        default=0.7,
        help="추론 logit 결합: w*structure+(1-w)*unit_mean. 1=structure만.",
    )
    parser.add_argument("--pos_bit", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="최대 샘플 수. 0이면 전체.",
    )
    cli = parser.parse_args()

    patch_parse_args(
        infer_structure_weight=cli.infer_structure_weight,
        no_amp=cli.no_amp,
    )
    print(
        f"infer_structure_weight={cli.infer_structure_weight} "
        f"(logits = w*structure + (1-w)*unit_mean; 1=structure only)"
    )

    run_dir = resolve_run_dir(cli.run_dir)
    ckpt_path = resolve_checkpoint(run_dir, cli.checkpoint)
    ckpt_name = os.path.basename(ckpt_path)

    csv_path = os.path.abspath(cli.csv)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir = (
        os.path.abspath(cli.output_dir)
        if cli.output_dir
        else os.path.join(run_dir, "eval_newsgpt")
    )
    os.makedirs(output_dir, exist_ok=True)

    device = cli.DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device = "cpu"

    print(f"Loading RoBERTa from {ckpt_path}")
    detector = load_detector(run_dir, device, cli.pos_bit, ckpt_path)
    setup_cuda_fast(device)

    # ── 데이터 로드 ───────────────────────────────────────────────
    csv_name = os.path.basename(csv_path)
    print(f"\n=== {csv_name} ===")
    texts, labels, meta = load_articles_from_labeled_csv(
        csv_path, max_samples=cli.max_samples
    )
    if not texts:
        print("  skip: no valid articles")
        return

    ai_cnt = sum(1 for l in labels if l == 1)
    human_cnt = sum(1 for l in labels if l == 0)
    print(f"  loaded: {len(texts)} samples (human={human_cnt}, ai={ai_cnt})")
    if cli.max_samples > 0:
        print(f"  (capped at {cli.max_samples})")

    # ── 예측 및 지표 계산 ─────────────────────────────────────────
    scores = predict_scores_fast(
        detector, texts, cli.batch_size, cli.embed_batch_size, cli.pos_bit
    )

    # ── DEBUG: score 방향 확인 ─────────────────────────
    labels_np = np.asarray(labels)
    scores_np = np.asarray(scores)

    print("\n[DEBUG score check]")
    print("pos_bit:", cli.pos_bit)
    print("score min:", scores_np.min())
    print("score max:", scores_np.max())
    print("score mean:", scores_np.mean())
    print("score median:", np.median(scores_np))
    print("score >= 0.5 ratio:", np.mean(scores_np >= 0.5))

    print("human mean score:", scores_np[labels_np == 0].mean())
    print("ai mean score   :", scores_np[labels_np == 1].mean())
    print(
        "human quantile:",
        np.quantile(scores_np[labels_np == 0], [0, 0.25, 0.5, 0.75, 1]),
    )
    print(
        "ai quantile   :",
        np.quantile(scores_np[labels_np == 1], [0, 0.25, 0.5, 0.75, 1]),
    )
    # ────────────────────────────────────────────────

    m = compute_metrics(labels, scores)
    pred_rows = build_prediction_rows(labels, scores, meta, m["best_threshold_youden"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_n{cli.max_samples}" if cli.max_samples > 0 else ""
    preds_dir = os.path.join(output_dir, f"predictions{suffix}_{stamp}")
    os.makedirs(preds_dir, exist_ok=True)

    for row in pred_rows:
        row["file"] = csv_name
        row["csv_path"] = csv_path

    stem = os.path.splitext(csv_name)[0]
    pred_out = os.path.join(preds_dir, f"{stem}_preds.csv")
    pd.DataFrame(pred_rows).to_csv(pred_out, index=False)

    m.update(
        {
            "file": csv_name,
            "csv_path": csv_path,
            "run_dir": run_dir,
            "checkpoint": ckpt_name,
            "infer_structure_weight": cli.infer_structure_weight,
            "max_samples_cap": cli.max_samples,
            "predictions_csv": pred_out,
        }
    )

    eval_config = {
        "csv_path": csv_path,
        "run_dir": run_dir,
        "checkpoint": ckpt_name,
        "checkpoint_path": ckpt_path,
        "DEVICE": device,
        "batch_size": cli.batch_size,
        "embed_batch_size": cli.embed_batch_size,
        "infer_structure_weight": cli.infer_structure_weight,
        "pos_bit": cli.pos_bit,
        "max_samples": cli.max_samples,
        "timestamp": stamp,
        "output_dir": output_dir,
    }

    flush_eval_results(
        output_dir, stamp, suffix, preds_dir, eval_config, [m], pred_rows
    )

    # ── 결과 출력 ─────────────────────────────────────────────────
    print(
        f"\n  n={m['n_samples']} (human={m['n_human']}, ai={m['n_ai']})\n"
        f"  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}\n"
        f"  tpr@fpr0.005={m['tpr_at_fpr_0.005']:.4f} "
        f"tpr@fpr0.01={m['tpr_at_fpr_0.01']:.4f} "
        f"tpr@fpr0.03={m['tpr_at_fpr_0.03']:.4f}\n"
        f"  acc@0.5={m['acc']:.4f}  f1@0.5={m['f1']:.4f}\n"
        f"  acc@best={m['acc_at_best_thresh']:.4f}  "
        f"f1@best={m['f1_at_best_thresh']:.4f}  "
        f"(thr={m['best_threshold_youden']:.4f})\n"
        f"  output: {output_dir}"
    )


if __name__ == "__main__":
    main()
