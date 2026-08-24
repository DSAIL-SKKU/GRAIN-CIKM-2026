#!/usr/bin/env python3
"""
Evaluate model/ RoBERTa (structure + fc) on multi_domains_xsum_{train,test}.json.

Dataset keys: text, label ("human"/"llm"), data_type, llm_type.

Example:
  cd /home/dsail/2026-AINews-Detection/model
  python eval_DetectRL.py --DEVICE cuda:0
  python eval_DetectRL.py --run_dir results/2026-05-31/RAID-news-20-58-38-342807 --DEVICE cuda:0
  python eval_DetectRL.py --run_dir results/... --checkpoint RAID-news_30_8_0.7_20260531-205838.pth
  python eval_DetectRL.py --max_samples 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
    load_detector,
    patch_parse_args,
    predict_scores_fast,
    resolve_checkpoint,
    resolve_run_dir,
    setup_cuda_fast,
)

DEFAULT_DATA_JSON = (
    "/home/dsail/2026-AINews-Detection/datasets/multi_domains_xsum_test.json"
)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_articles_from_json(
    json_path: str, max_samples: int = 0
) -> Tuple[List[str], List[int], List[Dict]]:
    """
    Load multi_domains_xsum_{train,test}.json.
    Returns (texts, labels, meta)  labels: 0=human, 1=llm.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    texts, labels, meta = [], [], []
    for i, rec in enumerate(records):
        if max_samples > 0 and len(texts) >= max_samples:
            break
        text = str(rec.get("text", "")).strip()
        if not text:
            continue
        raw_label = str(rec.get("label", "")).strip().lower()
        label = 0 if raw_label == "human" else 1
        texts.append(text)
        labels.append(label)
        meta.append(
            {
                "csv_row": i,
                "label_name": raw_label,
                "data_type": rec.get("data_type", ""),
                "llm_type": rec.get("llm_type", ""),
            }
        )
    return texts, labels, meta


def _mean_numeric_metrics(summary_df: pd.DataFrame) -> dict:
    skip = {"file", "json_path", "run_dir", "checkpoint", "predictions_csv"}
    return {
        col: float(summary_df[col].mean())
        for col in summary_df.columns
        if col not in skip and pd.api.types.is_numeric_dtype(summary_df[col])
    }


def flush_eval_results(
    output_dir, stamp, suffix, preds_dir, eval_config, rows, all_predictions
):
    summary_df = pd.DataFrame(rows)
    csv_out = os.path.join(output_dir, f"detectRL_eval{suffix}_{stamp}.csv")
    json_out = os.path.join(output_dir, f"detectRL_eval{suffix}_{stamp}.json")
    config_out = os.path.join(output_dir, f"eval_config{suffix}_{stamp}.json")
    preds_all_out = os.path.join(preds_dir, "all_predictions.csv")

    summary_df.to_csv(csv_out, index=False)
    if all_predictions:
        pd.DataFrame(all_predictions).to_csv(preds_all_out, index=False)

    payload = {
        "eval_config": eval_config,
        "n_files_done": len(rows),
        "mean_metrics": _mean_numeric_metrics(summary_df) if len(rows) else {},
        "per_file_metrics": rows,
        "predictions_dir": preds_dir,
        "all_predictions_csv": preds_all_out if all_predictions else "",
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(eval_config, f, indent=2, ensure_ascii=False)

    return csv_out, json_out, preds_all_out


def compute_metrics_safe(y_true, y_score) -> Optional[Dict]:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return None
    return compute_metrics(y_true, y_score)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model/ RoBERTa on multi_domains_xsum JSON."
    )
    parser.add_argument("--data_json", type=str, default=DEFAULT_DATA_JSON)
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="train.py save dir (results/날짜/...). 비우면 results/ 최신 .pth 폴더.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help=".pth filename under run_dir. 비우면 run_dir 안 최신 파일.",
    )
    parser.add_argument("--DEVICE", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128, help="fc 배치 (기사 수)")
    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=128,
        help="structure RoBERTa 임베딩 배치 (OOM 시 줄이기)",
    )
    parser.add_argument(
        "--no_amp", action="store_true", help="structure 임베딩 AMP(fp16) 끄기"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="PyTorch 2.x: structure RoBERTa에 torch.compile (첫 배치 느림)",
    )
    parser.add_argument(
        "--infer_structure_weight",
        type=float,
        default=1.0,
        help="추론 logit 결합: w*structure+(1-w)*unit_mean. 1=structure만.",
    )
    parser.add_argument("--pos_bit", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="최대 샘플 수(human+ai 합). 0이면 전체",
    )
    cli = parser.parse_args()

    patch_parse_args(
        infer_structure_weight=cli.infer_structure_weight, no_amp=cli.no_amp
    )
    print(
        f"infer_structure_weight={cli.infer_structure_weight} "
        f"(logits = w*structure + (1-w)*unit_mean; 1=structure only)"
    )

    run_dir = resolve_run_dir(cli.run_dir)
    ckpt_path = resolve_checkpoint(run_dir, cli.checkpoint)
    ckpt_name = os.path.basename(ckpt_path)

    data_json = os.path.abspath(cli.data_json)
    if not os.path.isfile(data_json):
        raise FileNotFoundError(f"--data_json not found: {data_json}")

    output_dir = (
        os.path.abspath(cli.output_dir)
        if cli.output_dir
        else os.path.join(run_dir, "eval_detectRL")
    )
    os.makedirs(output_dir, exist_ok=True)

    device = cli.DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device = "cpu"

    print(f"Loading RoBERTa from {ckpt_path}")
    if cli.max_samples > 0:
        print(f"Quick mode: up to {cli.max_samples} samples")
    detector = load_detector(run_dir, device, cli.pos_bit, ckpt_path)
    setup_cuda_fast(device)

    if cli.compile and hasattr(torch, "compile"):
        from methods import structure as structure_mod

        if structure_mod.model is not None:
            structure_mod.model = torch.compile(structure_mod.model)
            print("torch.compile enabled on structure embedder")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_n{cli.max_samples}" if cli.max_samples > 0 else ""
    preds_dir = os.path.join(output_dir, f"predictions{suffix}_{stamp}")
    os.makedirs(preds_dir, exist_ok=True)

    eval_config = {
        "data_json": data_json,
        "run_dir": run_dir,
        "checkpoint": ckpt_name,
        "checkpoint_path": ckpt_path,
        "DEVICE": device,
        "batch_size": cli.batch_size,
        "embed_batch_size": cli.embed_batch_size,
        "infer_structure_weight": cli.infer_structure_weight,
        "pos_bit": cli.pos_bit,
        "max_samples": cli.max_samples,
        "no_amp": cli.no_amp,
        "compile": cli.compile,
        "timestamp": stamp,
        "output_dir": output_dir,
    }

    print(f"\n=== {os.path.basename(data_json)} ===")
    texts, labels, meta = load_articles_from_json(
        data_json, max_samples=cli.max_samples
    )
    if not texts:
        print("  skip: no valid samples")
        return

    scores = predict_scores_fast(
        detector, texts, cli.batch_size, cli.embed_batch_size, cli.pos_bit
    )
    m = compute_metrics(labels, scores)
    pred_rows = build_prediction_rows(labels, scores, meta, m["best_threshold_youden"])

    for row, info in zip(pred_rows, meta):
        row["file"] = os.path.basename(data_json)
        row["json_path"] = data_json
        row["data_type"] = info.get("data_type", "")
        row["llm_type"] = info.get("llm_type", "")

    pred_out = os.path.join(preds_dir, "all_predictions.csv")
    pd.DataFrame(pred_rows).to_csv(pred_out, index=False)

    m.update(
        {
            "file": os.path.basename(data_json),
            "json_path": data_json,
            "run_dir": run_dir,
            "checkpoint": ckpt_name,
            "infer_structure_weight": cli.infer_structure_weight,
            "max_samples_cap": cli.max_samples,
            "predictions_csv": pred_out,
        }
    )

    # ── Per-llm_type / per-data_type breakdown ───────────────────────────────
    preds_df = pd.DataFrame(pred_rows)
    llm_rows, data_rows = [], []

    for ltype in sorted(preds_df["llm_type"].unique()):
        idx = preds_df[preds_df["llm_type"] == ltype].index.tolist()
        m_sub = compute_metrics_safe(np.array(labels)[idx], scores[idx])
        if m_sub:
            m_sub["group"] = f"llm_type={ltype}"
            llm_rows.append(m_sub)

    for dtype in sorted(preds_df["data_type"].unique()):
        idx = preds_df[preds_df["data_type"] == dtype].index.tolist()
        m_sub = compute_metrics_safe(np.array(labels)[idx], scores[idx])
        if m_sub:
            m_sub["group"] = f"data_type={dtype}"
            data_rows.append(m_sub)

    csv_out, json_out, preds_all_out = flush_eval_results(
        output_dir, stamp, suffix, preds_dir, eval_config, [m], pred_rows
    )

    if llm_rows or data_rows:
        breakdown_out = os.path.join(
            output_dir, f"detectRL_breakdown{suffix}_{stamp}.csv"
        )
        pd.DataFrame([{"group": "overall", **m}] + llm_rows + data_rows).to_csv(
            breakdown_out, index=False
        )

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    print(
        f"  n={m['n_samples']} (human={m['n_human']}, ai={m['n_ai']}) | "
        f"ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} | "
        f"tpr@fpr0.005={m['tpr_at_fpr_0.005']:.4f} "
        f"tpr@fpr0.01={m['tpr_at_fpr_0.01']:.4f} "
        f"tpr@fpr0.03={m['tpr_at_fpr_0.03']:.4f} | "
        f"acc@0.5={m['acc']:.4f} f1@0.5={m['f1']:.4f} | "
        f"acc@best={m['acc_at_best_thresh']:.4f} "
        f"f1@best={m['f1_at_best_thresh']:.4f} "
        f"(thr={m['best_threshold_youden']:.4f})"
    )

    if llm_rows:
        print("\nPer LLM type:")
        for r in llm_rows:
            print(
                f"  [{r['group']}] n={r['n_samples']} | "
                f"ROC-AUC={r['roc_auc']:.4f} acc@0.5={r['acc']:.4f} f1@0.5={r['f1']:.4f}"
            )

    if data_rows:
        print("\nPer data type:")
        for r in data_rows:
            print(
                f"  [{r['group']}] n={r['n_samples']} | "
                f"ROC-AUC={r['roc_auc']:.4f} acc@0.5={r['acc']:.4f} f1@0.5={r['f1']:.4f}"
            )

    print(f"\nWrote (final):\n  {csv_out}\n  {json_out}")
    print(f"  {preds_dir}/  (all_predictions.csv)")
    if preds_all_out:
        print(f"  {preds_all_out}")
    print("\nMean ROC-AUC:", m["roc_auc"])
    print("Mean PR-AUC:", m["pr_auc"])


if __name__ == "__main__":
    main()
