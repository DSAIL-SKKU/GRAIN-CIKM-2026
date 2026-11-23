#!/usr/bin/env python3
"""
Evaluate model/ RoBERTa (structure + fc) on ai_gen_news *_Eng_merged.csv files.

Example:
  cd /home/dsail/2026-AINews-Detection/model
  python eval.py --DEVICE cuda:0
  python eval.py --run_dir results/2026-05-17/MultiDomainXSum-23-32-59-950098 --DEVICE cuda:0
  python eval.py --run_dir results/... --checkpoint MultiDomainXSum_5_32_8.0_20260517-233259.pth
  python eval.py --max_samples 500
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.dirname(SRC_DIR)
os.chdir(MODEL_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DEFAULT_DATA_ROOT = (
    # "/home/dsail/yjlee/police/ai_gen/dataset/ai_gen_news_dataset/merged_by_model_lang"
    # "/home/dsail/yjlee/police/ai_gen/dataset/ai_gen_news_dataset/test"
    "/home/dsail/yjlee/police/ai_gen/dataset/ai_gen_news_dataset"
)

INCLUDE_CSV_PREFIXES = (
    # "GPT_5.4_pro_",
    # "Gemini_2.5_Flash_",
    "GPT_5.4_nano_",
    "GPT_5_",
    "GPT_5_nano_",
    "GPT_5.4_",
    "Claude_Opus_4.1_",
    "Claude_Sonnet_4.5_",
    "Claude_Opus_4.6_",
    "Claude_Sonnet_4.6_",
    "Gemini_2.5_Pro_",
    "Gemini_3.1_Flash_",
    "Gemini_3.1_Pro_",
)
RESULTS_ROOT = os.path.join(MODEL_DIR, "results")


def patch_parse_args(infer_structure_weight=None, no_amp=False):
    import parse as parse_mod

    parse_mod.args.article_level = True
    parse_mod.args.article_max_units = 32
    parse_mod.args.reg = getattr(parse_mod.args, "reg", 0.7)
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
        if not os.path.isdir(path):
            raise FileNotFoundError(f"--run_dir not found: {path}")
        return path

    if not os.path.isdir(RESULTS_ROOT):
        raise FileNotFoundError(
            f"No --run_dir given and {RESULTS_ROOT} does not exist. "
            "Train with train.py first or pass --run_dir."
        )

    best_mtime = -1.0
    best_dir = None
    for root, _dirs, files in os.walk(RESULTS_ROOT):
        if any(f.endswith(".pth") for f in files):
            mtime = os.path.getmtime(root)
            if mtime > best_mtime:
                best_mtime = mtime
                best_dir = root

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


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def title_from_body(body) -> str:
    body = "" if pd.isna(body) else str(body).strip()
    if not body:
        return ""
    parts = _SENTENCE_SPLIT.split(body, maxsplit=1)
    return parts[0].strip()


def format_article(title, body) -> str:
    title = "" if pd.isna(title) else str(title).strip()
    body = "" if pd.isna(body) else str(body).strip()
    if not body:
        return ""
    return body


def load_articles_from_csv(csv_path: str, max_samples: int = 0):
    df = pd.read_csv(csv_path)
    texts, labels, meta = [], [], []

    for row_idx, row in enumerate(df.itertuples(index=False)):
        if max_samples > 0 and len(texts) >= max_samples:
            break

        human_body = getattr(row, "human_real", "")
        ai_body = getattr(row, "ai_article", "")

        human_text = format_article(title_from_body(human_body), human_body)
        if human_text and (max_samples <= 0 or len(texts) < max_samples):
            texts.append(human_text)
            labels.append(0)
            meta.append({"csv_row": row_idx, "label_name": "human"})

        ai_text = format_article(title_from_body(ai_body), ai_body)
        if ai_text and (max_samples <= 0 or len(texts) < max_samples):
            texts.append(ai_text)
            labels.append(1)
            meta.append({"csv_row": row_idx, "label_name": "ai"})

    return texts, labels, meta


def predict_scores_fast(detector, texts, fc_batch_size, embed_batch_size, pos_bit):
    """fc_batch_size = 기사 배치 크기."""
    detector.eval()
    n = len(texts)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    chunks = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, n, fc_batch_size),
            desc="predict",
            leave=False,
            total=max(1, (n + fc_batch_size - 1) // fc_batch_size),
        ):
            chunk = texts[start : start + fc_batch_size]
            probs = detector.forward(chunk, embed_batch_size=embed_batch_size)
            chunks.append(probs[:, pos_bit].detach())
    return torch.cat(chunks).cpu().numpy().astype(np.float64)


def metrics_at_threshold(y_true, y_score, threshold: float):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_metrics(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        raise ValueError("Need both human (0) and AI (1) labels to compute AUC.")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_idx = int(np.argmax(tpr - fpr))
    best_thresh = float(thresholds[youden_idx])

    def tpr_at_fpr(max_fpr):
        idx = np.where(fpr <= max_fpr)[0]
        if len(idx) == 0:
            return float("nan")
        return float(tpr[idx[-1]])

    result = {
        "n_samples": int(len(y_true)),
        "n_human": int((y_true == 0).sum()),
        "n_ai": int((y_true == 1).sum()),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "tpr_at_fpr_0.005": tpr_at_fpr(0.005),
        "tpr_at_fpr_0.01": tpr_at_fpr(0.01),
        "tpr_at_fpr_0.03": tpr_at_fpr(0.03),
        "best_threshold_youden": best_thresh,
    }
    at_05 = metrics_at_threshold(y_true, y_score, 0.5)
    result.update(
        {
            "threshold": at_05["threshold"],
            "acc": at_05["acc"],
            "precision": at_05["precision"],
            "recall": at_05["recall"],
            "f1": at_05["f1"],
        }
    )

    at_best = metrics_at_threshold(y_true, y_score, best_thresh)
    result.update(
        {
            "acc_at_best_thresh": at_best["acc"],
            "precision_at_best_thresh": at_best["precision"],
            "recall_at_best_thresh": at_best["recall"],
            "f1_at_best_thresh": at_best["f1"],
        }
    )
    return result


def build_prediction_rows(labels, scores, meta, best_threshold: float):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    rows = []
    for i, (label, score) in enumerate(zip(labels, scores)):
        info = meta[i] if i < len(meta) else {}
        pred_05 = int(score >= 0.5)
        pred_best = int(score >= best_threshold)
        rows.append(
            {
                "sample_idx": i,
                "csv_row": info.get("csv_row"),
                "label": int(label),
                "label_name": info.get("label_name", "ai" if label else "human"),
                "score": float(score),
                "pred_at_0.5": pred_05,
                "pred_at_best": pred_best,
                "correct_at_0.5": int(pred_05 == label),
                "correct_at_best": int(pred_best == label),
            }
        )
    return rows


def _mean_numeric_metrics(summary_df: pd.DataFrame) -> dict:
    skip = {"file", "csv_path", "run_dir", "checkpoint", "predictions_csv"}
    return {
        col: float(summary_df[col].mean())
        for col in summary_df.columns
        if col not in skip and pd.api.types.is_numeric_dtype(summary_df[col])
    }


def flush_eval_results(
    output_dir,
    stamp,
    suffix,
    preds_dir,
    eval_config,
    rows,
    all_predictions,
):
    """CSV 하나 평가할 때마다 summary/predictions를 덮어써서 중간 저장."""
    summary_df = pd.DataFrame(rows)
    csv_out = os.path.join(output_dir, f"ai_gen_news_eval{suffix}_{stamp}.csv")
    json_out = os.path.join(output_dir, f"ai_gen_news_eval{suffix}_{stamp}.json")
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model/ RoBERTa on ai_gen_news Eng merged CSV files."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="train.py save dir (results/날짜/...). 비우면 results/ 최신 .pth 폴더.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="(deprecated) use --run_dir",
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
        "--no_amp",
        action="store_true",
        help="structure 임베딩 AMP(fp16) 끄기",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="PyTorch 2.x: structure RoBERTa에 torch.compile (첫 배치 느림)",
    )
    parser.add_argument(
        "--infer_structure_weight",
        type=float,
        default=0.7,
        help="추론 logit 결합: w*structure+(1-w)*unit_mean. 1=structure만.",
    )
    parser.add_argument("--pos_bit", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*_Eng_merged.csv")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="CSV당 최대 기사 수(human+ai 합). 0이면 전체",
    )
    cli = parser.parse_args()

    patch_parse_args(
        infer_structure_weight=cli.infer_structure_weight, no_amp=cli.no_amp
    )

    print(
        f"infer_structure_weight={cli.infer_structure_weight} "
        f"(logits = w*structure + (1-w)*unit_mean; 1=structure only)"
    )

    run_dir = resolve_run_dir(cli.run_dir or cli.model_dir)
    ckpt_path = resolve_checkpoint(run_dir, cli.checkpoint)
    ckpt_name = os.path.basename(ckpt_path)

    data_root = os.path.abspath(cli.data_root)
    output_dir = (
        os.path.abspath(cli.output_dir)
        if cli.output_dir
        else os.path.join(run_dir, "eval_ai_gen_news")
    )
    os.makedirs(output_dir, exist_ok=True)

    def _prefix_order(path):
        name = os.path.basename(path)
        for i, prefix in enumerate(INCLUDE_CSV_PREFIXES):
            if name.startswith(prefix):
                return i
        return len(INCLUDE_CSV_PREFIXES)

    csv_files = sorted(
        [
            p
            for p in glob.glob(os.path.join(data_root, cli.pattern))
            if os.path.basename(p).startswith(INCLUDE_CSV_PREFIXES)
        ],
        key=_prefix_order,
    )
    # csv_files = sorted(
    #     glob.glob(os.path.join(data_root, cli.pattern)),
    # )
    if cli.max_files > 0:
        csv_files = csv_files[: cli.max_files]
    if not csv_files:
        raise FileNotFoundError(f"No files matching {cli.pattern!r} under {data_root}")

    device = cli.DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device = "cpu"

    print(f"Loading RoBERTa from {ckpt_path}")
    if cli.max_samples > 0:
        print(f"Quick mode: up to {cli.max_samples} articles per CSV file")
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
        "data_root": data_root,
        "run_dir": run_dir,
        "checkpoint": ckpt_name,
        "checkpoint_path": ckpt_path,
        "DEVICE": device,
        "batch_size": cli.batch_size,
        "embed_batch_size": cli.embed_batch_size,
        "infer_structure_weight": cli.infer_structure_weight,
        "pos_bit": cli.pos_bit,
        "pattern": cli.pattern,
        "max_samples": cli.max_samples,
        "max_files": cli.max_files,
        "no_amp": cli.no_amp,
        "compile": cli.compile,
        "timestamp": stamp,
        "output_dir": output_dir,
    }

    rows = []
    all_predictions = []
    last_csv_out = last_json_out = last_preds_out = ""

    for csv_path in csv_files:
        name = os.path.basename(csv_path)
        print(f"\n=== {name} ===")
        texts, labels, meta = load_articles_from_csv(
            csv_path, max_samples=cli.max_samples
        )
        if len(texts) == 0:
            print("  skip: no valid articles")
            continue

        scores = predict_scores_fast(
            detector,
            texts,
            cli.batch_size,
            cli.embed_batch_size,
            cli.pos_bit,
        )
        m = compute_metrics(labels, scores)
        pred_rows = build_prediction_rows(
            labels, scores, meta, m["best_threshold_youden"]
        )
        for row in pred_rows:
            row["file"] = name
            row["csv_path"] = csv_path
        all_predictions.extend(pred_rows)

        stem = os.path.splitext(name)[0]
        pred_out = os.path.join(preds_dir, f"{stem}_preds.csv")
        pd.DataFrame(pred_rows).to_csv(pred_out, index=False)

        m["file"] = name
        m["csv_path"] = csv_path
        m["run_dir"] = run_dir
        m["checkpoint"] = ckpt_name
        m["infer_structure_weight"] = cli.infer_structure_weight
        m["max_samples_cap"] = cli.max_samples
        m["predictions_csv"] = pred_out
        rows.append(m)

        last_csv_out, last_json_out, last_preds_out = flush_eval_results(
            output_dir,
            stamp,
            suffix,
            preds_dir,
            eval_config,
            rows,
            all_predictions,
        )

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
        print(f"  checkpoint save: {last_csv_out}")

    if not rows:
        print("No results.")
        return

    summary_df = pd.DataFrame(rows)
    print(f"\nWrote (final):\n  {last_csv_out}\n  {last_json_out}")
    print(f"  {preds_dir}/  ({len(rows)} per-file CSV + all_predictions.csv)")
    if last_preds_out:
        print(f"  {last_preds_out}")
    print("\nMean ROC-AUC:", summary_df["roc_auc"].mean())
    print("Mean PR-AUC:", summary_df["pr_auc"].mean())


if __name__ == "__main__":
    main()
