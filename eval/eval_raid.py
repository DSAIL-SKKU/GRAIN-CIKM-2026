#!/usr/bin/env python3
"""
RAID-news test split 전용 평가 스크립트.

데이터: RAID HuggingFace train split → news domain → 8:1:1 stratified split → test 10%만 사용.
모델:   eval.py 와 동일한 RoBERTa (structure fc) 체크포인트.

Example:
  cd /home/dsail/2026-AINews-Detection/model
  python eval_raid.py --run_dir results/2026-05-31/RAID-news-... --DEVICE cuda:0
  python eval_raid.py --run_dir results/... --checkpoint RAID-news_5_8_8.0_....pth
  python eval_raid.py --detectLLM all --max_samples 500
"""

from __future__ import annotations

import argparse
import json
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
    compute_metrics,
    build_prediction_rows,
    flush_eval_results,
    load_detector,
    predict_scores_fast,
    resolve_checkpoint,
    resolve_run_dir,
    setup_cuda_fast,
)


def patch_parse_args(infer_structure_weight=None, no_amp=False):
    import parse as parse_mod

    parse_mod.args.article_level = True
    parse_mod.args.article_max_units = 50
    parse_mod.args.reg = getattr(parse_mod.args, "reg", 8.0)
    parse_mod.args.iter = getattr(parse_mod.args, "iter", 0)
    parse_mod.args.no_amp = no_amp
    if infer_structure_weight is not None:
        parse_mod.args.infer_structure_weight = infer_structure_weight


def load_raid_test(
    detect_llm: str,
    domain: str = "news",
    include_adversarial: bool = False,
    iter_seed: int = 0,
    cache_path: str = "",
) -> tuple[list[str], list[int]]:
    """
    RAID train split → domain 필터 → 8:1:1 → test 10% 반환.
    반환값: (texts, labels)  labels: 0=human, 1=AI
    """
    import parse as parse_mod

    parse_mod.args.detectLLM = detect_llm
    parse_mod.args.raid_domain = domain
    parse_mod.args.raid_include_adversarial = include_adversarial
    parse_mod.args.iter = iter_seed
    parse_mod.args.split_cache = cache_path
    parse_mod.args.sentence_length = -1

    from dataloader_raid import load_raid_news

    data = load_raid_news(
        detect_llm=detect_llm,
        iter_seed=iter_seed,
        include_adversarial=include_adversarial,
        cache_path=cache_path or "",
        domain=domain,
    )
    texts = data["test"]["text"]
    labels = data["test"]["label"]
    print(
        f"RAID test split: {len(texts)} samples "
        f"(human={sum(1 for l in labels if l==0)}, "
        f"ai={sum(1 for l in labels if l==1)})"
    )
    return texts, labels


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RoBERTa on RAID-news test split (8:1:1)."
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
        help=".pth filename under run_dir. 비우면 run_dir 안 최신 파일.",
    )
    parser.add_argument("--DEVICE", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128, help="fc 배치 (기사 수)")
    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=128,
        help="structure RoBERTa 임베딩 배치 크기",
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--infer_structure_weight",
        type=float,
        default=1.0,
        help="추론 logit 결합: w*structure+(1-w)*unit_mean. 1=structure만.",
    )
    parser.add_argument("--pos_bit", type=int, default=1)
    parser.add_argument(
        "--detectLLM",
        type=str,
        default="all",
        help="RAID LLM 필터: 모델명 또는 all/non_human.",
    )
    parser.add_argument("--domain", type=str, default="news")
    parser.add_argument("--include_adversarial", action="store_true")
    parser.add_argument(
        "--iter", type=int, default=0, help="random seed index (split 재현용)"
    )
    parser.add_argument(
        "--split_cache",
        type=str,
        default="",
        help="split 캐시 JSON 경로. 비우면 data_cache/ 아래 자동 생성.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="test set에서 최대 샘플 수. 0이면 전체.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
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

    output_dir = (
        os.path.abspath(cli.output_dir)
        if cli.output_dir
        else os.path.join(run_dir, "eval_raid")
    )
    os.makedirs(output_dir, exist_ok=True)

    device = cli.DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device = "cpu"

    print(f"Loading RoBERTa from {ckpt_path}")
    detector = load_detector(run_dir, device, cli.pos_bit, ckpt_path)
    setup_cuda_fast(device)

    # ── RAID test split 로드 ───────────────────────────────────────
    texts, labels = load_raid_test(
        detect_llm=cli.detectLLM,
        domain=cli.domain,
        include_adversarial=cli.include_adversarial,
        iter_seed=cli.iter,
        cache_path=cli.split_cache,
    )
    if cli.max_samples > 0:
        texts = texts[: cli.max_samples]
        labels = labels[: cli.max_samples]
        print(f"Quick mode: using first {len(texts)} samples")

    meta = [
        {"csv_row": i, "label_name": "human" if l == 0 else "ai"}
        for i, l in enumerate(labels)
    ]

    # ── 예측 및 지표 계산 ─────────────────────────────────────────
    scores = predict_scores_fast(
        detector, texts, cli.batch_size, cli.embed_batch_size, cli.pos_bit
    )
    m = compute_metrics(labels, scores)
    pred_rows = build_prediction_rows(labels, scores, meta, m["best_threshold_youden"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_n{cli.max_samples}" if cli.max_samples > 0 else ""
    preds_dir = os.path.join(output_dir, f"predictions{suffix}_{stamp}")
    os.makedirs(preds_dir, exist_ok=True)

    llm_tag = cli.detectLLM.replace("/", "_")
    m.update(
        {
            "file": f"RAID-{cli.domain}-{llm_tag}",
            "run_dir": run_dir,
            "checkpoint": ckpt_name,
            "infer_structure_weight": cli.infer_structure_weight,
            "detectLLM": cli.detectLLM,
            "domain": cli.domain,
            "max_samples_cap": cli.max_samples,
            "predictions_csv": os.path.join(preds_dir, "raid_test_preds.csv"),
        }
    )

    pd.DataFrame(pred_rows).to_csv(m["predictions_csv"], index=False)

    eval_config = {
        "dataset": "RAID",
        "domain": cli.domain,
        "detectLLM": cli.detectLLM,
        "split": "test (8:1:1)",
        "iter": cli.iter,
        "run_dir": run_dir,
        "checkpoint": ckpt_name,
        "DEVICE": device,
        "batch_size": cli.batch_size,
        "embed_batch_size": cli.embed_batch_size,
        "infer_structure_weight": cli.infer_structure_weight,
        "max_samples": cli.max_samples,
        "timestamp": stamp,
        "output_dir": output_dir,
    }

    flush_eval_results(
        output_dir, stamp, suffix, preds_dir, eval_config, [m], pred_rows
    )

    # ── 결과 출력 ─────────────────────────────────────────────────
    print(
        f"\n=== RAID-{cli.domain} ({cli.detectLLM}) test set ===\n"
        f"  n={m['n_samples']} (human={m['n_human']}, ai={m['n_ai']})\n"
        f"  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}\n"
        f"  tpr@fpr0.005={m['tpr_at_fpr_0.005']:.4f} "
        f"tpr@fpr0.01={m['tpr_at_fpr_0.01']:.4f} "
        f"tpr@fpr0.03={m['tpr_at_fpr_0.03']:.4f}\n"
        f"  acc@0.5={m['acc']:.4f}  f1@0.5={m['f1']:.4f}\n"
        f"  acc@best={m['acc_at_best_thresh']:.4f}  "
        f"f1@best={m['f1_at_best_thresh']:.4f}  "
        f"(thr={m['best_threshold_youden']:.4f})\n"
        f"  checkpoint: {ckpt_name}\n"
        f"  output: {output_dir}"
    )


if __name__ == "__main__":
    main()


# cd /home/dsail/2026-AINews-Detection/model
# # 최신 체크포인트로 RAID news 전체 평가
# python eval_raid.py --DEVICE cuda:0
# # 특정 체크포인트 지정
# python eval_raid.py \
#   --run_dir results/2026-05-31/RAID-news-20-58-38-342807 \
#   --checkpoint RAID-news_5_8_8.0_20260531-205838.pth \
#   --infer_structure_weight 0.7
