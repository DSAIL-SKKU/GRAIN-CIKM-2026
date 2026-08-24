"""
RAID HuggingFace train split, news domain only, stratified 8:1:1 train/val/test.

Labels: 0=human, 1=LLM.
detectLLM=chatgpt 등 특정 모델, 또는 all/non_human 이면 human이 아닌 모든 model.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

import tqdm
from sklearn.model_selection import train_test_split

from dataloader import process_spaces
from parse import args

RAID_HF_SPLIT = "train"
RAID_DOMAIN = "news"
DEFAULT_CACHE_DIR = "data_cache"
SPLIT_SEED_BASE = 42


def _load_raid_raw(split: str, include_adv: bool):
    try:
        from raid.utils import load_data as raid_load_data  # type: ignore

        return raid_load_data(split=split, include_adversarial=include_adv)
    except ModuleNotFoundError:
        from datasets import load_dataset  # type: ignore

        last_err = None
        for call in (
            lambda: load_dataset("liamdugan/raid", split=split),
            lambda: load_dataset("liamdugan/raid", "raid", split=split),
        ):
            try:
                df = call().to_pandas()
                break
            except Exception as e:
                last_err = e
        else:
            raise RuntimeError(
                "Failed to load Hugging Face RAID. Install `raid` or upgrade "
                "`datasets huggingface_hub fsspec`. "
                f"Original error: {last_err}"
            ) from last_err

        if not include_adv and "attack" in df.columns:
            attack_col = df["attack"].astype(str).str.strip().str.lower()
            df = df[attack_col.isin({"none", "nan", ""}) | df["attack"].isna()]
        return df


def _is_human(cell) -> bool:
    return str(cell).strip().lower() == "human"


def _use_all_llm(detect_llm: str) -> bool:
    v = str(detect_llm).strip().lower()
    return v in ("", "all", "*", "any", "non_human", "non-human", "llm", "machine")


def _is_llm(cell, detect_llm: str) -> bool:
    if _use_all_llm(detect_llm):
        return not _is_human(cell)
    return _model_matches(cell, detect_llm)


def _model_matches(cell, detect_llm: str) -> bool:
    if _is_human(cell):
        return False
    m = str(cell).strip().lower()
    d = str(detect_llm).strip().lower()
    if m == d:
        return True
    m2 = m.replace("-", "").replace("_", "")
    d2 = d.replace("-", "").replace("_", "")
    if m2 == d2:
        return True
    if len(d) >= 3 and (d in m or m in d):
        return True
    return False


def _texts_from_df(df, desc: str = "") -> List[str]:
    out: List[str] = []
    rows = tqdm.tqdm(df.iterrows(), total=len(df), desc=desc) if desc else df.iterrows()
    for _, row in rows:
        t = process_spaces(str(row["generation"]).strip())
        if t and len(t.split()) > 1:
            out.append(t)
    return out


def _cache_tag(detect_llm: str) -> str:
    if _use_all_llm(detect_llm):
        return "all_llm"
    return str(detect_llm).strip().replace("/", "_")


def _default_cache_path(detect_llm: str, iter_seed: int, sentence_length: int) -> str:
    sl = "full" if sentence_length == -1 else str(sentence_length)
    name = f"raid_news_{_cache_tag(detect_llm)}_iter{iter_seed}_sl{sl}.json"
    return os.path.join(DEFAULT_CACHE_DIR, name)


def load_raid_news(
    detect_llm: str,
    iter_seed: int = 0,
    sentence_length: int = -1,
    include_adversarial: bool = False,
    cache_path: Optional[str] = None,
    domain: str = RAID_DOMAIN,
) -> Dict[str, Any]:
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded cached RAID split: {cache_path}")
        return data

    df = _load_raid_raw(RAID_HF_SPLIT, include_adversarial)

    dom = (domain or "").strip().lower()
    if dom:
        if "domain" not in df.columns:
            raise ValueError("RAID dataframe has no `domain` column.")
        if dom == "wikipedia":
            dom = "wiki"
        dcol = df["domain"].astype(str).str.strip().str.lower()
        df = df[dcol == dom]

    if "generation" not in df.columns or "model" not in df.columns:
        raise ValueError("RAID requires `generation` and `model` columns.")

    df_llm = df[df["model"].apply(lambda x: _is_llm(x, detect_llm))]
    df_human = df[df["model"].apply(_is_human)]
    if df_llm.empty:
        raise ValueError(
            f"No RAID LLM rows for detectLLM={detect_llm!r}. "
            "Use a specific model name or all/non_human for every non-human model."
        )
    if df_human.empty:
        raise ValueError("No human-labeled RAID rows after filtering.")

    llm_label = "all non-human models" if _use_all_llm(detect_llm) else detect_llm
    if _use_all_llm(detect_llm):
        models = sorted(df_llm["model"].astype(str).str.strip().unique().tolist())
        print(
            f"  LLM models ({len(models)}): {', '.join(models[:12])}{'...' if len(models) > 12 else ''}"
        )
    print(
        f"RAID {RAID_HF_SPLIT} domain={domain}: "
        f"{len(df_human)} human articles, {len(df_llm)} LLM articles ({llm_label})"
    )

    h_texts = _texts_from_df(df_human, desc="RAID human")
    l_texts = _texts_from_df(df_llm, desc=f"RAID {llm_label}")

    rng = random.Random(iter_seed)
    rng.shuffle(h_texts)
    rng.shuffle(l_texts)
    n = min(len(h_texts), len(l_texts))
    if n == 0:
        raise ValueError(
            "No text samples after preprocessing; check sentence_length or RAID filters."
        )
    h_texts = h_texts[:n]
    l_texts = l_texts[:n]

    texts = h_texts + l_texts
    labels = [0] * n + [1] * n

    split_seed = SPLIT_SEED_BASE + int(iter_seed)
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.1,
        random_state=split_seed,
        shuffle=True,
        stratify=labels,
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts,
        train_val_labels,
        test_size=1.0 / 9.0,
        random_state=split_seed,
        shuffle=True,
        stratify=train_val_labels,
    )

    data = {
        "train": {"text": train_texts, "label": train_labels},
        "val": {"text": val_texts, "label": val_labels},
        "test": {"text": test_texts, "label": test_labels},
    }

    print(
        f"RAID 8:1:1 split — train={len(train_texts)} val={len(val_texts)} test={len(test_texts)}"
    )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"Cached RAID split: {cache_path}")

    return data


def load(name: Optional[str] = None, detectLLM: Optional[str] = None) -> Dict[str, Any]:
    dataset = name or getattr(args, "dataset", "RAID-news")
    if dataset not in ("RAID", "RAID-news"):
        raise ValueError(
            f"dataloader_raid.load expects RAID or RAID-news, got {dataset!r}"
        )

    detect = detectLLM or getattr(args, "detectLLM", "all")
    iter_seed = int(getattr(args, "iter", 0))
    sentence_length = int(getattr(args, "sentence_length", -1))
    include_adv = bool(getattr(args, "raid_include_adversarial", False))
    domain = getattr(args, "raid_domain", RAID_DOMAIN) or RAID_DOMAIN

    cache = (getattr(args, "split_cache", "") or "").strip()
    if not cache:
        cache = _default_cache_path(detect, iter_seed, sentence_length)

    return load_raid_news(
        detect_llm=detect,
        iter_seed=iter_seed,
        sentence_length=sentence_length,
        include_adversarial=include_adv,
        cache_path=cache,
        domain=domain,
    )
