import json
import random

import tqdm

from parse import args


def process_spaces(text):
    return (
        text.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ;", ";")
        .replace(" '", "'")
        .replace(" ’ ", "'")
        .replace(" :", ":")
        .replace("<newline>", "\n")
        .replace("`` ", '"')
        .replace(" ''", '"')
        .replace("''", '"')
        .replace(".. ", "... ")
        .replace(" )", ")")
        .replace("( ", "(")
        .replace(" n't", "n't")
        .replace(" i ", " I ")
        .replace(" i'", " I'")
        .replace("\\'", "'")
        .replace("\n ", "\n")
        .strip()
    )


def _stratified_train_val(texts_by_label, val_fraction):
    train_tx, train_lb = [], []
    val_tx, val_lb = [], []
    for lab_int, texts in enumerate(texts_by_label):
        if not texts:
            continue
        xs = texts[:]
        random.shuffle(xs)
        if val_fraction <= 0 or len(xs) == 1:
            train_tx.extend(xs)
            train_lb.extend([lab_int] * len(xs))
            continue
        n_val = int(len(xs) * val_fraction)
        if n_val == 0:
            n_val = 1
        n_val = min(n_val, len(xs) - 1)
        val_x = xs[:n_val]
        tr_x = xs[n_val:]
        val_tx.extend(val_x)
        val_lb.extend([lab_int] * len(val_x))
        train_tx.extend(tr_x)
        train_lb.extend([lab_int] * len(tr_x))
    return train_tx, train_lb, val_tx, val_lb


def load(name, detectLLM=None):
    # detectLLM: API 호환용 (MultiDomainXSum에서는 미사용)
    if name not in ("MultiDomainXSum", "kor_aigen"):
        raise ValueError(
            f"Supported datasets: MultiDomainXSum, kor_aigen. Got {name!r}"
        )

    if name == "kor_aigen":
        return _load_kor_aigen()

    random.seed(getattr(args, "iter", 0))
    train_path = args.xsum_train_path
    test_path = args.xsum_test_path

    with open(train_path, "r", encoding="utf-8") as file:
        train_raw = json.load(file)

    humans, llms = [], []
    for row in train_raw:
        t = process_spaces(str(row.get("text", "")))
        if not t.strip():
            continue
        lab = row.get("label", "")
        if lab == "human":
            humans.append(t)
        elif lab == "llm":
            llms.append(t)

    tr_x, tr_y, va_x, va_y = _stratified_train_val([humans, llms], args.val_fraction)

    if len(va_x) == 0:
        raise ValueError(
            "MultiDomainXSum: 검증 샘플이 없습니다. --val_fraction을 올리거나 "
            "train JSON에 human/llm이 모두 있는지 확인하세요."
        )

    with open(test_path, "r", encoding="utf-8") as file:
        test_raw = json.load(file)

    te_x, te_y = [], []
    for row in tqdm.tqdm(test_raw, desc="MultiDomainXSum test"):
        t = process_spaces(str(row.get("text", "")))
        if not t.strip():
            continue
        lab = row.get("label", "")
        if lab == "human":
            te_x.append(t)
            te_y.append(0)
        elif lab == "llm":
            te_x.append(t)
            te_y.append(1)

    return {
        "train": {"text": tr_x, "label": tr_y},
        "val": {"text": va_x, "label": va_y},
        "test": {"text": te_x, "label": te_y},
    }


def _load_kor_aigen():
    random.seed(getattr(args, "iter", 0))
    train_path = args.kor_train_path
    valid_path = args.kor_valid_path
    test_path = args.kor_test_path

    with open(train_path, "r", encoding="utf-8") as file:
        train_raw = json.load(file)
    with open(valid_path, "r", encoding="utf-8") as file:
        valid_raw = json.load(file)
    with open(test_path, "r", encoding="utf-8") as file:
        test_raw = json.load(file)

    def to_int_label(lab):
        """int(0/1) 또는 str("human"/"ai") 라벨을 0/1로 통일"""
        if lab == 0 or lab == "human":
            return 0
        if lab == 1 or lab == "ai":
            return 1
        return None

    tr_x, tr_y = [], []
    for row in train_raw:
        t = process_spaces(str(row.get("text", "")))
        if not t.strip():
            continue
        lab = to_int_label(row.get("label"))
        if lab is not None:
            tr_x.append(t)
            tr_y.append(lab)

    va_x, va_y = [], []
    for row in valid_raw:
        t = process_spaces(str(row.get("text", "")))
        if not t.strip():
            continue
        lab = to_int_label(row.get("label"))
        if lab is not None:
            va_x.append(t)
            va_y.append(lab)

    te_x, te_y = [], []
    for row in tqdm.tqdm(test_raw, desc="kor test"):
        t = process_spaces(str(row.get("text", "")))
        if not t.strip():
            continue
        lab = to_int_label(row.get("label"))
        if lab == 0:
            te_x.append(t)
            te_y.append(0)
        elif lab == 1:
            te_x.append(t)
            te_y.append(1)

    return {
        "train": {"text": tr_x, "label": tr_y},
        "val": {"text": va_x, "label": va_y},
        "test": {"text": te_x, "label": te_y},
    }
