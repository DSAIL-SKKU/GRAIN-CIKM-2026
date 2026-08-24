import os
import re
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import _amp_autocast, _embed_amp_enabled

tokenizer, model, device = None, None, None

NUM_STRUCTURE_SCALARS = 2
MAX_EMBED_LENGTH = 512


def _default_RoBERTa_dir():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (
        os.path.join(base, "models", "RoBERTa"),
        os.path.normpath(os.path.join(base, "..", "src", "save_models", "RoBERTa")),
    ):
        if os.path.isdir(candidate) and os.path.isfile(
            os.path.join(candidate, "config.json")
        ):
            return candidate
    return os.path.join(base, "models", "RoBERTa")


def init_structure_embedder(
    target_device="cpu",
    backbone=None,
    shared_tokenizer=None,
    device_=None,
    model_=None,
    tokenizer_=None,
):
    """RoBERTa 학습 시 model_/tokenizer_ 또는 backbone/shared_tokenizer로 roberta 공유."""
    global tokenizer, model, device

    if device_ is not None:
        target_device = device_
    if model_ is not None:
        backbone = model_
        shared_tokenizer = tokenizer_

    target_device = str(target_device)

    if backbone is not None:
        tokenizer = shared_tokenizer
        model = backbone
        device = target_device
        model.eval()
        model.to(device)
        return

    if model is not None:
        if str(device) != target_device:
            model = model.to(target_device)
        device = target_device
        return

    device = target_device
    RoBERTa_dir = _default_RoBERTa_dir()
    tokenizer = AutoTokenizer.from_pretrained(RoBERTa_dir)
    classifier = AutoModelForSequenceClassification.from_pretrained(
        RoBERTa_dir,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model = classifier.roberta
    model.to(device).eval()


def embedding_dim():
    if model is not None:
        return model.config.hidden_size
    init_structure_embedder("cpu")
    return model.config.hidden_size


def structure_feature_dim():
    return embedding_dim() + NUM_STRUCTURE_SCALARS


def split_units(text):
    """
    문단이 있으면 문단 기준.
    문단 구분이 없으면 문장 기준 fallback.
    """

    if not isinstance(text, str):
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if len(paragraphs) >= 2:
        return paragraphs

    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20
    ]

    return sentences


def _merge_texts(texts):
    parts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    return "\n\n".join(parts)


def split_article(text):
    """
    뉴스 형식(첫 줄 제목 / 제목: 접두) 또는 본문만 있는 텍스트에서
    title, body, paragraphs를 분리한다.
    """

    if not isinstance(text, str) or not text.strip():
        return "", "", []

    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return "", "", []

    first = lines[0]
    if first.startswith("제목:"):
        title = first.replace("제목:", "", 1).strip()
        body_lines = lines[1:]
    elif len(lines) >= 2 and len(first) <= 120:
        title = first
        body_lines = lines[1:]
    else:
        title = ""
        body_lines = lines

    body_text = "\n".join(body_lines)
    paragraphs = split_units(body_text if body_text else text)
    if not paragraphs and body_text:
        paragraphs = [p.strip() for p in body_lines if p.strip()]

    return title, body_text or text, paragraphs


def _collect_paragraphs(texts):
    paragraphs = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        units = split_units(text)
        if units:
            paragraphs.extend(units)
        else:
            paragraphs.append(text.strip())
    return paragraphs


def mean_pooling(outputs, attention_mask):
    token_embeddings = outputs.last_hidden_state

    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(
        mask.sum(dim=1), min=1e-9
    )


def get_embedding(text):
    if model is None:
        init_structure_embedder(device or "cpu")

    if not isinstance(text, str) or len(text.strip()) == 0:
        text = ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_EMBED_LENGTH,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        with _amp_autocast(enabled=_embed_amp_enabled()):
            outputs = model(**inputs)

    emb = mean_pooling(outputs, inputs["attention_mask"])

    return emb.squeeze().detach().cpu().numpy()


def cosine(a, b):
    return float(cosine_similarity([a], [b])[0][0])


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def paragraph_information_gains(embeddings):
    gains = [1.0]
    for i in range(1, len(embeddings)):
        prev_context_emb = np.mean(embeddings[:i], axis=0)
        gains.append(1.0 - cosine(embeddings[i], prev_context_emb))
    return gains


def title_body_similarity(title, body):
    """
    제목-본문 의미 유사도.
    높을수록 AI-like.
    """

    if not isinstance(title, str) or len(title.strip()) == 0:
        return 0.0

    if not isinstance(body, str) or len(body.strip()) == 0:
        return 0.0

    title_emb = get_embedding(title)
    body_emb = get_embedding(body)

    return cosine(title_emb, body_emb)


def redundancy_from_embeddings(embeddings):
    if len(embeddings) < 2:
        return 0.0

    pairwise_sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            pairwise_sims.append(cosine(embeddings[i], embeddings[j]))

    return float(np.mean(pairwise_sims))


def get_structure_features(texts):
    """
    structure_signal의 모든 feature를 dict로 반환.
    """

    merged = _merge_texts(texts)
    title, body, article_paragraphs = split_article(merged)
    paragraphs = _collect_paragraphs(texts)

    if article_paragraphs and len(paragraphs) < len(article_paragraphs):
        paragraphs = article_paragraphs
    if not paragraphs and body:
        paragraphs = split_units(body)
    if not paragraphs and merged:
        paragraphs = [merged]

    if len(paragraphs) == 0:
        return {
            "combined_embedding": np.zeros(embedding_dim(), dtype=np.float32),
            "redundancy_score": 0.0,
            "title_body_similarity": 0.0,
        }

    embeddings = [get_embedding(p) for p in paragraphs]
    gains = paragraph_information_gains(embeddings)
    weights = np.asarray(gains, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-9)
    combined = np.sum(
        [weights[i] * embeddings[i] for i in range(len(embeddings))], axis=0
    )

    body_for_title = body if body else "\n\n".join(paragraphs)
    title_sim = title_body_similarity(title, body_for_title)

    return {
        "combined_embedding": combined.astype(np.float32),
        "redundancy_score": redundancy_from_embeddings(embeddings),
        "title_body_similarity": title_sim,
    }


def structure_combined_features(texts):
    """
    RoBERTa fc 입력 벡터 (여러 텍스트 조각을 하나의 문서로 합쳐 계산 — 학습용 synthetic 조합).

    [information-gain 가중 문단 embedding | redundancy | title_body_similarity ]
    """

    features = get_structure_features(texts)
    scalars = np.asarray(
        [
            features["redundancy_score"],
            features["title_body_similarity"],
        ],
        dtype=np.float32,
    )

    return np.concatenate([features["combined_embedding"], scalars])


def encode_texts_batched(texts, batch_size=32):
    if not texts:
        return np.zeros((0, embedding_dim()), dtype=np.float32)

    if model is None:
        init_structure_embedder(device or "cpu")

    dim = embedding_dim()
    out = np.zeros((len(texts), dim), dtype=np.float32)
    amp_on = _embed_amp_enabled(device)

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        chunk_clean = [t if isinstance(t, str) and t.strip() else "" for t in chunk]
        inputs = tokenizer(
            chunk_clean,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_EMBED_LENGTH,
        )
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        # with torch.inference_mode():
        with torch.no_grad():
            with _amp_autocast(enabled=amp_on):
                outputs = model(**inputs)
            embs = mean_pooling(outputs, inputs["attention_mask"]).float()
        out[start : start + len(chunk)] = embs.cpu().numpy()
    return out


def _resolve_article_paragraphs(article_text):
    merged = article_text if isinstance(article_text, str) else ""
    title, body, article_paragraphs = split_article(merged)
    paragraphs = split_units(merged)
    if article_paragraphs and len(paragraphs) < len(article_paragraphs):
        paragraphs = article_paragraphs
    if not paragraphs and body:
        paragraphs = split_units(body)
    if not paragraphs and merged.strip():
        paragraphs = [merged.strip()]
    body_for_title = body if body else "\n\n".join(paragraphs)
    return title, body_for_title, paragraphs


def redundancy_from_embeddings_np(embeddings):
    if len(embeddings) < 2:
        return 0.0
    pairwise_sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            pairwise_sims.append(cosine_np(embeddings[i], embeddings[j]))
    return float(np.mean(pairwise_sims))


def _structure_vector_from_embeddings(
    paragraph_embeddings, title, body_for_title, title_emb, body_emb
):
    dim = embedding_dim()
    if len(paragraph_embeddings) == 0:
        return np.zeros(dim + NUM_STRUCTURE_SCALARS, dtype=np.float32)

    embeddings = np.asarray(paragraph_embeddings, dtype=np.float32)
    gains = [1.0]
    for i in range(1, len(embeddings)):
        prev_mean = embeddings[:i].mean(axis=0)
        gains.append(1.0 - cosine_np(embeddings[i], prev_mean))

    weights = np.asarray(gains, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-9)
    combined = np.average(embeddings, axis=0, weights=weights)

    if (
        title_emb is not None
        and body_emb is not None
        and title.strip()
        and body_for_title.strip()
    ):
        title_sim = cosine_np(title_emb, body_emb)
    else:
        title_sim = 0.0

    scalars = np.asarray(
        [
            redundancy_from_embeddings_np(embeddings),
            title_sim,
        ],
        dtype=np.float32,
    )

    return np.concatenate([combined.astype(np.float32), scalars.astype(np.float32)])


def batch_structure_combined_features(article_texts, embed_batch_size=32):
    if not article_texts:
        return np.zeros((0, structure_feature_dim()), dtype=np.float32)

    metas = [_resolve_article_paragraphs(t) for t in article_texts]
    unique_texts = []
    text_to_idx = {}

    def idx_for(t):
        key = t if isinstance(t, str) else ""
        if key not in text_to_idx:
            text_to_idx[key] = len(unique_texts)
            unique_texts.append(key)
        return text_to_idx[key]

    article_maps = []
    for title, body_for_title, paragraphs in metas:
        para_idx = [idx_for(p) for p in paragraphs]
        title_idx = idx_for(title) if title.strip() else None
        body_idx = idx_for(body_for_title) if body_for_title.strip() else None
        article_maps.append((para_idx, title_idx, body_idx, title, body_for_title))

    all_embs = encode_texts_batched(unique_texts, batch_size=embed_batch_size)

    vectors = []
    for para_idx, title_idx, body_idx, title, body_for_title in article_maps:
        para_embs = all_embs[para_idx] if para_idx else np.zeros((0, all_embs.shape[1]))
        title_emb = all_embs[title_idx] if title_idx is not None else None
        body_emb = all_embs[body_idx] if body_idx is not None else None
        vectors.append(
            _structure_vector_from_embeddings(
                para_embs, title, body_for_title, title_emb, body_emb
            )
        )

    return np.stack(vectors, axis=0)
