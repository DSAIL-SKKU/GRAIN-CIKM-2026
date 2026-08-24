"""RoBERTa pretrained weight
- RADAR pretrained weights
"""

import os
import glob
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from parse import args
from datetime import datetime

from .structure import (
    batch_structure_combined_features,
    init_structure_embedder,
    structure_combined_features,
    structure_feature_dim,
    split_units,
)
from .utils import _amp_autocast, _amp_grad_scaler

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # deadlock prevention
import transformers

try:
    from torch.optim import AdamW
except ImportError:
    from transformers import AdamW


class RoBERTa(nn.Module):
    def __init__(self, DEVICE, save_path, load_path, pos_bit=1):
        """RoBERTa model init"""
        super(RoBERTa, self).__init__()
        model = "RoBERTa"
        model_path = model if "/" in model else "models/" + model

        self.pos_bit = pos_bit
        self.device = DEVICE
        self.save_path = save_path
        self.load_path = load_path
        self._run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.sent_detector = (
            transformers.AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=2, ignore_mismatched_sizes=True
            ).to(DEVICE)
        )

        self.max_token = 256
        self.reg = args.reg

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        init_structure_embedder(
            device_=DEVICE,
            model_=self.sent_detector.roberta,
            tokenizer_=self.tokenizer,
        )
        struct_input_dim = structure_feature_dim()

        self.fc = nn.Sequential(
            nn.Linear(struct_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        ).to(DEVICE)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.sent_detector.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.sent_detector.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.fc.named_parameters()],
                "weight_decay": 0.01,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=1e-6)
        self._amp_enabled = (
            str(DEVICE).startswith("cuda")
            and torch.cuda.is_available()
            and not getattr(args, "no_amp", False)
        )
        self.scaler = _amp_grad_scaler(enabled=self._amp_enabled)

    def _infer_structure_weight(self):
        """parse.args.infer_structure_weight: w*structure + (1-w)*unit (logit space)."""
        w = float(getattr(args, "infer_structure_weight", 0.7))
        return max(0.0, min(1.0, w))

    def _structure_batch_from_articles(self, x_list, embed_batch_size=None):
        """structure_batch_from_articles"""
        embed_bs = embed_batch_size or getattr(args, "embed_batch_size", 32)
        if not x_list:
            return torch.zeros(
                (0, structure_feature_dim()),
                dtype=torch.float32,
                device=self.device,
            )
        feats = batch_structure_combined_features(x_list, embed_batch_size=embed_bs)
        return torch.from_numpy(feats).to(
            self.device, dtype=torch.float32, non_blocking=True
        )

    def _collect_article_units(self, article_text):
        units = split_units(article_text)
        if not units:
            t = article_text.strip() if isinstance(article_text, str) else ""
            units = [t] if t else []
        return units[: args.article_max_units]

    def _article_unit_logits(self, x_list):
        """기사별 문단 sent_detector logit 평균."""
        n = len(x_list)
        if n == 0:
            return torch.zeros((0, 2), device=self.device), torch.zeros(
                0, device=self.device
            )

        unit_texts = []
        article_index = []
        for ai, article in enumerate(x_list):
            for unit in self._collect_article_units(article):
                unit_texts.append(unit)
                article_index.append(ai)

        out = torch.zeros(n, 2, device=self.device)
        counts = torch.zeros(n, device=self.device)
        if not unit_texts:
            return out, counts

        encoded_input = self.tokenizer(
            unit_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_token,
        ).to(self.device)

        with _amp_autocast(enabled=self._amp_enabled):
            unit_logits = self.sent_detector(**encoded_input).logits

        for ui, ai in enumerate(article_index):
            out[ai] += unit_logits[ui]
            counts[ai] += 1
        out = out / counts.clamp(min=1).unsqueeze(-1)
        return out, counts

    def _inference_logits(self, x_list, embed_batch_size=None):
        """
        추론 logit 결합: w*structure + (1-w)*unit_mean.
        w = args.infer_structure_weight (1=structure만, 0=unit만).
        """
        w = self._infer_structure_weight()
        x_list = list(x_list)

        if w <= 0.0:
            unit_logits, counts = self._article_unit_logits(x_list)
            if counts.sum() == 0:
                return torch.zeros(len(x_list), 2, device=self.device)
            return unit_logits

        structure_batch = self._structure_batch_from_articles(
            x_list, embed_batch_size=embed_batch_size
        )
        structure_logits = self.fc(structure_batch)

        if w >= 1.0:
            return structure_logits

        unit_logits, counts = self._article_unit_logits(x_list)
        if counts.sum() == 0:
            return structure_logits

        combined = w * structure_logits + (1.0 - w) * unit_logits
        no_units = counts == 0
        if no_units.any():
            combined[no_units] = structure_logits[no_units]

        return combined

    def forward(self, x, embed_batch_size=None):
        with torch.inference_mode():
            logits = self._inference_logits(list(x), embed_batch_size=embed_batch_size)
        return logits.softmax(-1)

    def _build_structure_batch(self, x, index_list, num_units):
        features = []
        for i in range(0, index_list.shape[0], num_units):
            selected_texts = [x[idx] for idx in index_list[i : i + num_units]]
            features.append(structure_combined_features(selected_texts))
        return torch.tensor(np.stack(features), dtype=torch.float32, device=self.device)

    def Optimize(self, x, y):
        self.optimizer.zero_grad(set_to_none=True)
        loss_fct = CrossEntropyLoss()

        x_list = list(x)
        unit_texts = []
        unit_labels = []
        y_long = y.long()

        if not self._structure_only_train():
            for bi in range(len(x_list)):
                article_text = x_list[bi]
                yi = int(y_long[bi].item())
                for unit in self._collect_article_units(article_text):
                    unit_texts.append(unit)
                    unit_labels.append(yi)

        with _amp_autocast(enabled=self._amp_enabled):
            structure_batch = self._structure_batch_from_articles(x_list)
            article_logits = self.fc(structure_batch)
            loss2 = loss_fct(article_logits, y_long)

            if self._structure_only_train():
                self.loss = loss2
            elif len(unit_texts) == 0:
                self.loss = loss2
            else:
                encoded_input = self.tokenizer(
                    unit_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_token,
                ).to(self.device)
                logits_units = self.sent_detector(**encoded_input).logits
                targets_units = torch.tensor(
                    unit_labels, dtype=torch.long, device=self.device
                )
                loss1 = loss_fct(logits_units, targets_units)
                self.loss = self._training_loss(loss1, loss2, has_units=True)

        if self._amp_enabled:
            self.scaler.scale(self.loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.loss.backward()
            self.optimizer.step()

    def _structure_only_train(self):
        return bool(getattr(args, "structure_only_train", False))

    def _training_loss(self, loss1, loss2, has_units):
        if self._structure_only_train():
            return loss2
        if not has_units:
            return loss2
        # reg: [0,1] mixing weight — 0=unit only, 1=structure only
        return (1.0 - args.reg) * loss1 + args.reg * loss2

    def checkpoint_filename(self, epoch=None):
        run_dataset = str(getattr(args, "dataset", "unknown")).replace("/", "_")
        run_epochs = getattr(args, "epochs", 0)
        run_batch_size = getattr(args, "batch_size", 0)
        reg_tag = "struct" if self._structure_only_train() else str(self.reg)
        base = f"{run_dataset}_{run_epochs}_{run_batch_size}_{reg_tag}"
        if epoch is not None:
            return f"{base}_epoch{int(epoch)}_{self._run_stamp}.pth"
        return f"{base}_{self._run_stamp}.pth"

    def _resolve_load_checkpoint(self):
        """load_path: .pth 파일 경로 또는 체크포인트가 있는 디렉터리."""
        lp = (self.load_path or "").strip()
        if not lp:
            raise ValueError("load_path is empty")

        if lp.endswith(".pth"):
            if os.path.isfile(lp):
                return lp
            raise FileNotFoundError(f"checkpoint not found: {lp}")

        if os.path.isdir(lp):
            files = glob.glob(os.path.join(lp, "*.pth"))
            if not files:
                raise FileNotFoundError(f"no .pth checkpoints in {lp}")
            return max(files, key=os.path.getmtime)

        raise FileNotFoundError(f"invalid load_path: {lp}")

    def save(self, epoch=None):
        os.makedirs(self.save_path, exist_ok=True)
        path = os.path.join(self.save_path, self.checkpoint_filename(epoch=epoch))
        torch.save(self.state_dict(), path)
        self._last_checkpoint_path = path
        print(f"saved checkpoint: {path}")

    def load(self):
        path = self._resolve_load_checkpoint()
        load_kw = {"map_location": self.device}
        try:
            state = torch.load(path, weights_only=True, **load_kw)
        except TypeError:
            state = torch.load(path, **load_kw)
        self.load_state_dict(state)
        print(f"loaded checkpoint: {path}")
