import os
import json
import pickle as pkl

import torch
import numpy as np
import torch.utils.data as Data

from tqdm import tqdm
from parse import args
from methods.RoBERTa import RoBERTa

from methods.utils import my_dataset
from methods.utils import timeit, metrics


def append_train_history(save_path, record):
    """epoch마다 train_history.json에 append 후 전체 덮어쓰기."""
    if not save_path:
        return
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, "train_history.json")
    history = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def collate_text_label_batch(batch):
    """
    article_level DataLoader용 collate_fn.

    샘플 (text, label) 리스트를 받아 기사 텍스트 리스트와
    LongTensor 라벨 배치로 묶는다. 문장 단위가 아닌 기사 단위 배치에 사용.
    """
    texts = [b[0] for b in batch]
    labels = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
    return texts, labels


def score_margin(labels, scores, ai_label=1):
    """
    Val/test 점수 분리도: mean(score | AI) - mean(score | human).

    labels: 데이터셋 기준 0=human, 1=AI/llm.
    scores: pos_bit에 해당하는 클래스(AI) 확률.
    ai_label: AI로 간주할 라벨 값 (기본 1).
    한 클래스만 있으면 nan 반환.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    ai_mask = labels == ai_label
    human_mask = labels != ai_label
    if ai_mask.sum() == 0 or human_mask.sum() == 0:
        return float("nan")
    return float(scores[ai_mask].mean() - scores[human_mask].mean())


def get_supervised_model_prediction_EM(
    model, x, y, batch_size, DEVICE, pos_bit=1, pre_train=False
):
    """
    텍스트 x에 대해 모델의 AI 클래스(pos_bit) 확률을 배치 단위로 수집한다.

    article_level이면 collate_text_label_batch로 기사 텍스트 리스트를 배치화하고,
    그렇지 않으면 기본 collate를 사용한다.
    pre_train=True이면 model.pretrain_forward, 아니면 model.forward(RoBERTa: structure head만).

    Returns:
        data: 입력 텍스트 리스트(순서 유지).
        preds: 각 샘플의 pos_bit 클래스 확률 리스트.
    """
    train_dataset = my_dataset(x, y)

    with torch.no_grad():
        preds = []
        data = []

        loader_kw = dict(dataset=train_dataset, batch_size=batch_size, shuffle=False)
        if getattr(args, "article_level", False):
            loader_kw["collate_fn"] = collate_text_label_batch
        loader = Data.DataLoader(num_workers=4, pin_memory=True, **loader_kw)
        pbar = tqdm(loader, "evaluation")

        for batch_x, batch_y in pbar:
            batch_data = batch_x
            data += batch_x
            if pre_train:
                preds.extend(model.pretrain_forward(batch_data)[:, pos_bit].tolist())
            else:
                preds.extend(model(batch_data)[:, pos_bit].tolist())

    return data, preds


def eval_data(
    data,
    detector,
    batch_size,
    DEVICE,
    pos_bit,
    pre_train=False,
):
    """
    검증(val)·테스트 split에 대해 예측 후 metrics를 계산하고 로그를 출력한다.

    val은 체크포인트 선정용, test는 epoch마다 모니터링용( pre_train 플래그로 forward 경로 선택 가능).
    val_margin은 score_margin으로 AI/human 확률 차이를 계산한다.

    Returns:
        (acc_val, auc_val, tpr1_val, tpr2_val, tpr3_val,
         acc_test, acc_test, auc_test, tpr1_test, tpr2_test, tpr3_test, val_margin)
    """
    train_text = data["val"]["text"]
    train_label = data["val"]["label"]
    test_text = data["test"]["text"]
    test_label = data["test"]["label"]

    _, train_preds = get_supervised_model_prediction_EM(
        detector, train_text, train_label, batch_size, DEVICE, pos_bit
    )
    _, test_preds = get_supervised_model_prediction_EM(
        detector, test_text, test_label, batch_size, DEVICE, pos_bit, pre_train
    )

    y_train_pred_prob = train_preds
    y_train_pred = [round(_) for _ in y_train_pred_prob]
    y_train = train_label

    y_test_pred_prob = test_preds
    y_test_pred = [round(_) for _ in y_test_pred_prob]
    y_test = test_label

    train_res = metrics(y_train, y_train_pred, y_train_pred_prob)
    test_res = metrics(y_test, y_test_pred, y_test_pred_prob)
    acc_train, auc_train, pr_train, tpr1_train, tpr2_train, tpr3_train = train_res
    acc_test, auc_test, pr_test, tpr1_test, tpr2_test, tpr3_test = test_res

    val_margin = score_margin(y_train, y_train_pred_prob, ai_label=1)
    print(
        f"acc_val: {acc_train}, auc_val: {auc_train}, pr_val: {pr_train}, val_margin: {val_margin:.4f}, "
        f"tpr1_val: {tpr1_train}, tpr2_val: {tpr2_train}, tpr3_val: {tpr3_train}"
    )
    print(
        f"acc_test: {acc_test}, roc-auc_test: {auc_test}, pr-auc_test: {pr_test}, tpr1_test: {tpr1_test}, tpr2_test: {tpr2_test}, tpr3_test: {tpr3_test}"
    )
    return (
        acc_train,
        auc_train,
        tpr1_train,
        tpr2_train,
        tpr3_train,
        acc_test,
        acc_test,
        auc_test,
        tpr1_test,
        tpr2_test,
        tpr3_test,
        val_margin,
    )


@timeit
def supervised_news(
    data,
    model,
    batch_size,
    DEVICE,
    REG,
    pos_bit=1,
    finetune=False,
    num_labels=2,
    epochs=5,
    save_path=None,
    load_path="",
):
    """
    RoBERTa 기반 뉴스 AI 탐지 학습·평가 진입점 (train.py에서 호출).

    finetune=True: epoch마다 Optimize(loss1+reg*loss2) 후 val 기준 best 체크포인트 저장.
    finetune=False: load만 하고 학습 루프는 건너뜀.
    학습 종료 후 train/test 전체에 대해 예측하고 metrics·pkl·결과 dict를 반환한다.

    Args:
        data: {"train","val","test"} 각각 "text", "label" 키를 가진 dict.
        model: 로그용 모델 이름 문자열 (실제 클래스는 RoBERTa 고정).
        pos_bit: AI 클래스 인덱스; 0이면 학습 라벨을 반전(0↔1)해 detector와 맞춤.
        save_path: 체크포인트·train_preds.pkl 저장 디렉터리.
        load_path: 사전 학습 체크포인트 경로(파일 또는 디렉터리).

    Returns:
        name, predictions(test 확률), general(acc/auc/tpr 등 test 지표) dict.
    """
    detector = RoBERTa(DEVICE, save_path, load_path, pos_bit)

    if len(load_path) > 0:
        print("load model")
        detector.load()

    if finetune:
        train_label = data["train"]["label"]  # 데이터: 0=human, 1=AI
        if pos_bit == 0 and num_labels == 2:  # RoBERTa pos_bit과 라벨 정의 맞추기
            train_label = [1 if _ == 0 else 0 for _ in train_label]

        best_acc = -1
        patience = max(0, int(getattr(args, "patience", 0) or 0))
        save_every_epochs = max(0, int(getattr(args, "save_every_epochs", 10) or 0))
        epochs_without_improve = 0

        temp_reg = REG

        for i in range(epochs):
            detector.train()
            args.reg = temp_reg

            train_dataset = my_dataset(data["train"]["text"], train_label)
            _dl_kw = dict(batch_size=batch_size, shuffle=True)
            if getattr(args, "article_level", False):
                _dl_kw["collate_fn"] = collate_text_label_batch
            loader = Data.DataLoader(
                dataset=train_dataset, num_workers=4, pin_memory=True, **_dl_kw
            )

            pbar = tqdm(loader, desc=f"Fine-tuning: {i} epoch")
            for batch_x, batch_y in pbar:
                batch_y = batch_y.to(DEVICE)
                detector.Optimize(batch_x, batch_y)

            detector.eval()
            (
                acc_train,
                auc_train,
                tpr1_train,
                tpr2_train,
                tpr3_train,
                acc_test,
                acc_test,
                auc_test,
                tpr1_test,
                tpr2_test,
                tpr3_test,
                val_margin,
            ) = eval_data(data, detector, batch_size, DEVICE, pos_bit, pre_train=False)

            epoch_num = i + 1
            append_train_history(
                save_path,
                {
                    "epoch": epoch_num,
                    "acc_val": acc_train,
                    "auc_val": auc_train,
                    "val_margin": val_margin,
                    "tpr1_val": tpr1_train,
                    "tpr2_val": tpr2_train,
                    "tpr3_val": tpr3_train,
                    "acc_test": acc_test,
                    "auc_test": auc_test,
                    "tpr1_test": tpr1_test,
                    "tpr2_test": tpr2_test,
                    "tpr3_test": tpr3_test,
                    "saved_best": False,
                    "saved_periodic": False,
                },
            )

            if save_every_epochs > 0 and epoch_num % save_every_epochs == 0:
                print(f"save periodic checkpoint (epoch {epoch_num}/{epochs})")
                detector.save(epoch=epoch_num)
                if save_path and os.path.isfile(
                    os.path.join(save_path, "train_history.json")
                ):
                    with open(
                        os.path.join(save_path, "train_history.json"),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        hist = json.load(f)
                    if hist:
                        hist[-1]["saved_periodic"] = True
                        hist[-1]["periodic_checkpoint"] = detector.checkpoint_filename(
                            epoch=epoch_num
                        )
                    with open(
                        os.path.join(save_path, "train_history.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(hist, f, indent=2, ensure_ascii=False)

            save_best = False
            if auc_train >= best_acc:
                save_best = True
                best_acc = auc_train

            if save_best:
                epochs_without_improve = 0
                print(
                    f"save best (criterion='auc', "
                    f"val_margin={val_margin:.4f}, val_auc={auc_train:.4f})"
                )
                print("****************************\n")
                detector.save()
                if save_path and os.path.isfile(
                    os.path.join(save_path, "train_history.json")
                ):
                    with open(
                        os.path.join(save_path, "train_history.json"),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        hist = json.load(f)
                    if hist:
                        hist[-1]["saved_best"] = True
                        hist[-1]["best_checkpoint"] = detector.checkpoint_filename()
                    with open(
                        os.path.join(save_path, "train_history.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(hist, f, indent=2, ensure_ascii=False)

                best_acc = auc_train

            elif patience > 0:
                epochs_without_improve += 1
                print(
                    f"no improvement (patience {epochs_without_improve}/{patience}, "
                    f"criterion='auc')"
                )
                if epochs_without_improve >= patience:
                    print(
                        f"early stopping at epoch {i + 1}/{epochs} "
                        f"(patience={patience})"
                    )
                    break

    detector.eval()

    train_text = data["train"]["text"]
    train_label = data["train"]["label"]
    test_text = data["test"]["text"]
    test_label = data["test"]["label"]

    temp, train_preds = get_supervised_model_prediction_EM(
        detector,
        train_text,
        train_label,
        batch_size,
        DEVICE,
        pos_bit,
    )
    _, test_preds = get_supervised_model_prediction_EM(
        detector,
        test_text,
        test_label,
        batch_size,
        DEVICE,
        pos_bit,
    )
    maps = {x: y for x, y in zip(temp, train_preds)}
    pkl_dir = save_path if save_path else os.path.join("datasets")
    os.makedirs(pkl_dir, exist_ok=True)
    pkl_path = os.path.join(pkl_dir, f"{args.dataset}_{args.iter}_train_preds.pkl")
    with open(pkl_path, "wb") as file:
        pkl.dump(maps, file)
    print(f"saved train preds map: {pkl_path}")

    predictions = {
        "test": test_preds,
    }

    y_test_pred_prob = test_preds
    y_test_pred = [round(_) for _ in y_test_pred_prob]
    y_test = test_label

    test_res = metrics(y_test, y_test_pred, y_test_pred_prob)
    acc_train, precision_train, recall_train, f1_train, auc_train, pr_train = (
        0,
        0,
        0,
        0,
        0,
        0,
    )
    acc_test, auc_test, pr_test, tpr1_test, tpr2_test, tpr3_test = test_res
    print(
        f"{model} acc_test: {acc_test}, roc-auc_test: {auc_test}, pr-auc_test: {pr_test}, tpr1_test: {tpr1_test}, tpr2_test: {tpr2_test}, tpr3_test: {tpr3_test}"
    )

    del detector
    with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()

    return {
        "name": model,
        "predictions": predictions,
        "general": {
            "acc_train": acc_train,
            "precision_train": precision_train,
            "recall_train": recall_train,
            "f1_train": f1_train,
            "auc_train": auc_train,
            "pr_train": pr_train,
            "acc_test": acc_test,
            "tpr1_test": tpr1_test,
            "tpr2_test": tpr2_test,
            "tpr3_test": tpr3_test,
            "auc_test": auc_test,
        },
    }
