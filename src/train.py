import os
import json
import random
import datetime

import torch

from parse import args
import dataloader
from methods.supervised import supervised_news


def set_seed(iter_idx: int) -> None:
    random.seed(0)
    seeds = [random.randint(0, 100000000) for _ in range(100)]
    seed = seeds[iter_idx]

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def main() -> None:
    set_seed(args.iter)

    start_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_time = datetime.datetime.now().strftime("%H-%M-%S-%f")
    save_path = f"results/{start_date}/{args.dataset}-{start_time}"
    os.makedirs(save_path, exist_ok=True)

    train_mode = (
        "structure-only" if args.structure_only_train else "full (loss1 + reg*loss2)"
    )
    print(f"Loading dataset {args.dataset}...")
    print(
        f"reg={args.reg} "
        f"train_mode={train_mode} "
        f"infer_structure_weight={args.infer_structure_weight} "
        f"(val/test logits = w*structure + (1-w)*unit_mean)"
    )

    data = dataloader.load(args.dataset)
    with open(os.path.join(save_path, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    supervised_news(
        data,
        model="RoBERTa",
        batch_size=args.batch_size,
        DEVICE=args.DEVICE,
        REG=args.reg,
        pos_bit=args.pos_bit,
        finetune=args.finetune,
        num_labels=args.num_labels,
        epochs=args.epochs,
        save_path=save_path,
        load_path=args.load_path,
    )
    print("Finish")


if __name__ == "__main__":
    main()
