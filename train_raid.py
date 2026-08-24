from parse import args

import os
import json
import random
import datetime
import dataloader_raid

import torch
from methods.supervised import supervised_news

START_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
START_TIME = datetime.datetime.now().strftime("%H-%M-%S-%f")

if __name__ == "__main__":

    random.seed(0)
    seeds = [random.randint(0, 100000000) for _ in range(100)]

    random.seed(seeds[args.iter])
    torch.manual_seed(seeds[args.iter])
    torch.cuda.manual_seed(seeds[args.iter])
    torch.cuda.manual_seed_all(seeds[args.iter])
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    DEVICE = args.DEVICE
    batch_size = args.batch_size
    epochs = args.epochs
    num_labels = args.num_labels
    outputs = []

    dataset_name = (
        args.dataset if args.dataset in ("RAID", "RAID-news") else "RAID-news"
    )
    print(
        f"Loading RAID dataset (domain={args.raid_domain}, detectLLM={args.detectLLM})..."
    )

    print(f"Loading dataset {args.dataset}...")
    print(
        f"reg={args.reg} "
        f"train_mode={'A3 structure-only' if getattr(args, 'structure_only_train', False) else 'full (loss1 + reg*loss2)'} "
        f"infer_structure_weight={args.infer_structure_weight} "
        f"(val/test logits = w*structure + (1-w)*unit_mean)"
    )
    data = dataloader_raid.load(name=dataset_name, detectLLM=args.detectLLM)

    base_model_name = args.base_model_name.replace("/", "_")
    SAVE_PATH = f"results/{START_DATE}/{dataset_name}-{START_TIME}"
    os.makedirs(SAVE_PATH, exist_ok=True)

    with open(os.path.join(SAVE_PATH, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    result = supervised_news(
        data,
        model="RoBERTa",
        batch_size=batch_size,
        DEVICE=DEVICE,
        REG=args.reg,
        pos_bit=1,
        finetune=args.finetune,
        num_labels=num_labels,
        epochs=epochs,
        save_path=SAVE_PATH,
        load_path=args.load_path,
    )
    outputs.append(result)

    print("Finish")
