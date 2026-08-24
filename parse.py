import argparse

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--dataset", type=str, default="MultiDomainXSum")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--embed_batch_size",
    type=int,
    default=32,
    help="structure φ(x) 임베딩 배치 크기 (Optimize/forward).",
)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument(
    "--patience",
    type=int,
    default=0,
    help="Finetune early stopping: val 기준 개선 없을 때 중단까지 epoch 수. 0=끔.",
)
parser.add_argument(
    "--save_every_epochs",
    type=int,
    default=10,
    help="Finetune 중 N epoch마다 체크포인트 저장. 0=끔.",
)
parser.add_argument("--num_labels", type=int, default=2)
parser.add_argument("--base_model_name", type=str, default="gpt2-medium")
parser.add_argument("--cache_dir", type=str, default="save_models")
parser.add_argument(
    "--RoBERTa_pretrained",
    type=str,
    default="",
    help="RoBERTa 사전학습 폴더 (비우면 src/save_models/RoBERTa 자동 탐색).",
)
parser.add_argument("--DEVICE", type=str, default="cuda")
parser.add_argument(
    "--no_amp",
    action="store_true",
    help="Disable CUDA mixed-precision (autocast) during RoBERTa training.",
)

parser.add_argument("--finetune", action="store_true")
parser.add_argument("--load_path", type=str, default="")
parser.add_argument(
    "--reg",
    type=float,
    default=0.9,
    help="학습 loss mixing weight [0,1]: 0=unit only, 1=structure only.",
)
parser.add_argument(
    "--structure_only_train",
    action="store_true",
    help="Ablation A3: loss2(structure fc)만으로 학습. unit loss1은 건너뜀.",
)

parser.add_argument(
    "--article_level",
    action="store_true",
    help="뉴스 등: 샘플 1개 = 기사 전체. structure는 기사 내 문단 담화, 평가는 기사 단위.",
)
parser.add_argument(
    "--article_max_units",
    type=int,
    default=32,
    help="기사당 sent_detector loss에 쓸 최대 문단·문장 수(split_units 순서).",
)
parser.add_argument(
    "--infer_structure_weight",
    type=float,
    default=1.0,
    help="추론 logit 결합: w*structure+(1-w)*unit_mean. 1=structure만.",
)
parser.add_argument(
    "--checkpoint_criterion",
    type=str,
    choices=["auc"],
    default=None,
    help="Finetune 체크포인트: val ROC-AUC(auc)",
)
parser.add_argument(
    "--margin",
    type=float,
    default=1.0,
    help="logit margin 목표: AI는 (logit_ai-logit_human)>=margin, human은 <=-margin.",
)
parser.add_argument(
    "--pos_bit",
    type=int,
    default=1,
    help="softmax 출력에서 AI(양성) 클래스 인덱스. 1=P(AI), 0=P(human).",
)
parser.add_argument("--iter", type=int, default=0, help="random seed index")
parser.add_argument(
    "--xsum_train_path",
    type=str,
    default="../src/datasets/multi_domains_xsum_train.json",
    help="MultiDomainXSum train JSON (text, label human|llm).",
)
parser.add_argument(
    "--xsum_test_path",
    type=str,
    default="../src/datasets/multi_domains_xsum_test.json",
    help="MultiDomainXSum test JSON.",
)
parser.add_argument(
    "--kor_train_path",
    type=str,
    default="../datasets/kor_aigen/train.json",
    help="kor_aigen train JSON (text, label 0=human/1=AI).",
)
parser.add_argument(
    "--kor_valid_path",
    type=str,
    default="../datasets/kor_aigen/valid.json",
    help="kor_aigen valid JSON.",
)
parser.add_argument(
    "--kor_test_path",
    type=str,
    default="../datasets/kor_aigen/test.json",
    help="kor_aigen test JSON.",
)
parser.add_argument(
    "--val_fraction",
    type=float,
    default=0.1,
    help="train JSON에서 human·llm 각각 검증으로 떼는 비율.",
)

# ── RAID 전용 인자 ──────────────────────────────────────────────
parser.add_argument(
    "--detectLLM",
    type=str,
    default="all",
    help="RAID: 탐지 대상 LLM 모델명 또는 all/non_human (기본: all).",
)
parser.add_argument(
    "--raid_domain",
    type=str,
    default="news",
    help="RAID: 필터링할 도메인 (기본: news).",
)
parser.add_argument(
    "--raid_include_adversarial",
    action="store_true",
    help="RAID: adversarial attack 샘플 포함 여부.",
)
parser.add_argument(
    "--split_cache",
    type=str,
    default="",
    help="RAID: train/val/test split 캐시 JSON 경로. 비우면 data_cache/ 아래 자동 생성.",
)

# eval.py 등 다른 스크립트 전용 CLI 플래그는 무시 (import 시 재파싱 방지)
args, _ = parser.parse_known_args()
