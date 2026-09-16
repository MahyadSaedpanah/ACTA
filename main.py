import argparse
import warnings

import sklearn.exceptions
import trainers


warnings.filterwarnings(
    "ignore",
    category=sklearn.exceptions.UndefinedMetricWarning,
)


parser = argparse.ArgumentParser()


# ========= Experiment naming =========
parser.add_argument(
    "--save_dir",
    default="logs",
    type=str,
    help="Directory containing all experiments",
)

parser.add_argument(
    "--experiment_description",
    default="ACTA",
    type=str,
)

parser.add_argument(
    "--run_description",
    default="ACTA",
    type=str,
)


# ========= Algorithm =========
parser.add_argument(
    "--da_method",
    default="ACTA",
    type=str,
)


# ========= Dataset / backbone =========
parser.add_argument(
    "--data_path",
    default="data",
    type=str,
    help="Root path containing dataset folders",
)

parser.add_argument(
    "--dataset",
    default="UCIHAR",
    type=str,
)

parser.add_argument(
    "--backbone",
    default="CNN",
    type=str,
)


# ========= Training =========
parser.add_argument(
    "--num_runs",
    default=5,
    type=int,
)

parser.add_argument(
    "--device",
    default="cpu",
    type=str,
)

parser.add_argument(
    "--source_epochs",
    default=50,
    type=int,
    help="Source-only pretraining epochs",
)

parser.add_argument(
    "--num_epochs",
    default=50,
    type=int,
    help="ACTA adaptation epochs",
)

parser.add_argument(
    "--bs",
    default=32,
    type=int,
)

parser.add_argument(
    "--lr",
    default=1e-3,
    type=float,
)

parser.add_argument(
    "--weight_decay",
    default=1e-4,
    type=float,
)

parser.add_argument(
    "--start",
    default=0,
    type=int,
)

parser.add_argument(
    "--end",
    default=None,
    type=int,
)

parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
)

parser.add_argument(
    "--num_workers",
    default=2,
    type=int,
)

parser.add_argument(
    "--shuffle",
    action="store_true",
)


# ========= ACTA =========
parser.add_argument(
    "--acta_ema",
    default=0.99,
    type=float,
)

parser.add_argument(
    "--selector_hid_dim",
    default=128,
    type=int,
)

parser.add_argument(
    "--disc_hid_dim",
    default=128,
    type=int,
)

parser.add_argument(
    "--selector_lr",
    default=1e-3,
    type=float,
)

parser.add_argument(
    "--disc_lr",
    default=1e-3,
    type=float,
)

parser.add_argument(
    "--lambda_adv",
    default=1.0,
    type=float,
)

parser.add_argument(
    "--lambda_dcg",
    default=1.0,
    type=float,
)


# ========= Phase =========
parser.add_argument(
    "--phase",
    default="train",
    type=str,
    choices=["train", "test"],
)

parser.add_argument(
    "--test_model_prefix",
    type=str,
)


# ========= Debug =========
parser.add_argument(
    "--debug",
    action="store_true",
)

args = parser.parse_args()


if args.debug:
    args.num_runs = 1
    args.source_epochs = 5
    args.num_epochs = 5
    args.start = 0
    args.end = 1


if __name__ == "__main__":
    trainer = trainers.da_trainer(args)

    if args.phase == "test":
        trainer.test()
    else:
        trainer.train()