import pickle
import sys
import argparse
import random
import time
import os
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.dataset import MyDataSet


def _build_subset_dict(source_dict, keys):
    return {k: source_dict[k] for k in keys}


def _split_keys(all_keys, train_ratio=0.7, valid_ratio=0.2, seed=42):
    keys = list(all_keys)
    random.Random(seed).shuffle(keys)
    total = len(keys)
    train_size = int(total * train_ratio)
    valid_size = int(total * valid_ratio)
    train_keys = keys[:train_size]
    valid_keys = keys[train_size:train_size + valid_size]
    test_keys = keys[train_size + valid_size:]
    return train_keys, valid_keys, test_keys


def _build_dataset(emb_graph, emb_seq_feature, emb_label, emb_ppi_node_id, keys):
    return MyDataSet(
        emb_graph=_build_subset_dict(emb_graph, keys),
        emb_seq_feature=_build_subset_dict(emb_seq_feature, keys),
        emb_label=_build_subset_dict(emb_label, keys),
        emb_ppi_node_id=_build_subset_dict(emb_ppi_node_id, keys) if emb_ppi_node_id else {},
    )


def _save_dataset(path, dataset):
    with open(path, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


def _resolve_data_dir() -> Path:
    env_data_dir = os.environ.get("DATA_DIR")
    candidates = []
    if env_data_dir:
        candidates.append(Path(env_data_dir))
    candidates.extend([Path(__file__).resolve().parents[1], Path.cwd()])
    for candidate in candidates:
        if (candidate / "proceed_data").is_dir():
            return candidate
    return Path(env_data_dir) if env_data_dir else Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--namespace",
        type=str,
        default="all",
        choices=["all", "bp", "mf", "cc"],
        help="Chi chia namespace cần dùng; dùng --namespace bp để chạy nhanh hơn.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed cho train/valid/test split")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ghi đè dataset split đã tồn tại",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "train", "valid", "test"],
        help="Chỉ ghi một split (vd. train khi valid/test trên Kaggle vẫn OK)",
    )
    args = parser.parse_args()

    data_dir = _resolve_data_dir()
    PROC_DIR    = data_dir / "proceed_data"
    DIVIDED_DIR = data_dir / "divided_data"
    DIVIDED_DIR.mkdir(parents=True, exist_ok=True)

    ns_type_list = ['bp', 'mf', 'cc'] if args.namespace == "all" else [args.namespace]
    for ns_type in ns_type_list:
        start_ts = time.perf_counter()
        train_out = DIVIDED_DIR / f'{ns_type}_train_dataset'
        valid_out = DIVIDED_DIR / f'{ns_type}_valid_dataset'
        test_out = DIVIDED_DIR / f'{ns_type}_test_dataset'
        outs = {
            "train": train_out,
            "valid": valid_out,
            "test": test_out,
        }
        if args.only != "all":
            target = outs[args.only]
            if target.exists() and not args.force:
                print(f"skip {ns_type}: {target.name} exists (use --force)")
                continue
        elif (not args.force) and train_out.exists() and valid_out.exists() and test_out.exists():
            print(f"skip {ns_type}: output datasets already exist (use --force to rebuild)")
            continue

        print("divide", ns_type, "dataset", f"(only={args.only})" if args.only != "all" else "")
        with open(PROC_DIR / f'emb_graph_{ns_type}', 'rb') as f:
            emb_graph = pickle.load(f)
        with open(PROC_DIR / f'emb_seq_feature_{ns_type}', 'rb') as f:
            emb_seq_feature = pickle.load(f)
        with open(PROC_DIR / f'emb_label_{ns_type}', 'rb') as f:
            emb_label = pickle.load(f)

        emb_ppi_node_id = {}
        ppi_id_path = PROC_DIR / f'emb_ppi_node_id_{ns_type}'
        if ppi_id_path.exists():
            with open(ppi_id_path, 'rb') as f:
                emb_ppi_node_id = pickle.load(f)

        train_keys, valid_keys, test_keys = _split_keys(emb_graph.keys(), seed=args.seed)

        train_dataset = _build_dataset(emb_graph, emb_seq_feature, emb_label, emb_ppi_node_id, train_keys)
        valid_dataset = _build_dataset(emb_graph, emb_seq_feature, emb_label, emb_ppi_node_id, valid_keys)
        test_dataset = _build_dataset(emb_graph, emb_seq_feature, emb_label, emb_ppi_node_id, test_keys)

        if args.only in ("all", "train"):
            _save_dataset(train_out, train_dataset)
            print("train dataset size", len(train_dataset))
        if args.only in ("all", "valid"):
            _save_dataset(valid_out, valid_dataset)
            print("valid dataset size", len(valid_dataset))
        if args.only in ("all", "test"):
            _save_dataset(test_out, test_dataset)
            print("test dataset size", len(test_dataset))
        print(f"done {ns_type} in {time.perf_counter() - start_ts:.2f}s")
