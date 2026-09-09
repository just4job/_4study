import argparse
import logging
import os
import pickle
import sys
import warnings
from functools import partial
from typing import Any
from pathlib import Path

# Patch DGL on disk before import (Kaggle Py3.12 / torchdata break)
from model.dgl_patch import ensure_dgl_importable

ensure_dgl_importable(verbose=False)

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from data_processing.divide_data import MyDataSet
from model.evaluation import cacul_aupr, calculate_performance, roc_auc_flat
from model.network import PPIEncoder, SAGNetworkHierarchical

warnings.filterwarnings("ignore")


def _as_flat_float_tensor(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1)


def _fix_dim_1d(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    cur = int(x.shape[0])
    if cur == target_dim:
        return x
    if cur > target_dim:
        return x[:target_dim]
    out = torch.zeros(target_dim, dtype=x.dtype)
    out[:cur] = x
    return out


def _train_collate(samples, seq_dim: int, label_dim: int):
    pids, graphs, labels, seq_feats, ppi_node_ids = zip(*samples)

    fixed_seq_feats = [_fix_dim_1d(_as_flat_float_tensor(seq), seq_dim) for seq in seq_feats]
    fixed_labels = [_fix_dim_1d(_as_flat_float_tensor(lbl), label_dim) for lbl in labels]

    return (
        list(pids),
        dgl.batch(list(graphs)),
        torch.stack(fixed_labels, dim=0),
        torch.stack(fixed_seq_feats, dim=0),
        torch.as_tensor(ppi_node_ids, dtype=torch.long),
    )


def _register_pickle_classes() -> None:
    """Pickle from `python divide_data.py` references __main__.MyDataSet."""
    import __main__

    __main__.MyDataSet = MyDataSet


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if name == "MyDataSet" and module in {"__main__", "data_processing.divide_data"}:
            return MyDataSet
        return super().find_class(module, name)


def _load_pickle(path: str):
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing dataset pickle: {file_path}")
    if file_path.stat().st_size == 0:
        raise RuntimeError(
            f"Dataset pickle is empty: {file_path}. Rebuild divided_data or re-run kaggle_link_data.py."
        )
    try:
        with open(file_path, "rb") as handle:
            return _CompatUnpickler(handle).load()
    except EOFError as exc:
        raise RuntimeError(
            f"Dataset pickle is truncated or corrupted: {file_path}. Rebuild divided_data or re-run kaggle_link_data.py."
        ) from exc
Thresholds = [x / 100 for x in range(1, 100)]


def _collect_ppi_node_ids(dataset: MyDataSet) -> set[int]:
    """Tập ppi_node_id (>=0) của các protein trong 1 dataset split."""
    return {nid for nid in dataset.ppi_node_id.values() if nid is not None and nid >= 0}


def build_train_only_ppi_graph(
    ppi_graph: "dgl.DGLGraph", hidden_node_ids: set[int]
) -> "dgl.DGLGraph":
    """Ẩn (mask) mọi cạnh PPI có ít nhất 1 đầu là node valid/test — chống leak khi train.

    ppi_graph_global là 1 đồ thị PPI TOÀN CỤC dựng từ toàn bộ protein (không phân biệt
    train/valid/test). PPIEncoder (GraphSAGE) mặc định encode nguyên đồ thị này mỗi
    epoch, nên embedding của 1 protein "train" có thể nhận message lan truyền từ
    hàng xóm PPI đang thuộc tập valid/test — rò rỉ gián tiếp thông tin (sequence
    feature) của valid/test vào lúc train (xem README mục "PPI leakage guard").

    Hàm này trả về 1 bản sao ppi_graph nhưng CẮT mọi cạnh chạm tới `hidden_node_ids`
    (thường là ppi_node_id của protein thuộc valid+test). Số node và node feature
    giữ nguyên (node valid/test vẫn tồn tại nhưng bị cô lập, không có cạnh) — vì vậy
    ppi_node_id lookup không đổi và graph vẫn tương thích với phần còn lại của code.
    Dùng graph này CHỈ khi train; lúc validate/test luôn dùng ppi_graph_global gốc
    (đầy đủ cạnh) để đánh giá đúng khả năng model dùng PPI thật.
    """
    if not hidden_node_ids:
        return ppi_graph

    num_nodes = ppi_graph.num_nodes()
    device = ppi_graph.device
    hidden_idx = torch.as_tensor(sorted(hidden_node_ids), dtype=torch.long, device=device)
    hidden_idx = hidden_idx[hidden_idx < num_nodes]
    hidden_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    hidden_mask[hidden_idx] = True

    src, dst = ppi_graph.edges()
    keep_eids = (~(hidden_mask[src] | hidden_mask[dst])).nonzero(as_tuple=True)[0]
    masked_graph = dgl.edge_subgraph(ppi_graph, keep_eids, relabel_nodes=False)
    masked_graph.ndata["feat"] = ppi_graph.ndata["feat"]
    return masked_graph


def _resolve_data_dir() -> str:
    env_data_dir = os.environ.get("DATA_DIR")
    candidates = []
    if env_data_dir:
        candidates.append(Path(env_data_dir))

    script_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            script_dir,
            Path.cwd(),
            script_dir.parent,
        ]
    )

    for candidate in candidates:
        if (candidate / "divided_data").is_dir() and (candidate / "proceed_data").is_dir():
            return str(candidate)

    return str(Path(env_data_dir) if env_data_dir else script_dir)


def _ckpt_path(data_dir: str, args: argparse.Namespace, tag: str) -> str:
    os.makedirs(os.path.join(data_dir, "save_models"), exist_ok=True)
    return os.path.join(
        data_dir,
        "save_models",
        f"{tag}_{args.branch}_{args.batch_size}_{args.learningrate}_{args.dropout}.pkl",
    )


def create_logger(branch_name: str, data_dir: str) -> logging.Logger:
    log_dir = os.path.join(data_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "save_models"), exist_ok=True)
    logger = logging.getLogger(branch_name)
    if logger.handlers:
        return logger
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=os.path.join(log_dir, f"{branch_name}.log"))
    logger.setLevel(logging.DEBUG)
    handler1.setLevel(logging.ERROR)
    handler2.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def resolve_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    # Kaggle/Linux: CUDA sẵn có. Windows: cần DGL_CUDA=1 nếu đã cài DGL CUDA wheel.
    on_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    dgl_cuda = os.environ.get("DGL_CUDA", "1" if on_kaggle else "0") == "1"
    if dgl_cuda or on_kaggle:
        return torch.device("cuda:0")
    return torch.device("cpu")


def _argv_has(*names: str) -> bool:
    return any(n in sys.argv for n in names)


def apply_kaggle_preset(args: argparse.Namespace) -> None:
    """Preset ~30 phút/nhánh trên Kaggle T4 (validate 1 lần ở epoch cuối)."""
    args.hid_dim = 256
    args.num_convs = 3
    args.ppi_out_dim = 96
    args.num_workers = 0  # tránh OOM trên Kaggle 30GB khi load pickle lớn
    args.amp = True
    args.cache_ppi = True
    global Thresholds
    Thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]  # 5 ngưỡng — giảm thời gian tính F-max

    if args.branch == "bp":
        args.batch_size = 96
    else:
        args.batch_size = 96
    # Kaggle preset validates only at the end; keep user overrides for other knobs
    # but do not let validate_every reintroduce long pauses every few epochs.
    args.validate_every = args.epochs


def apply_baseline_parity_preset(args: argparse.Namespace) -> None:
    """Train protocol khớp baseline Struct2GO (Table 1); kiến trúc vẫn có PPI."""
    from scripts.baseline_config import (
        BASELINE_BATCH_SIZE,
        BASELINE_LR,
        BRANCH_BASELINE_DROPOUT,
    )

    args.epochs = 20
    args.batch_size = BASELINE_BATCH_SIZE
    args.learningrate = BASELINE_LR
    args.dropout = BRANCH_BASELINE_DROPOUT.get(args.branch, 0.3)
    args.hid_dim = 512
    args.num_convs = 6
    args.pool_ratio = 0.75
    args.ppi_out_dim = 256
    args.validate_every = 4
    args.num_workers = 4
    args.amp = False
    global Thresholds
    Thresholds = [x / 100 for x in range(1, 100)]


def _restore_cli_overrides(
    args: argparse.Namespace, overrides: dict[str, Any], extra_keys: tuple[str, ...] = ()
) -> None:
    """Giữ giá trị CLI nếu user truyền flag tương ứng."""
    key_to_flags = {
        "batch_size": ("-batch_size", "--batch_size"),
        "epochs": ("-epochs", "--epochs"),
        "dropout": ("-dropout", "--dropout"),
        "learningrate": ("-learningrate", "--learningrate"),
        "hid_dim": ("-hid_dim", "--hid_dim"),
        "num_convs": ("-num_convs", "--num_convs"),
        "pool_ratio": ("-pool_ratio", "--pool_ratio"),
        "ppi_out_dim": ("-ppi_out_dim", "--ppi_out_dim"),
        "validate_every": ("-validate_every", "--validate_every"),
    }
    for key in extra_keys:
        flags = key_to_flags.get(key)
        if flags and _argv_has(*flags) and key in overrides:
            setattr(args, key, overrides[key])


def labels_to_device(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    labels = torch.squeeze(labels)
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(0)
    return labels.to(device).float()


def _estimate_pos_weight(train_dataset, label_dim: int, max_samples: int = 3000, cap: float = 50.0) -> float:
    """Tỷ lệ neg/pos trên train — giúp AUPR (term hiếm) cho concat / no-ppi."""
    pos = 0.0
    total = 0.0
    n = min(len(train_dataset), max_samples)
    for i in range(n):
        sample = train_dataset[i]
        lbl = sample[2] if len(sample) > 2 else sample[1]
        arr = np.asarray(lbl, dtype=np.float64).reshape(-1)[:label_dim]
        pos += float(arr.sum())
        total += float(arr.size)
    if pos <= 0:
        return 1.0
    return min((total - pos) / pos, cap)


def _ckpt_selection_score(fmax: float, aupr: float, metric: str) -> float:
    if metric == "aupr":
        return aupr
    if metric == "combo":
        return fmax + 0.35 * aupr
    return fmax


def _resolve_ckpt_metric(args: argparse.Namespace) -> str:
    if args.ckpt_metric != "auto":
        return args.ckpt_metric
    if args.fusion_mode == "concat" or not args.use_ppi:
        return "combo"
    return "fmax"


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-batch_size", "--batch_size", type=int, default=64)
    parser.add_argument("-learningrate", "--learningrate", type=float, default=1e-4)
    parser.add_argument("-dropout", "--dropout", type=float, default=0.1)
    parser.add_argument("-branch", "--branch", type=str, default="mf", choices=["bp", "mf", "cc"])
    parser.add_argument("-labels_num", "--labels_num", type=int, default=328)
    parser.add_argument("-epochs", "--epochs", type=int, default=3)
    parser.add_argument("-hid_dim", "--hid_dim", type=int, default=256)
    parser.add_argument("-num_convs", "--num_convs", type=int, default=3)
    parser.add_argument("-pool_ratio", "--pool_ratio", type=float, default=0.5)
    parser.add_argument("-seq_dim", "--seq_dim", type=int, default=1024)
    parser.add_argument("-ppi_out_dim", "--ppi_out_dim", type=int, default=256)
    parser.set_defaults(use_ppi=True, baseline_parity=True, fusion_mode="attention")
    parser.add_argument(
        "--no-ppi",
        dest="use_ppi",
        action="store_false",
        help="Tắt nhánh PPI (GraphSAGE trên mạng PPI)",
    )
    parser.add_argument(
        "--fusion",
        dest="fusion_mode",
        choices=["attention", "concat"],
        default="attention",
        help="Cách gộp struct/seq/ppi: attention (cross-attn) hoặc concat",
    )
    parser.add_argument(
        "--no-baseline-parity",
        dest="baseline_parity",
        action="store_false",
        help="Không dùng preset hyperparameter baseline (20 epoch, hid=512, …)",
    )
    parser.add_argument("-num_workers", "--num_workers", type=int, default=4)
    parser.add_argument("-validate_every", "--validate_every", type=int, default=4,
                        help="Validate mỗi N epoch (1 = mỗi epoch)")
    parser.add_argument("--amp", action="store_true", help="Mixed precision (FP16) — khuyến nghị trên T4")
    parser.add_argument("--no_cache_ppi", dest="cache_ppi", action="store_false",
                        help="Tắt cache PPI embedding mỗi epoch")
    parser.set_defaults(cache_ppi=True)
    parser.add_argument(
        "--no_ppi_leakage_guard",
        dest="ppi_leakage_guard",
        action="store_false",
        help=(
            "Tắt cơ chế ẩn cạnh PPI nối tới valid/test khi train (bán-inductive hoá PPI). "
            "Mặc định BẬT để chống leak; chỉ tắt khi cố tình muốn tái tạo hành vi "
            "transductive cũ (vd. để so sánh/ablation)."
        ),
    )
    parser.set_defaults(ppi_leakage_guard=True)
    parser.add_argument("--cpu", action="store_true", help="Bắt buộc train trên CPU")
    parser.add_argument(
        "--pos-weight",
        action="store_true",
        help="BCE pos_weight từ train set (cải thiện AUPR — khuyến nghị cho concat / no-ppi)",
    )
    parser.add_argument(
        "--ckpt-metric",
        choices=["auto", "fmax", "aupr", "combo"],
        default="auto",
        help="Chọn checkpoint: auto=combo cho concat/no-ppi, fmax cho PPI+attention",
    )
    parser.add_argument("--kaggle", action="store_true",
                        help="Preset T4 ~30p/nhánh: mf/cc 5 epoch, bp 4 epoch, hid=256, amp")
    parser.add_argument(
        "--baseline-parity",
        dest="baseline_parity",
        action="store_true",
        help="Preset train khớp baseline paper (mặc định bật; dùng --no-baseline-parity để tắt)",
    )
    args = parser.parse_args()

    cli_overrides = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "learningrate": args.learningrate,
        "hid_dim": args.hid_dim,
        "num_convs": args.num_convs,
        "pool_ratio": args.pool_ratio,
        "ppi_out_dim": args.ppi_out_dim,
        "validate_every": args.validate_every,
    }

    print("train_Struct2GO2: parsing args done, applying presets...", flush=True)

    if args.baseline_parity:
        apply_baseline_parity_preset(args)
        _restore_cli_overrides(
            args,
            cli_overrides,
            (
                "batch_size", "epochs", "dropout", "learningrate", "hid_dim", "num_convs",
                "pool_ratio", "ppi_out_dim", "validate_every",
            ),
        )
    elif args.kaggle:
        args.baseline_parity = False
        apply_kaggle_preset(args)
        _restore_cli_overrides(
            args,
            cli_overrides,
            (
                "batch_size",
                "epochs",
                "dropout",
                "learningrate",
                "hid_dim",
                "num_convs",
                "validate_every",
            ),
        )
        if not _argv_has("-validate_every", "--validate_every"):
            args.validate_every = args.epochs

    device = resolve_device(force_cpu=args.cpu)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    data_dir = _resolve_data_dir()
    train_data_path = f"{data_dir}/divided_data/{args.branch}_train_dataset"
    valid_data_path = f"{data_dir}/divided_data/{args.branch}_valid_dataset"
    test_data_path = f"{data_dir}/divided_data/{args.branch}_test_dataset"
    label_network_path = f"{data_dir}/proceed_data/label_{args.branch}_network"
    ppi_graph_path = f"{data_dir}/proceed_data/ppi_graph_global"

    logger = create_logger(args.branch, data_dir)
    ckpt_metric = _resolve_ckpt_metric(args)
    logger.info(
        f"device={device}, amp={args.amp}, cache_ppi={args.cache_ppi}, "
        f"kaggle={args.kaggle}, baseline_parity={args.baseline_parity}, "
        f"use_ppi={args.use_ppi}, fusion_mode={args.fusion_mode}, "
        f"ppi_leakage_guard={args.ppi_leakage_guard}, "
        f"ckpt_metric={ckpt_metric}, pos_weight={args.pos_weight}"
    )
    logger.info(
        f"epochs={args.epochs}, batch_size={args.batch_size}, dropout={args.dropout}, "
        f"lr={args.learningrate}, validate_every={args.validate_every}, "
        f"hid={args.hid_dim}, convs={args.num_convs}, pool_ratio={args.pool_ratio}"
    )
    logger.info(f"data_dir={data_dir}, cwd={os.getcwd()}")

    _register_pickle_classes()
    print(f"Loading train pickle: {train_data_path} ...", flush=True)
    train_dataset = _load_pickle(train_data_path)
    print(f"  train OK, n={len(train_dataset)}", flush=True)
    print(f"Loading valid pickle: {valid_data_path} ...", flush=True)
    valid_dataset = _load_pickle(valid_data_path)
    print(f"  valid OK, n={len(valid_dataset)}", flush=True)
    label_network = _load_pickle(label_network_path)
    label_network = label_network.to(device)

    ppi_graph = None
    train_ppi_graph = None
    ppi_feat_dim = args.seq_dim
    if args.use_ppi:
        print(f"Loading PPI graph: {ppi_graph_path} ...", flush=True)
        ppi_graph = _load_pickle(ppi_graph_path)
        ppi_graph = ppi_graph.to(device)
        print("  PPI graph OK", flush=True)
        ppi_feat_dim = int(ppi_graph.ndata["feat"].shape[1])

        train_ppi_graph = ppi_graph
        if args.ppi_leakage_guard:
            hidden_ids = _collect_ppi_node_ids(valid_dataset)
            if Path(test_data_path).is_file():
                test_dataset_for_mask = _load_pickle(test_data_path)
                hidden_ids |= _collect_ppi_node_ids(test_dataset_for_mask)
                del test_dataset_for_mask
            else:
                logger.warning(
                    f"[ppi-leak-guard] Không tìm thấy {test_data_path} — chỉ ẩn được node "
                    "valid, chưa chắc chắn ẩn hết node test. Chạy divide_data.py để có "
                    "test_dataset đầy đủ."
                )
            train_ppi_graph = build_train_only_ppi_graph(ppi_graph, hidden_ids)
            logger.info(
                f"[ppi-leak-guard] train-only PPI subgraph: giữ {train_ppi_graph.num_edges():,}/"
                f"{ppi_graph.num_edges():,} cạnh, ẩn {len(hidden_ids):,} node valid/test "
                "(dùng --no_ppi_leakage_guard để tắt)"
            )
        else:
            logger.warning(
                "[ppi-leak-guard] TẮT (--no_ppi_leakage_guard) — PPIEncoder sẽ thấy toàn bộ "
                "PPI graph (kể cả cạnh tới valid/test) trong lúc train. Chỉ dùng để so sánh/ablation."
            )

    sample_label = train_dataset[0][2]
    detected_labels = int(np.asarray(sample_label).reshape(-1).shape[0])
    network_labels = int(label_network.num_nodes())
    if _argv_has("-labels_num", "--labels_num"):
        labels_num = args.labels_num
        print(f"[INFO] Using CLI labels_num={labels_num}")
    elif detected_labels != network_labels:
        print(
            f"[WARN] label dim in dataset={detected_labels}, label_network nodes={network_labels}; "
            f"using labels_num={detected_labels} (dataset wins — graph may list extra GO nodes)"
        )
        labels_num = detected_labels
    elif detected_labels != args.labels_num:
        print(f"[INFO] Auto-detected labels_num = {detected_labels} (override CLI {args.labels_num})")
        labels_num = detected_labels
    else:
        labels_num = args.labels_num

    sample_seq = np.asarray(train_dataset[0][3]).reshape(-1)
    args.seq_dim = int(sample_seq.shape[0])
    if args.use_ppi and ppi_feat_dim != args.seq_dim:
        print(f"[INFO] ppi_in_dim={ppi_feat_dim} (from ppi_graph.ndata['feat'])")

    loader_kw = dict(
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True

    collate_fn = partial(_train_collate, seq_dim=args.seq_dim, label_dim=labels_num)

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, collate_fn=collate_fn, **loader_kw)
    valid_dataloader = DataLoader(dataset=valid_dataset, shuffle=False, collate_fn=collate_fn, **loader_kw)

    model = SAGNetworkHierarchical(
        56,
        args.hid_dim,
        labels_num,
        num_convs=args.num_convs,
        pool_ratio=args.pool_ratio,
        dropout=args.dropout,
        seq_dim=args.seq_dim,
        ppi_in_dim=ppi_feat_dim,
        ppi_hid_dim=args.hid_dim,
        ppi_out_dim=args.ppi_out_dim,
        use_ppi=args.use_ppi,
        fusion_mode=args.fusion_mode,
    ).to(device)

    # Neighbor index cho cross-attention (fusion_mode="attention") phải khớp với đồ thị
    # đang dùng ở từng thời điểm — model chỉ cache 1 index nội bộ (theo lệnh gọi đầu
    # tiên), nên khi train dùng train_ppi_graph còn valid/test dùng ppi_graph gốc, ta
    # tự dựng & truyền tay 2 bảng riêng để tránh dùng nhầm index của đồ thị kia.
    train_ppi_neighbor_index = None
    full_ppi_neighbor_index = None
    if args.use_ppi and args.fusion_mode == "attention":
        train_ppi_neighbor_index = PPIEncoder.build_neighbor_index(
            train_ppi_graph, model.ppi_max_neighbors
        )
        if train_ppi_graph is ppi_graph:
            full_ppi_neighbor_index = train_ppi_neighbor_index
        else:
            full_ppi_neighbor_index = PPIEncoder.build_neighbor_index(
                ppi_graph, model.ppi_max_neighbors
            )

    total_steps = args.epochs * max(len(train_dataloader), 1)
    if args.baseline_parity:
        optimizer = optim.Adam(model.parameters(), lr=args.learningrate)
        warmup_steps = min(100, max(1, total_steps // 20))
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.learningrate, weight_decay=1e-4)
        warmup_steps = min(200, total_steps // 10)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    if args.pos_weight:
        pw = _estimate_pos_weight(train_dataset, labels_num)
        pos_weight = torch.full((labels_num,), pw, device=device, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info(f"pos_weight={pw:.2f} (neg/pos trên train, cap=50)")
    else:
        criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and args.amp)

    best_fscore = 0.0
    best_aupr = 0.0
    best_ckpt_score = -1.0
    best_scores = []
    best_score_dict = {}

    logger.info("#########" + args.branch + "###########")
    logger.info("########start training###########")

    for epoch in range(args.epochs):
        print("epoch:", epoch)
        logger.info("epoch: " + str(epoch))
        model.train()

        ppi_node_emb = None
        if args.use_ppi and args.cache_ppi:
            # train_ppi_graph = full graph khi guard tắt hoặc use_ppi=False không tới đây.
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda and args.amp):
                ppi_node_emb = model.encode_ppi_nodes(train_ppi_graph).detach()

        train_loss = 0.0
        for i, (_, graphs, labels, seq_feats, ppi_node_ids) in enumerate(
            tqdm(train_dataloader, desc=f"train e{epoch}")
        ):
            graphs = graphs.to(device, non_blocking=use_cuda)
            seq_feats = seq_feats.to(device, non_blocking=use_cuda)
            labels = labels_to_device(labels, device)
            ppi_node_ids = ppi_node_ids.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_cuda and args.amp):
                logits = model(
                    graphs,
                    seq_feats,
                    label_network,
                    ppi_graph=train_ppi_graph if args.use_ppi else None,
                    ppi_node_ids=ppi_node_ids if args.use_ppi else None,
                    ppi_node_emb=ppi_node_emb,
                    ppi_neighbor_index=train_ppi_neighbor_index if args.use_ppi else None,
                )
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            train_loss += loss.item()
            if i % 30 == 29:
                logger.info(
                    f"Epoch {epoch}/{args.epochs}, step {i}/{len(train_dataloader)}, loss={loss.item():.4f}"
                )

        run_valid = (epoch + 1) % args.validate_every == 0 or epoch == args.epochs - 1
        if not run_valid:
            continue

        model.eval()
        print("validating")
        logger.info("validating")
        valid_loss = 0.0
        pred, actual = [], []

        if args.use_ppi and args.cache_ppi:
            # Validate LUÔN dùng ppi_graph gốc (đầy đủ cạnh) — khác train_ppi_graph ở trên.
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda and args.amp):
                ppi_node_emb = model.encode_ppi_nodes(ppi_graph).detach()

        with torch.no_grad():
            for i, (_, graphs, labels, seq_feats, ppi_node_ids) in enumerate(
                tqdm(valid_dataloader, desc=f"valid e{epoch}")
            ):
                graphs = graphs.to(device, non_blocking=use_cuda)
                seq_feats = seq_feats.to(device, non_blocking=use_cuda)
                labels = labels_to_device(labels, device)
                ppi_node_ids = ppi_node_ids.to(device, non_blocking=use_cuda)

                with torch.cuda.amp.autocast(enabled=use_cuda and args.amp):
                    logits = model(
                        graphs,
                        seq_feats,
                        label_network,
                        ppi_graph=ppi_graph if args.use_ppi else None,
                        ppi_node_ids=ppi_node_ids if args.use_ppi else None,
                        ppi_node_emb=ppi_node_emb,
                        ppi_neighbor_index=full_ppi_neighbor_index if args.use_ppi else None,
                    )
                    loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)

                valid_loss += loss.item()
                pred += probs.tolist()
                actual += labels.tolist()
                if i == 0 or (i + 1) % 10 == 0:
                    logger.info(f"valid batch {i + 1}/{len(valid_dataloader)}")

        logger.info("valid forward done, computing metrics...")
        auc_score = roc_auc_flat(actual, pred)
        aupr = cacul_aupr(actual, pred)

        each_best_fcore = 0.0
        each_best_scores = []
        score_dict = {}
        for thresh in Thresholds:
            f_score, precision, recall = calculate_performance(
                actual, pred, label_network, threshold=thresh
            )
            if f_score >= each_best_fcore:
                each_best_fcore = f_score
                each_best_scores = [thresh, f_score, recall, precision, auc_score]
                score_dict[thresh] = [f_score, recall, precision, auc_score]

        ckpt_score = _ckpt_selection_score(each_best_fcore, aupr, ckpt_metric)
        if ckpt_score >= best_ckpt_score:
            best_ckpt_score = ckpt_score
            best_fscore = each_best_fcore
            best_scores = each_best_scores
            best_score_dict = score_dict
            best_aupr = aupr
            ckpt = _ckpt_path(data_dir, args, "bestmodel")
            torch.save(model, ckpt)
            print(f"SAVED {ckpt}", flush=True)
            logger.info(
                f"saved checkpoint: {ckpt} (metric={ckpt_metric}, score={ckpt_score:.4f})"
            )

        if each_best_scores:
            thresh, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
            precision, auc_score = each_best_scores[3], each_best_scores[4]
            logger.info("########valid metric###########")
            logger.info(
                f"epoch={epoch}, train_loss={train_loss / len(train_dataloader):.4f}, "
                f"valid_loss={valid_loss / max(len(valid_dataloader), 1):.4f}"
            )
            logger.info(
                f"threshold={thresh}, f_score={f_score}, auc={auc_score}, "
                f"recall={recall}, precision={precision}, aupr={aupr}"
            )
        else:
            logger.warning(f"epoch={epoch}: no valid F-score (empty pred/actual?)")

    logger.info("best_fscore: " + str(best_fscore))
    logger.info(f"best_ckpt_metric: {ckpt_metric}, best_ckpt_score: {best_ckpt_score}")
    logger.info("best_scores[thresh,fmax,recall,precision,auc]: " + str(best_scores))

    final_ckpt = _ckpt_path(data_dir, args, "final")
    torch.save(model, final_ckpt)
    print(f"SAVED {final_ckpt}", flush=True)
    logger.info(f"saved final model: {final_ckpt}")
    logger.info("best_aupr: " + str(best_aupr))
    logger.info("best_score_dict: " + str(best_score_dict))


if __name__ == "__main__":
    main()
