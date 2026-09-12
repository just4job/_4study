import torch
from torch import nn
import torch.nn.functional as F
import argparse
import numpy as np
import warnings

from model.dgl_patch import ensure_dgl_importable

ensure_dgl_importable(verbose=False)

import dgl
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import pickle
from data_processing.divide_data import MyDataSet
from model.evaluation import cacul_aupr, calculate_performance, macro_and_bucket_report, roc_auc_flat
from model.network import patch_legacy_checkpoint
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import warnings
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging
import json
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import re


def _argv_has(*names: str) -> bool:
    return any(n in sys.argv for n in names)


try:
    from scripts.baseline_config import (
        BASELINE_EVAL_THRESH as _BASELINE_EVAL_THRESH,
        baseline_checkpoint_name,
    )
except ImportError:
    from baseline_config import (  # type: ignore
        BASELINE_EVAL_THRESH as _BASELINE_EVAL_THRESH,
        baseline_checkpoint_name,
    )


def _dataset_label_dim(dataset) -> int:
    return int(np.asarray(dataset[0][2]).reshape(-1).shape[0])


def _trim_to_dataset_labels(pred: list, actual: list, dataset_label_dim: int) -> tuple[list, list, int]:
    """Metrics/JSON only on real GO dims when checkpoint was trained with padded label head."""
    output_dim = len(pred[0])
    if output_dim <= dataset_label_dim:
        return pred, actual, output_dim
    print(
        f"[WARN] Model outputs {output_dim} dims but dataset has {dataset_label_dim}; "
        f"metrics/JSON use first {dataset_label_dim} label columns only."
    )
    pred = [row[:dataset_label_dim] for row in pred]
    actual = [row[:dataset_label_dim] for row in actual]
    return pred, actual, dataset_label_dim


def _load_model_checkpoint(model_path, device):
    """Load trusted full-model checkpoints across torch versions."""
    try:
        # PyTorch>=2.6 defaults to weights_only=True, but this project saves full model objects.
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Older torch versions do not support weights_only argument.
        return torch.load(model_path, map_location=device)


def create_logger(branch_name, data_dir: str):
    log_dir = os.path.join(data_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"test_{branch_name}")
    if logger.handlers:
        return logger
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=os.path.join(log_dir, "test_" + branch_name + ".log"))
    logger.setLevel(logging.DEBUG)
    handler1.setLevel(logging.ERROR)
    handler2.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

warnings.filterwarnings('ignore')


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'MyDataSet' and module in {'__main__', 'data_processing.divide_data'}:
            return MyDataSet
        return super().find_class(module, name)


def _load_pickle(path):
    with open(path, 'rb') as handle:
        return _CompatUnpickler(handle).load()

# TODO 个人认为，测试集不用再枚举thresh了，直接使用验证集得出的最优thresh即可
Thresholds = [x / 100 for x in range(1, 100)]

_ACS_FILES = {
    "mf": "human_MF_ACS.json",
    "cc": "human_CC_ACS.json",
    "bp": "human_BP_ACS.json",
}


def _resolve_data_dir() -> str:
    """Pick the first usable CAFA6 data root for local or Kaggle runs."""
    candidates = []
    env_data_dir = os.environ.get("DATA_DIR")
    if env_data_dir:
        candidates.append(Path(env_data_dir))
    candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.cwd())
    candidates.append(Path("D:/CAFA6"))

    for candidate in candidates:
        if (candidate / "divided_data").exists() and (candidate / "proceed_data").exists():
            return str(candidate)

    return env_data_dir or str(Path(__file__).resolve().parent)


def _resolve_eval_dataset(data_dir: str, branch: str, split: str) -> tuple[str, str]:
    """Return (path, split_name). Kaggle pack often has valid but not test."""
    divided = os.path.join(data_dir, "divided_data")
    order: list[str] = []
    if split in ("test", "auto"):
        order.append(f"{branch}_test_dataset")
    if split in ("valid", "auto"):
        order.append(f"{branch}_valid_dataset")
    if split == "train":
        order.append(f"{branch}_train_dataset")
    for name in order:
        path = os.path.join(divided, name)
        if os.path.isfile(path):
            split_name = name.replace(f"{branch}_", "").replace("_dataset", "")
            return path, split_name
    raise FileNotFoundError(
        f"No eval dataset under {divided} for branch={branch}, split={split}. "
        f"Tried: {order}. Re-run kaggle_link_data.py or pack test_dataset in kaggle_data.zip."
    )


def _resolve_latest_model_path(data_dir: str, branch: str, baseline_parity: bool) -> str:
    """Pick bestmodel checkpoint for a branch when no path is provided."""
    save_models = Path(data_dir) / "save_models"
    if baseline_parity:
        preferred = save_models / baseline_checkpoint_name(branch)
        if preferred.is_file():
            return str(preferred)
        # Newest bestmodel for this branch (avoid stale 0.3.pkl after failed MF run)
        by_branch = list(save_models.glob(f"bestmodel_{branch}_*.pkl"))
        if by_branch:
            return str(max(by_branch, key=lambda p: p.stat().st_mtime))

    candidates: list[tuple[float, Path]] = []
    for path in save_models.glob(f"bestmodel_{branch}_*.pkl"):
        match = re.search(r"_(\d+(?:\.\d+)?)\.pkl$", path.name)
        if not match:
            continue
        candidates.append((float(match.group(1)), path))

    if candidates:
        return str(max(candidates, key=lambda item: item[0])[1])

    fallback = save_models / f"bestmodel_{branch}_96_0.0001_0.2.pkl"
    return str(fallback)


def _load_vocab(data_dir: str, branch: str) -> list:
    proc = os.path.join(data_dir, "proceed_data")
    vocab_path = os.path.join(proc, f"label_vocab_{branch}.json")
    if os.path.isfile(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            return json.load(f)
    acs_path = os.path.join(proc, _ACS_FILES[branch])
    if os.path.isfile(acs_path):
        with open(acs_path, "r", encoding="utf-8") as f:
            protein_labels = json.load(f)
        terms = sorted({t for terms in protein_labels.values() for t in terms})
        print(f"[INFO] Built vocab from {acs_path} ({len(terms)} terms)")
        return terms
    warnings.warn(
        f"Missing label_vocab_{branch}.json and {_ACS_FILES[branch]} under {proc}; "
        "falling back to placeholder GO term names based on label graph size. "
        "This lets evaluation run, but result JSON will not contain real GO IDs."
    )
    return []


def _align_vocab_to_output(idx2term: list, output_dim: int) -> list:
    """Make vocabulary length match model output dimension."""
    terms = list(idx2term)
    if len(terms) == output_dim:
        return terms
    if len(terms) > output_dim:
        print(f"[WARN] Vocab has {len(terms)} terms but model outputs {output_dim}; truncating vocab")
        return terms[:output_dim]
    print(f"[WARN] Vocab has {len(terms)} terms but model outputs {output_dim}; padding placeholders")
    start = len(terms)
    terms.extend([f"GO_TERM_{i:05d}" for i in range(start, output_dim)])
    return terms


def _as_flat_float_tensor(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1)


def _as_scalar_long_tensor(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.long).reshape(())


def _fix_dim_1d(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    cur = int(x.shape[0])
    if cur == target_dim:
        return x
    if cur > target_dim:
        return x[:target_dim]
    out = torch.zeros(target_dim, dtype=x.dtype)
    out[:cur] = x
    return out


def _build_eval_collate_fn(seq_dim: int, label_dim: int):
    warned = {"seq": False, "label": False}

    def _collate(batch):
        pids, graphs, labels, seq_feats, ppi_node_ids = zip(*batch)

        fixed_seq_feats = []
        for s in seq_feats:
            t = _as_flat_float_tensor(s)
            if (not warned["seq"]) and int(t.shape[0]) != seq_dim:
                print(
                    f"[WARN] Sequence feature dim mismatch in batch "
                    f"(got {int(t.shape[0])}, expected {seq_dim}); applying pad/truncate"
                )
                warned["seq"] = True
            fixed_seq_feats.append(_fix_dim_1d(t, seq_dim))

        fixed_labels = []
        for y in labels:
            t = _as_flat_float_tensor(y)
            if (not warned["label"]) and int(t.shape[0]) != label_dim:
                print(
                    f"[WARN] Label dim mismatch in batch "
                    f"(got {int(t.shape[0])}, expected {label_dim}); applying pad/truncate"
                )
                warned["label"] = True
            fixed_labels.append(_fix_dim_1d(t, label_dim))

        batched_graph = dgl.batch(list(graphs))
        return (
            list(pids),
            batched_graph,
            torch.stack(fixed_labels, dim=0),
            torch.stack(fixed_seq_feats, dim=0),
            torch.stack([_as_scalar_long_tensor(pid) for pid in ppi_node_ids], dim=0),
        )

    return _collate


def _run_inference(
    dataset,
    model,
    label_network,
    ppi_graph,
    ppi_node_emb,
    use_ppi: bool,
    seq_dim: int,
    label_dim: int,
    device,
    criterion,
    batch_size: int = 32,
    desc: str = "eval",
) -> tuple[list, list, list, float]:
    """Forward pass 1 lần trên `dataset`, trả về (pred, actual, protein_list, avg_loss).

    Dùng chung cho cả pass chọn threshold (trên valid) lẫn pass tính metric cuối
    (trên split đang eval) — đảm bảo 2 pass xử lý giống hệt nhau, tránh lặp code.
    """
    collate_fn = _build_eval_collate_fn(seq_dim=seq_dim, label_dim=label_dim)
    dataloader = GraphDataLoader(
        dataset=dataset, batch_size=batch_size, drop_last=False, shuffle=False, collate_fn=collate_fn,
    )
    t_loss = 0.0
    pred: list = []
    actual: list = []
    protein_list: list = []
    model.eval()
    with torch.no_grad():
        for pids, graphs, labels, seq_feats, ppi_node_ids in tqdm(dataloader, desc=desc):
            graphs = graphs.to(device)
            seq_feats = seq_feats.to(device)
            labels = labels.to(device)
            ppi_node_ids = ppi_node_ids.to(device)
            labels = torch.squeeze(labels)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(0)

            logits = model(
                graphs,
                seq_feats,
                label_network,
                ppi_graph=ppi_graph if use_ppi else None,
                ppi_node_ids=ppi_node_ids if use_ppi else None,
                ppi_node_emb=ppi_node_emb,
            )
            logits = F.sigmoid(logits)
            loss = criterion(logits, labels.float())

            protein_list += pids
            t_loss += loss.item()
            pred += logits.tolist()
            actual += labels.tolist()

    avg_loss = t_loss / max(len(dataloader), 1)
    return pred, actual, protein_list, avg_loss


def _select_threshold(actual, pred, label_network, thresholds) -> tuple[float, float, float, float]:
    """Quét `thresholds` trên (actual, pred) truyền vào, trả về (best_thresh, f_score,
    precision, recall) tốt nhất. CHỈ gọi hàm này trên VALID (hoặc trên chính split
    đang eval nếu đó đã LÀ valid) — không bao giờ gọi trên test, vì đó chính là
    threshold-leak: chọn siêu tham số bằng cách nhìn thấy trước nhãn thật của test."""
    best = (thresholds[0], 0.0, 0.0, 0.0)
    for thresh in tqdm(thresholds, desc="threshold search"):
        f_score, precision, recall = calculate_performance(actual, pred, label_network, threshold=thresh)
        if f_score >= best[1]:
            best = (thresh, f_score, precision, recall)
    return best


if __name__ == "__main__":
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-branch', '--branch',type=str,default='mf')
    parser.add_argument(
        '-thresh', '--thresh', type=float, default=0.71,
        help=(
            "Threshold dùng CHỈ để liệt kê nhãn 'mới dự đoán' trong "
            "test_result/{branch}_result.json. KHÔNG dùng cho F-max/precision/recall "
            "báo cáo — threshold đó giờ tự chọn từ valid (xem [thresh-select] trong log)."
        ),
    )
    parser.add_argument('-batch', '--batch', type = str, default = '1')
    parser.add_argument("-model_path", "--model_path", type=str, default="")
    parser.add_argument(
        "--split",
        type=str,
        default="auto",
        choices=["auto", "test", "valid", "train"],
        help="auto: test nếu có, không thì valid (phù hợp Kaggle pack)",
    )
    parser.set_defaults(use_ppi=True, baseline_parity=True, fusion_mode="attention")
    parser.add_argument(
        "--no-ppi",
        dest="use_ppi",
        action="store_false",
        help="Không load PPI graph khi eval (model train không PPI)",
    )
    parser.add_argument(
        "--fusion",
        dest="fusion_mode",
        choices=["attention", "bi_attention", "concat"],
        default="attention",
        help="Fusion struct/seq/ppi: attention (1 chiều) | bi_attention (2 chiều) | concat (mặc định lấy từ checkpoint)",
    )
    parser.add_argument(
        "--no-baseline-parity",
        dest="baseline_parity",
        action="store_false",
        help="Eval nhanh: split auto, model path theo run Kaggle",
    )
    parser.add_argument(
        "--baseline-parity",
        dest="baseline_parity",
        action="store_true",
        help="Eval khớp baseline (mặc định bật)",
    )
    args = parser.parse_args()

    if args.baseline_parity:
        args.split = "test"
        if not _argv_has("-thresh", "--thresh"):
            args.thresh = _BASELINE_EVAL_THRESH.get(args.branch, 0.71)

    input_thresh = args.thresh
    data_dir = _resolve_data_dir()
    test_data_path, eval_split = _resolve_eval_dataset(data_dir, args.branch, args.split)
    label_network_path = f"{data_dir}/proceed_data/label_{args.branch}_network"
    ppi_graph_path = f"{data_dir}/proceed_data/ppi_graph_global"
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = _resolve_latest_model_path(data_dir, args.branch, args.baseline_parity)
    if not os.path.isabs(model_path) and not os.path.isfile(model_path):
        alt = os.path.join(data_dir, model_path)
        if os.path.isfile(alt):
            model_path = alt

    result_dir = os.path.join(data_dir, "test_result")
    os.makedirs(result_dir, exist_ok=True)

    logger = create_logger(args.branch, data_dir)
    import __main__

    __main__.MyDataSet = MyDataSet

    test_dataset = _load_pickle(test_data_path)
    label_network = _load_pickle(label_network_path)
    label_network = label_network.to(device)
    ppi_graph = None
    if args.use_ppi:
        ppi_graph = _load_pickle(ppi_graph_path)
        ppi_graph = ppi_graph.to(device)
    idx2term = _load_vocab(data_dir, args.branch)
    if not idx2term:
        idx2term = [f"GO_TERM_{i:05d}" for i in range(label_network.num_nodes())]
        print(f"[WARN] Using placeholder vocabulary with {len(idx2term)} terms")
    model = _load_model_checkpoint(model_path, device)
    model = model.to(device)
    model_has_ppi = patch_legacy_checkpoint(model)
    if not _argv_has("--fusion"):
        args.fusion_mode = getattr(model, "fusion_mode", args.fusion_mode)
    if args.fusion_mode in ("attention", "bi_attention") and getattr(model, "fusion_attn", None) is None:
        print(f"[WARN] fusion_mode={args.fusion_mode} but checkpoint has no fusion_attn; using concat")
        args.fusion_mode = "concat"
    use_ppi = model_has_ppi and args.use_ppi
    if args.use_ppi and not model_has_ppi:
        print("[WARN] --use_ppi set but checkpoint has no ppi_encoder; eval without PPI")
    if not use_ppi:
        args.use_ppi = False
        ppi_graph = None

    logger.info(
        f"eval split={eval_split}, dataset={test_data_path}, "
        f"baseline_parity={args.baseline_parity}, use_ppi={args.use_ppi}, "
        f"fusion_mode={args.fusion_mode}, thresh={input_thresh}, model={model_path}"
    )

    ppi_node_emb = None
    if use_ppi and ppi_graph is not None and hasattr(model, "encode_ppi_nodes"):
        with torch.no_grad():
            ppi_node_emb = model.encode_ppi_nodes(ppi_graph)

    if hasattr(model, "fusion_attn") and hasattr(model.fusion_attn, "seq_proj"):
        seq_dim = int(model.fusion_attn.seq_proj.in_features)
    else:
        seq_dim = int(np.asarray(test_dataset[0][3]).reshape(-1).shape[0])
    if hasattr(model, "lin3"):
        label_dim = int(model.lin3.out_features)
    else:
        label_dim = int(label_network.num_nodes())
    dataset_label_dim = _dataset_label_dim(test_dataset)
    if label_dim != dataset_label_dim:
        print(
            f"[INFO] dataset labels={dataset_label_dim}, model head={label_dim} "
            f"(collate pads to model; metrics trim to dataset)"
        )

    batch_size = 32
    criterion = nn.CrossEntropyLoss()
    logger.info('#########'+args.branch+'###########')
    logger.info('########start testing###########')

    # 1) Chọn threshold trên VALID (không phải trên split đang eval) — chống
    #    threshold-leak: trước đây threshold "tốt nhất" được chọn bằng cách quét
    #    99 mức NGAY TRÊN chính tập test, thổi phồng F-max báo cáo. Nếu split
    #    đang eval CHÍNH LÀ valid (vd. `--split valid` để tune) thì không cần
    #    bước này — tự chọn threshold trên chính nó là hợp lệ.
    best_thresh = None
    if eval_split != "valid":
        valid_path = os.path.join(data_dir, "divided_data", f"{args.branch}_valid_dataset")
        if os.path.isfile(valid_path):
            valid_dataset = _load_pickle(valid_path)
            v_pred, v_actual, _, v_loss = _run_inference(
                valid_dataset, model, label_network, ppi_graph, ppi_node_emb,
                args.use_ppi, seq_dim, label_dim, device, criterion,
                batch_size=batch_size, desc="valid (chọn threshold)",
            )
            v_dim = _dataset_label_dim(valid_dataset)
            v_pred, v_actual, _ = _trim_to_dataset_labels(v_pred, v_actual, v_dim)
            best_thresh, v_fscore, v_precision, v_recall = _select_threshold(
                v_actual, v_pred, label_network, Thresholds
            )
            msg = (
                f"[thresh-select] threshold={best_thresh} chọn từ VALID "
                f"(f_score_valid={v_fscore:.4f}, loss_valid={v_loss:.4f}) -> áp dụng "
                f"nguyên threshold này lên '{eval_split}', KHÔNG quét lại trên '{eval_split}'."
            )
            logger.info(msg)
            print(msg)
        else:
            msg = (
                f"[thresh-select][WARN] Không tìm thấy {valid_path} — fallback: quét "
                f"threshold trực tiếp trên '{eval_split}' (LEAK nếu '{eval_split}'=test). "
                "Chạy divide_data.py để có đủ valid_dataset và tránh cảnh báo này."
            )
            logger.warning(msg)
            print(msg)

    # 2) Forward pass trên split đang eval (test/valid/train theo --split)
    print("testing")
    pred, actual, protein_list, t_loss = _run_inference(
        test_dataset, model, label_network, ppi_graph, ppi_node_emb,
        args.use_ppi, seq_dim, label_dim, device, criterion,
        batch_size=batch_size, desc=f"{eval_split} eval",
    )
    
    # 为了保持可控，这里使用传入的thresh来确定最终分类结果
    assert len(pred) == len(actual)
    assert len(pred) == len(protein_list)
    if not pred:
        raise RuntimeError("No predictions were generated. Check eval dataset and dataloader.")

    pred, actual, output_dim = _trim_to_dataset_labels(pred, actual, dataset_label_dim)
    idx2term = _align_vocab_to_output(idx2term, output_dim)

    result = {}
    
    temp = {}
    temp["pred"] = pred
    temp["actual"] = actual
    with open(os.path.join(result_dir, args.branch + args.batch + "_pred_actual.pkl"), "wb") as f:
        pickle.dump(temp, f)

    for i in range(len(pred)):
        protein = protein_list[i]
        result[protein] = []
        y_ = pred[i]
        y = actual[i]
        for j in range(output_dim):
            x = idx2term[j]
            # 寻找新预测出来的标签
            if y[j] < 1.0 and y_[j] > input_thresh:
                result[protein].append(x + f" {y_[j]:.5f}")
    with open(os.path.join(result_dir, args.branch + "_result.json"), "w") as f:
        json.dump(result, f, indent=4)
        
    flat_actual = np.asarray(actual).reshape(-1)
    flat_pred = np.asarray(pred).reshape(-1)
    auc_score = roc_auc_flat(flat_actual, flat_pred)
    aupr = cacul_aupr(flat_actual, flat_pred)
    fpr, tpr, _ = roc_curve(flat_actual, flat_pred, pos_label=1)

    # 3) Áp threshold đã chọn từ valid (best_thresh) lên split đang eval — KHÔNG
    #    quét lại 99 mức trên chính nó. Chỉ khi không có valid_dataset (best_thresh
    #    is None, xem cảnh báo [thresh-select] ở trên) mới fallback quét trực tiếp
    #    (giữ hành vi cũ để không crash, nhưng đã cảnh báo rõ đây là leak).
    if best_thresh is not None:
        t = best_thresh
        f_score, precision, recall = calculate_performance(actual, pred, label_network, threshold=t)
    else:
        t, f_score, precision, recall = _select_threshold(actual, pred, label_network, Thresholds)
    logger.info('loss: {}, thresh: {}, f_score {}'.format(t_loss, t, f_score))
    logger.info('auc {}, recall {}, precision {},aupr {}'.format(auc_score, recall, precision, aupr))
    print('loss: {}, thresh: {}, f_score {}'.format(t_loss, t, f_score))
    print('auc {}, recall {}, precision {},aupr {}'.format(auc_score, recall, precision, aupr))

    # Chẩn đoán mất cân bằng (không đổi F-max/threshold báo cáo ở trên, chỉ log
    # thêm): micro-F1 phía trên có thể "đẹp" trong khi model gần như bỏ rơi
    # label hiếm — xem README mục đề xuất cải tiến (Tier B2).
    bucket_report = macro_and_bucket_report(actual, pred, threshold=t)
    bucket_msg = (
        f"macro_f1={bucket_report['macro_f1']:.4f} | "
        f"rare(n={bucket_report['rare_n_labels']})_f1={bucket_report['rare_f1']} | "
        f"medium(n={bucket_report['medium_n_labels']})_f1={bucket_report['medium_f1']} | "
        f"common(n={bucket_report['common_n_labels']})_f1={bucket_report['common_f1']}"
    )
    logger.info(bucket_msg)
    print(bucket_msg)


    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # 参考线（随机分类器）
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({args.branch})')
    plt.legend(loc="lower right")

    # save ROC pic
    roc_curve_path = os.path.join(result_dir, f"{args.branch}_roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.show()
    
    logger.info(f"ROC curve saved to {roc_curve_path}")

