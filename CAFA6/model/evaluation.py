from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import numpy as np
import dgl



def update_parent_features(label_network:dgl.DGLGraph, labels):
    # 获取图中的所有边
    edges = label_network.edges()
    # 对于图中的每条边
    for child_idx, parent_idx in zip(edges[0], edges[1]):
        # 如果child节点的特征值大于parent节点的特征值
        if labels[0][child_idx] > labels[0][parent_idx]:
            # 更新parent节点的特征值为child节点的特征值
            labels[0][parent_idx] = labels[0][child_idx]
        # 更新labels的第二列为second_dim_elements
    return labels

def cacul_aupr(lables, pred):
    labels = np.asarray(lables).reshape(-1)
    scores = np.asarray(pred).reshape(-1)
    if labels.size > 500_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(labels.size, 500_000, replace=False)
        labels = labels[idx]
        scores = scores[idx]
    precision, recall, _thresholds = metrics.precision_recall_curve(labels, scores)
    return metrics.auc(recall, precision)


def roc_auc_flat(actual, pred, max_n: int = 500_000) -> float:
    y = np.asarray(actual).reshape(-1)
    p = np.asarray(pred).reshape(-1)
    if y.size > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(y.size, max_n, replace=False)
        y = y[idx]
        p = p[idx]
    fpr, tpr, _ = roc_curve(y, p, pos_label=1)
    return float(auc(fpr, tpr))

def calculate_performance(actual, pred_prob, label_network:dgl.DGLGraph, threshold=0.2, average='micro'):
    pred_lable = []
    actual_label = []
    for l in range(len(pred_prob)):
        eachline = (np.array(pred_prob[l]).flatten() > threshold).astype(np.int32)
        eachline = eachline.tolist()
        # eachline = update_parent_features(label_network,eachline)
        pred_lable.append(list(eachline))
    for l in range(len(actual)):
        eachline = (np.array(actual[l]).flatten()).astype(np.int32)
        eachline = eachline.tolist()
        actual_label.append(list(eachline))
    f_score = f1_score(actual_label, pred_lable, average=average)
    recall = recall_score(actual_label, pred_lable, average=average)
    precision = precision_score(actual_label,  pred_lable, average=average)
    return f_score, precision, recall


def per_label_f1(actual, pred_prob, threshold: float = 0.2) -> np.ndarray:
    """F1 RIÊNG cho từng label (average=None) — dùng để tính macro-F1 hoặc
    breakdown theo nhóm tần suất. KHÔNG dùng để chọn checkpoint (vẫn micro như
    calculate_performance ở trên) — chỉ để CHẨN ĐOÁN model có bỏ rơi label hiếm
    hay không, vì micro-F1 bị label phổ biến che mất (mỗi mẫu đóng góp như nhau
    vào TP/FP/FN gộp chung, label hiếm gần như không ảnh hưởng tới micro-F1)."""
    pred_label = (np.asarray(pred_prob) > threshold).astype(np.int32)
    actual_label = np.asarray(actual).astype(np.int32)
    return f1_score(actual_label, pred_label, average=None, zero_division=0)


def label_frequency_buckets(
    actual, rare_max: int = 10, medium_max: int = 50
) -> dict[str, np.ndarray]:
    """Chia index label thành 3 nhóm theo số positive quan sát được TRONG CHÍNH
    `actual` (nhãn thật của split đang đánh giá) — không phải tần suất trên
    toàn bộ train, để không cần truyền thêm dữ liệu train vào hàm này:
      rare   : < rare_max positive trong split này
      medium : [rare_max, medium_max)
      common : >= medium_max
    Ngưỡng mặc định (10/50) hợp lý cho valid/test cỡ vài trăm–vài nghìn mẫu,
    ứng với min-count train mặc định 100–250 (xem split_protein_ids.py)."""
    counts = np.asarray(actual).sum(axis=0)
    return {
        "rare": np.where(counts < rare_max)[0],
        "medium": np.where((counts >= rare_max) & (counts < medium_max))[0],
        "common": np.where(counts >= medium_max)[0],
    }


def macro_and_bucket_report(
    actual, pred_prob, threshold: float = 0.2, rare_max: int = 10, medium_max: int = 50
) -> dict:
    """Tổng hợp macro-F1 toàn bộ + F1 trung bình riêng từng nhóm tần suất
    (rare/medium/common) tại 1 threshold cố định — dùng để log chẩn đoán bên
    cạnh F-max micro (không thay thế cơ chế chọn checkpoint hiện có)."""
    f1_per_label = per_label_f1(actual, pred_prob, threshold=threshold)
    buckets = label_frequency_buckets(actual, rare_max=rare_max, medium_max=medium_max)
    report: dict = {
        "macro_f1": float(np.mean(f1_per_label)) if len(f1_per_label) else 0.0,
        "n_labels": int(len(f1_per_label)),
    }
    for name, idx in buckets.items():
        report[f"{name}_n_labels"] = int(len(idx))
        report[f"{name}_f1"] = float(np.mean(f1_per_label[idx])) if len(idx) else None
    return report
