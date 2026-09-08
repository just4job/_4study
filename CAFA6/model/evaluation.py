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
