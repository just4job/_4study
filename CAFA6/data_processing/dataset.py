from __future__ import annotations

from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, emb_graph, emb_seq_feature, emb_label, emb_ppi_node_id=None):
        super().__init__()
        self.list = list(emb_graph.keys())
        self.graphs = emb_graph
        self.seq_feature = emb_seq_feature
        self.label = emb_label
        # ppi_node_id: node index của protein trong PPI global graph.
        # -1 nếu protein không xuất hiện trong PPI graph (isolated).
        self.ppi_node_id = emb_ppi_node_id or {}

    def __getitem__(self, index):
        protein = self.list[index]
        graph = self.graphs[protein]
        seq_feature = self.seq_feature[protein]
        label = self.label[protein]
        ppi_node_id = self.ppi_node_id.get(protein, -1)
        return protein, graph, label, seq_feature, ppi_node_id

    def __len__(self):
        return len(self.list)
