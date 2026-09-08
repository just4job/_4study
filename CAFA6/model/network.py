import torch
import torch.nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn import GraphConv, GATConv, AvgPooling, MaxPooling, SAGEConv
from model.layer import (
    ConvPoolBlock,
    SAGPool,
    BimodalCrossAttention,
    MultiModalCrossAttention,
    ConcatFusion,
)


def patch_legacy_checkpoint(model: torch.nn.Module) -> bool:
    """Full-model .pkl from older code may lack use_ppi / fusion_mode on the instance."""
    has_ppi = getattr(model, "ppi_encoder", None) is not None
    if not hasattr(model, "use_ppi"):
        model.use_ppi = has_ppi
    if not hasattr(model, "fusion_mode"):
        model.fusion_mode = "attention" if getattr(model, "fusion_attn", None) is not None else "concat"
    if model.fusion_mode == "concat" and not hasattr(model, "fusion_concat"):
        model.fusion_concat = None
    if not hasattr(model, "use_label_hierarchy"):
        model.use_label_hierarchy = True
    if not hasattr(model, "label_iters"):
        model.label_iters = 20
    return bool(model.use_ppi)


class PPIEncoder(torch.nn.Module):
    """
    2-layer GraphSAGE encoder trên PPI global graph.
    Nhận toàn bộ PPI graph và list node indices của batch,
    trả về embedding [batch_size, out_dim] cho từng protein.

    Protein không có trong PPI graph (node_id == -1) → zero vector.
    """
    def __init__(self, in_dim: int = 1024, hid_dim: int = 512, out_dim: int = 256,
                 dropout: float = 0.3):
        super(PPIEncoder, self).__init__()
        self.sage1 = SAGEConv(in_dim, hid_dim, aggregator_type="mean")
        self.sage2 = SAGEConv(hid_dim, out_dim, aggregator_type="mean")
        self.dropout = dropout

    def encode_all_nodes(self, ppi_graph: dgl.DGLGraph) -> torch.Tensor:
        """Encode toàn bộ PPI graph một lần — dùng cho cache trên GPU (Kaggle T4)."""
        h = ppi_graph.ndata["feat"]
        h = F.relu(self.sage1(ppi_graph, h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.sage2(ppi_graph, h)

    @staticmethod
    def gather_batch(node_emb: torch.Tensor, node_ids: torch.Tensor) -> torch.Tensor:
        valid_mask = node_ids >= 0
        out = torch.zeros(node_ids.shape[0], node_emb.shape[1], device=node_emb.device, dtype=node_emb.dtype)
        out[valid_mask] = node_emb[node_ids[valid_mask]]
        return out

    @staticmethod
    def build_neighbor_index(ppi_graph: dgl.DGLGraph, max_neighbors: int = 16) -> torch.Tensor:
        """
        Xây bảng index láng giềng [N, K] cho TOÀN BỘ PPI graph (K = max_neighbors + 1).
        - Cột 0 luôn là chính node đó (self-loop ngầm) ⇒ query luôn có ít nhất 1 token hợp lệ.
        - Các cột còn lại là out-neighbor (STRING thường là đồ thị 2 chiều).
        - Padding bằng -1 khi thiếu láng giềng.
        Tính 1 lần rồi cache vì PPI graph là tĩnh (xem prepare_ppi_neighbor_index).
        """
        from collections import defaultdict

        n_nodes = ppi_graph.num_nodes()
        src, dst = ppi_graph.edges()
        src = src.cpu().tolist()
        dst = dst.cpu().tolist()

        adj = defaultdict(list)
        for s, d in zip(src, dst):
            if s != d:                       # bỏ self-loop trùng (đã có ở cột 0)
                adj[s].append(d)

        k = max_neighbors + 1
        idx = torch.full((n_nodes, k), -1, dtype=torch.long)
        idx[:, 0] = torch.arange(n_nodes)
        for n in range(n_nodes):
            nbrs = adj.get(n, [])
            if len(nbrs) > max_neighbors:
                nbrs = nbrs[:max_neighbors]  # cắt theo bậc; có thể đổi sang sampling nếu muốn
            if nbrs:
                idx[n, 1:1 + len(nbrs)] = torch.tensor(nbrs, dtype=torch.long)
        return idx

    @staticmethod
    def gather_neighbor_batch(
        node_emb: torch.Tensor,
        neighbor_index: torch.Tensor,
        node_ids: torch.Tensor,
    ):
        """
        Trả về chuỗi K/V đa-token cho cross-attention.
        node_emb       : [N, D]
        neighbor_index : [N, K] (-1 = padding)
        node_ids       : [B]    (-1 = protein vắng mặt trong PPI graph)
        returns:
            seq      : [B, K, D] embedding (self + láng giềng), vị trí padding = 0
            key_pad  : [B, K] bool, True = bỏ qua trong attention
        Protein vắng mặt → đúng 1 token zero hợp lệ (tránh hàng toàn-mask gây NaN),
        kết quả attention = 0, đồng nhất với hành vi gather_batch cũ.
        """
        device = node_emb.device
        node_ids = node_ids.to(device)
        neighbor_index = neighbor_index.to(device)

        valid_protein = node_ids >= 0                      # [B]
        safe_ids = node_ids.clamp(min=0)
        nbr = neighbor_index[safe_ids]                     # [B, K]
        key_pad = nbr < 0                                  # True = padding
        seq = node_emb[nbr.clamp(min=0)]                   # [B, K, D]
        seq = seq.masked_fill(key_pad.unsqueeze(-1), 0.0)

        # protein vắng mặt: ép cả hàng = 0 nhưng giữ token 0 hợp lệ
        seq[~valid_protein] = 0.0
        key_pad[~valid_protein] = True
        key_pad[~valid_protein, 0] = False
        return seq, key_pad

    def forward(self, ppi_graph: dgl.DGLGraph, node_ids: torch.Tensor) -> torch.Tensor:
        """
        ppi_graph : DGL graph toàn cục (đã ở đúng device), ndata["feat"] (N, in_dim)
        node_ids  : LongTensor [batch_size] — index node của từng protein (-1 = absent)
        returns   : FloatTensor [batch_size, out_dim]
        """
        return self.gather_batch(self.encode_all_nodes(ppi_graph), node_ids)


class SAGNetworkHierarchical(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with hierarchical readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_convs: int = 3,
                 pool_ratio: float = 0.5, dropout: float = 0.5,
                 seq_dim: int = 640,
                 ppi_in_dim: int = 640, ppi_hid_dim: int = 512, ppi_out_dim: int = 256,
                 fusion_attn_heads: int = 4, use_ppi: bool = True,
                 fusion_mode: str = "attention", ppi_max_neighbors: int = 16,
                 use_label_hierarchy: bool = False, label_iters: int = 20):
        super(SAGNetworkHierarchical, self).__init__()

        if fusion_mode not in {"attention", "concat"}:
            raise ValueError(f"fusion_mode must be 'attention' or 'concat', got {fusion_mode!r}")

        self.dropout = dropout
        self.num_convpools = num_convs
        self.use_ppi = use_ppi
        self.fusion_mode = fusion_mode
        self.ppi_out_dim = ppi_out_dim
        self.ppi_max_neighbors = ppi_max_neighbors
        self._ppi_neighbor_index = None  # cache [N, K], dựng 1 lần từ PPI graph tĩnh
        self.use_label_hierarchy = use_label_hierarchy
        self.label_iters = label_iters   # số hop lan truyền (≈ độ sâu DAG của GO)

        convpools = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convpools.append(ConvPoolBlock(_i_dim, _o_dim, pool_ratio=pool_ratio))
        self.convpools = torch.nn.ModuleList(convpools)

        if use_ppi:
            self.ppi_encoder = PPIEncoder(
                in_dim=ppi_in_dim,
                hid_dim=ppi_hid_dim,
                out_dim=ppi_out_dim,
                dropout=dropout,
            )
        else:
            self.ppi_encoder = None

        if fusion_mode == "attention":
            if use_ppi:
                self.fusion_attn = MultiModalCrossAttention(
                    struct_dim=hid_dim * 2,
                    seq_dim=seq_dim,
                    ppi_dim=ppi_out_dim,
                    attn_dim=hid_dim,
                    num_heads=fusion_attn_heads,
                    dropout=dropout,
                )
            else:
                self.fusion_attn = BimodalCrossAttention(
                    struct_dim=hid_dim * 2,
                    seq_dim=seq_dim,
                    attn_dim=hid_dim,
                    num_heads=fusion_attn_heads,
                    dropout=dropout,
                )
            fusion_dim = self.fusion_attn.out_dim
        else:
            self.fusion_attn = None
            ppi_d = ppi_out_dim if use_ppi else 0
            self.fusion_concat = ConcatFusion(
                struct_dim=hid_dim * 2,
                seq_dim=seq_dim,
                ppi_dim=ppi_d,
                out_dim=hid_dim * 2,
                dropout=dropout,
            )
            fusion_dim = self.fusion_concat.out_dim

        self.lin1 = torch.nn.Linear(fusion_dim, hid_dim * 2)
        self.lin2 = torch.nn.Linear(hid_dim * 2, hid_dim)
        self.lin3 = torch.nn.Linear(hid_dim, out_dim)

        self.line_new = torch.nn.Linear(fusion_dim, out_dim)



    def propagate_true_path(self, label_network: dgl.DGLGraph,
                            logits: torch.Tensor) -> torch.Tensor:
        """
        Áp 'true-path rule' của GO: điểm của term cha ≥ max điểm các term con,
        vì gán một term con thì hàm ý gán mọi tổ tiên của nó.

        logits        : [B, L] logit thô. Vì sigmoid đơn điệu nên lấy max trên
                        logit tương đương lấy max trên xác suất ⇒ không cần đổi loss
                        (vẫn dùng BCEWithLogitsLoss trên đầu ra).
        label_network : DGLGraph L node (= out_dim), cạnh hướng child → parent
                        (src = child, dst = parent). Nếu graph của bạn đang là
                        parent → child thì truyền dgl.reverse(label_network).

        Lan truyền lặp self.label_iters lần để max trườn qua nhiều tầng tổ tiên.
        """
        if logits.shape[1] != label_network.num_nodes():
            raise ValueError(
                f"label_network có {label_network.num_nodes()} node nhưng logits có "
                f"{logits.shape[1]} nhãn — hai con số phải bằng nhau."
            )
        g = label_network.local_var()
        h = logits.t().contiguous()                 # [L, B] : mỗi node = 1 nhãn
        for _ in range(self.label_iters):
            g.ndata["h"] = h
            g.update_all(fn.copy_u("h", "m"), fn.max("m", "hm"))
            # parent (dst) nhận max điểm các child (src); giữ lại nếu node không có child
            h = torch.maximum(h, g.ndata["hm"])
        return h.t().contiguous()                   # [B, L]

    def encode_ppi_nodes(self, ppi_graph: dgl.DGLGraph) -> torch.Tensor:
        """Cache PPI node embeddings [N, ppi_out_dim] — gọi 1 lần/epoch khi train."""
        if not getattr(self, "use_ppi", getattr(self, "ppi_encoder", None) is not None):
            raise RuntimeError("encode_ppi_nodes called but use_ppi=False")
        return self.ppi_encoder.encode_all_nodes(ppi_graph)

    def prepare_ppi_neighbor_index(self, ppi_graph: dgl.DGLGraph) -> torch.Tensor:
        """Dựng + cache bảng index láng giềng PPI. Gọi 1 lần (PPI graph tĩnh).
        Nên gọi cùng lúc với encode_ppi_nodes khi dùng đường cache trên GPU."""
        if self._ppi_neighbor_index is None:
            self._ppi_neighbor_index = PPIEncoder.build_neighbor_index(
                ppi_graph, self.ppi_max_neighbors
            )
        return self._ppi_neighbor_index

    def _get_neighbor_index(self, ppi_graph, ppi_neighbor_index, device):
        """Thứ tự ưu tiên: tham số truyền vào → cache → dựng từ ppi_graph → None."""
        nbr = ppi_neighbor_index
        if nbr is None:
            nbr = self._ppi_neighbor_index
        if nbr is None and ppi_graph is not None:
            nbr = self.prepare_ppi_neighbor_index(ppi_graph)
        return nbr.to(device) if nbr is not None else None

    def forward(self, graph: dgl.DGLGraph, sequence_feature: torch.Tensor,
                label_network: dgl.DGLGraph,
                ppi_graph: dgl.DGLGraph = None, ppi_node_ids: torch.Tensor = None,
                ppi_node_emb: torch.Tensor = None,
                ppi_neighbor_index: torch.Tensor = None) -> torch.Tensor:
        """
        graph            : batched contact-map DGL graph
        sequence_feature : [batch, seq_dim]
        label_network    : GO label co-occurrence graph (giữ nguyên, không dùng hiện tại)
        ppi_graph        : PPI global DGL graph (bỏ qua nếu truyền ppi_node_emb)
        ppi_node_ids     : [batch] — index node của từng protein trong PPI graph
        ppi_node_emb     : [N, ppi_out_dim] cache từ encode_ppi_nodes (tối ưu GPU)
        """
        # --- Protein structure branch ---
        feat = graph.ndata["feature"]
        final_readout = None
        for i in range(self.num_convpools):
            graph, feat, readout = self.convpools[i](graph, feat)
            final_readout = readout if final_readout is None else final_readout + readout

        fusion_mode = getattr(self, "fusion_mode", "attention" if self.fusion_attn is not None else "concat")
        use_ppi = getattr(self, "use_ppi", self.ppi_encoder is not None)

        ppi_emb = None          # vector gộp 1-token (dùng cho concat fusion)
        node_emb_full = None     # [N, ppi_out_dim] (dùng để gom láng giềng cho attention)
        if use_ppi:
            if ppi_node_emb is not None:
                node_emb_full = ppi_node_emb
            else:
                node_emb_full = self.ppi_encoder.encode_all_nodes(ppi_graph)
            ppi_emb = PPIEncoder.gather_batch(node_emb_full, ppi_node_ids)

        if fusion_mode == "attention":
            if use_ppi:
                nbr_index = self._get_neighbor_index(
                    ppi_graph, ppi_neighbor_index, node_emb_full.device
                )
                if nbr_index is not None:
                    # K/V = self + láng giềng PPI ⇒ cross-attention thật (softmax > 1 token)
                    ppi_seq, ppi_key_pad = PPIEncoder.gather_neighbor_batch(
                        node_emb_full, nbr_index, ppi_node_ids
                    )
                    final_readout = self.fusion_attn(
                        final_readout, sequence_feature, ppi_seq, ppi_key_pad
                    )
                else:
                    # không dựng được index (vd cache-only không kèm graph) → fallback 1-token
                    final_readout = self.fusion_attn(final_readout, sequence_feature, ppi_emb)
            else:
                final_readout = self.fusion_attn(final_readout, sequence_feature)
        else:
            fusion_concat = getattr(self, "fusion_concat", None)
            if fusion_concat is not None:
                final_readout = fusion_concat(final_readout, sequence_feature, ppi_emb)
            else:
                parts = [final_readout, sequence_feature]
                if use_ppi:
                    parts.append(ppi_emb)
                final_readout = torch.cat(parts, dim=-1)

        feat = F.relu(self.lin1(final_readout))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        feat = self.lin3(feat)
        # Ép tính nhất quán phân cấp GO (parent ≥ children) nếu được bật.
        if getattr(self, "use_label_hierarchy", False) and label_network is not None:
            feat = self.propagate_true_path(label_network, feat)
        # feat: [batch_size, label_num]
        return feat

class SAGNetworkHierarchical2(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with hierarchical readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """
    def __init__(self, in_dim:int, hid_dim:int, out_dim:int, num_convs:int=3,
                 pool_ratio:float=0.5, dropout:float=0.5):
        super(SAGNetworkHierarchical, self).__init__()

        self.dropout = dropout
        self.num_convpools = num_convs
       #self.classify = torch.nn.Linear(hid_dim, out_dim)
        convpools = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convpools.append(ConvPoolBlock(_i_dim, _o_dim, pool_ratio=pool_ratio))
        self.convpools = torch.nn.ModuleList(convpools)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hid_dim * 2 + 1024, nhead=8), num_layers=3)    
        self.lin1 = torch.nn.Linear(hid_dim*2 + 1024, hid_dim*2)
        self.lin2 = torch.nn.Linear(hid_dim*2, hid_dim)
        self.lin3 = torch.nn.Linear(hid_dim, out_dim)
        # self.label_network1 = GATConv(1,1,num_heads=8,allow_zero_in_degree=True)

        self.line_new = torch.nn.Linear(hid_dim * 2 + 1024, out_dim)
        


    def update_parent_features(self,label_network:dgl.DGLGraph, feat):
        # feat: [batch_size, label_num]
        feat = feat.t()
        # feat: [label_num, batch_size]
        
        return feat
    

    def forward(self, graph:dgl.DGLGraph, sequence_feature,label_network:dgl.DGLGraph, ):
        feat = graph.ndata["feature"]
        final_readout = None

        for i in range(self.num_convpools):
            graph, feat, readout = self.convpools[i](graph, feat)
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout
        final_readout = torch.cat((final_readout,sequence_feature), -1)
        final_readout = self.transformer_encoder(final_readout)
        feat = F.relu(self.lin1(final_readout))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        #feat = F.log_softmax(self.lin3(feat), dim=-1)
        feat = self.lin3(feat)
        # feat = feat.t()
        # max_value,_ = torch.max(self.label_network1(label_network,feat),dim=1)
        # feat = F.relu(max_value)
        # feat = self.update_parent_features(label_network, feat)
        # feat = self.line_new(final_readout)
        # feat = torch.sigmoid(feat)
        
        # feat: [batch_size, label_num]
        return feat



class SAGNetworkGlobal(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with global readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """
    def __init__(self, in_dim:int, hid_dim:int, out_dim:int, num_convs=3,
                 pool_ratio:float=0.5, dropout:float=0.0):
        super(SAGNetworkGlobal, self).__init__()
        self.dropout = dropout
        self.num_convs = num_convs

        convs = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convs.append(GraphConv(_i_dim, _o_dim))
        self.convs = torch.nn.ModuleList(convs)

        concat_dim = num_convs * hid_dim
        self.pool = SAGPool(concat_dim, ratio=pool_ratio)
        self.avg_readout = AvgPooling()
        self.max_readout = MaxPooling()

        self.lin1 = torch.nn.Linear(concat_dim * 2 + 1024, hid_dim)
        self.lin2 = torch.nn.Linear(hid_dim, hid_dim // 2)
        self.lin3 = torch.nn.Linear(hid_dim // 2, out_dim)
    
    def forward(self, graph:dgl.DGLGraph, sequence_feature):
        feat = graph.ndata["feature"]
        conv_res = []

        for i in range(self.num_convs):
            feat = self.convs[i](graph, feat)
            conv_res.append(feat)
        
        conv_res = torch.cat(conv_res, dim=-1)
        graph, feat, _ = self.pool(graph, conv_res)
        feat = torch.cat([self.avg_readout(graph, feat), self.max_readout(graph, feat)], dim=-1)
        feat = torch.cat((feat,sequence_feature),-1)
        feat = F.relu(self.lin1(feat))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        #feat = F.log_softmax(self.lin3(feat), dim=-1)
        feat = self.lin3(feat)

        return feat



def get_sag_network(net_type:str="hierarchical"):
    if net_type == "hierarchical":
        return SAGNetworkHierarchical
    elif net_type == "global":
        return SAGNetworkGlobal
    else:
        raise ValueError("SAGNetwork type {} is not supported.".format(net_type))
