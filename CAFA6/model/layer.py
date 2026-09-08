import torch
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, AvgPooling, MaxPooling, GATConv,SumPooling,SAGEConv,ChebConv
from model.utils import topk, get_batch_id

class SAGPool(torch.nn.Module):
    """The Self-Attention Pooling layer in paper 
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The dimension of node feature.
        ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        conv_op (torch.nn.Module, optional): The graph convolution layer in dgl used to
        compute scale for each node. (default: :obj:`dgl.nn.GraphConv`)
        non_linearity (Callable, optional): The non-linearity function, a pytorch function.
            (default: :obj:`torch.tanh`)
    """
    def __init__(self, in_dim:int, ratio=0.5, conv_op=GraphConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.score_layer1 = GraphConv(in_dim, 1, allow_zero_in_degree=True)
        self.score_layer2 = GraphConv(in_dim, 1, allow_zero_in_degree=True)
        self.non_linearity = non_linearity
    
    def forward(self, graph:dgl.DGLGraph, feature:torch.Tensor):

        score1 = self.score_layer1(graph, feature).squeeze()
        score2 = self.score_layer2(graph, feature).squeeze()
        score  = (score1+score2)/2
        perm, next_batch_num_nodes = topk(score, self.ratio, get_batch_id(graph.batch_num_nodes()), graph.batch_num_nodes())
        feature = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
        graph = dgl.node_subgraph(graph, perm)

        # node_subgraph currently does not support batch-graph,
        # the 'batch_num_nodes' of the result subgraph is None.
        # So we manually set the 'batch_num_nodes' here.
        # Since global pooling has nothing to do with 'batch_num_edges',
        # we can leave it to be None or unchanged.
        # DGL 1.x: set_batch_num_nodes; DGL 2.x: batch info giữ qua node_subgraph
        if hasattr(graph, "set_batch_num_nodes"):
            graph.set_batch_num_nodes(next_batch_num_nodes)

        return graph, feature, perm


class ConvPoolBlock(torch.nn.Module):
    """A combination of GCN layer and SAGPool layer,
    followed by a concatenated (max||sum) readout operation.
    """
    def __init__(self, in_dim:int, out_dim:int, pool_ratio=0.5):
        super(ConvPoolBlock, self).__init__()
        self.conv1 = GraphConv(in_dim, out_dim, allow_zero_in_degree=True)
        self.conv2 = GraphConv(out_dim, out_dim, allow_zero_in_degree=True)
        self.pool = SAGPool(out_dim, ratio=pool_ratio)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        self.sumpool = SumPooling()
    
    def forward(self, graph, feature):
        # Reshape động theo out_dim của conv (không hard-code 512)
        out = F.relu(self.conv1(graph, feature))
        out = out.reshape(-1, out.shape[-1])
        out = F.relu(self.conv2(graph, out))
        out = out.reshape(-1, out.shape[-1])
        out = F.relu(self.conv2(graph, out))
        out = out.reshape(-1, out.shape[-1])
        graph, out, _ = self.pool(graph, out)
        g_out = torch.cat([self.maxpool(graph, out), self.sumpool(graph, out)], dim=-1)
        return graph, out, g_out


class BimodalCrossAttention(torch.nn.Module):
    """Struct + sequence cross-attention (không dùng PPI)."""

    def __init__(
        self,
        struct_dim: int,
        seq_dim: int,
        attn_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.struct_proj = torch.nn.Linear(struct_dim, attn_dim)
        self.seq_proj = torch.nn.Linear(seq_dim, attn_dim)
        self.cross_attn = torch.nn.MultiheadAttention(
            attn_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = torch.nn.LayerNorm(attn_dim)
        self.out_dim = attn_dim * 2

    def forward(self, struct_feat: torch.Tensor, seq_feat: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack(
            [self.struct_proj(struct_feat), self.seq_proj(seq_feat)], dim=1
        )
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        attn_out = self.norm(attn_out + tokens)
        return attn_out.reshape(attn_out.shape[0], -1)


class MultiModalCrossAttention(torch.nn.Module):
    """Struct + sequence embeddings attend to PPI context via cross-attention."""

    def __init__(
        self,
        struct_dim: int,
        seq_dim: int,
        ppi_dim: int,
        attn_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.struct_proj = torch.nn.Linear(struct_dim, attn_dim)
        self.seq_proj = torch.nn.Linear(seq_dim, attn_dim)
        self.ppi_proj = torch.nn.Linear(ppi_dim, attn_dim)
        self.cross_attn = torch.nn.MultiheadAttention(
            attn_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = torch.nn.LayerNorm(attn_dim)
        self.out_dim = attn_dim * 2

    def forward(
        self,
        struct_feat: torch.Tensor,
        seq_feat: torch.Tensor,
        ppi_feat: torch.Tensor,
        ppi_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        struct_feat / seq_feat : [B, *] — tạo 2 token query (struct, seq).
        ppi_feat :
            - [B, ppi_dim]        → 1 token K/V (hành vi cũ, tương thích ngược)
            - [B, L, ppi_dim]     → L token K/V (self + láng giềng PPI) ⇒ attention thật
        ppi_key_padding_mask : [B, L] bool, True = vị trí padding (bỏ qua trong softmax).
        """
        query = torch.stack(
            [self.struct_proj(struct_feat), self.seq_proj(seq_feat)], dim=1
        )  # [B, 2, attn_dim]

        if ppi_feat.dim() == 2:
            ppi_ctx = self.ppi_proj(ppi_feat).unsqueeze(1)  # [B, 1, attn_dim]
            ppi_key_padding_mask = None
        else:
            ppi_ctx = self.ppi_proj(ppi_feat)               # [B, L, attn_dim]

        attn_out, _ = self.cross_attn(
            query, ppi_ctx, ppi_ctx, key_padding_mask=ppi_key_padding_mask
        )
        attn_out = self.norm(attn_out + query)
        return attn_out.reshape(attn_out.shape[0], -1)


class ConcatFusion(torch.nn.Module):
    """LayerNorm + MLP trên vector concat — cùng out_dim với cross-attention."""

    def __init__(
        self,
        struct_dim: int,
        seq_dim: int,
        ppi_dim: int = 0,
        out_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_ppi = ppi_dim > 0
        self.struct_ln = torch.nn.LayerNorm(struct_dim)
        self.seq_ln = torch.nn.LayerNorm(seq_dim)
        if self.use_ppi:
            self.ppi_ln = torch.nn.LayerNorm(ppi_dim)
            in_dim = struct_dim + seq_dim + ppi_dim
        else:
            self.ppi_ln = None
            in_dim = struct_dim + seq_dim
        self.out_dim = out_dim
        hidden = max(out_dim, in_dim // 2)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        struct_feat: torch.Tensor,
        seq_feat: torch.Tensor,
        ppi_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [self.struct_ln(struct_feat), self.seq_ln(seq_feat)]
        if self.use_ppi and ppi_feat is not None:
            parts.append(self.ppi_ln(ppi_feat))
        return self.mlp(torch.cat(parts, dim=-1))