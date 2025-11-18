import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from typing import Sequence


class FastAttentionLayer(nn.Module):
    def __init__(self, model_dim: int, num_heads: int = 8, qkv_bias: bool = False, kernel: int = 1) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(2 * model_dim if kernel != 12 else model_dim, model_dim)
        self.fast = 1

    def forward(self, x: torch.Tensor, edge_index=None, dim: int = 0) -> torch.Tensor:
        query, key, value = self.qkv(x).chunk(3, -1)

        qs = torch.stack(torch.split(query, self.head_dim, dim=-1), dim=-2).flatten(start_dim=dim, end_dim=dim + 1)
        ks = torch.stack(torch.split(key, self.head_dim, dim=-1), dim=-2).flatten(start_dim=dim, end_dim=dim + 1)
        vs = torch.stack(torch.split(value, self.head_dim, dim=-1), dim=-2).flatten(start_dim=dim, end_dim=dim + 1)

        if self.fast:
            out_s = self.fast_attention(x, qs, ks, vs, dim=dim)
        else:
            out_s = self.normal_attention(x, qs, ks, vs, dim=dim)

        if x.size(1) > 1:
            qs = torch.stack(torch.split(query.transpose(1, 2), self.head_dim, dim=-1), dim=-2).flatten(
                start_dim=dim, end_dim=dim + 1
            )
            ks = torch.stack(torch.split(key.transpose(1, 2), self.head_dim, dim=-1), dim=-2).flatten(
                start_dim=dim, end_dim=dim + 1
            )
            vs = torch.stack(torch.split(value.transpose(1, 2), self.head_dim, dim=-1), dim=-2).flatten(
                start_dim=dim, end_dim=dim + 1
            )

            if self.fast:
                out_t = self.fast_attention(x.transpose(1, 2), qs, ks, vs, dim=dim).transpose(1, 2)
            else:
                out_t = self.normal_attention(x.transpose(1, 2), qs, ks, vs, dim=dim).transpose(1, 2)

            out = torch.concat([out_s, out_t], -1)
            out = self.out_proj(out)
        else:
            out = self.out_proj(out_s)

        return out

    def fast_attention(self, x: torch.Tensor, qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor, dim: int = 0) -> torch.Tensor:
        qs = nn.functional.normalize(qs, dim=-1)
        ks = nn.functional.normalize(ks, dim=-1)
        N = qs.shape[1]
        b, l = x.shape[dim : dim + 2]

        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
        attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)
        attention_num += N * vs

        all_ones = torch.ones([ks.shape[1]], device=ks.device)
        ks_sum = torch.einsum("blhm,l->bhm", ks, all_ones)
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)

        attention_normalizer = attention_normalizer.unsqueeze(-1)
        attention_normalizer += torch.ones_like(attention_normalizer) * N

        out = attention_num / attention_normalizer
        out = torch.unflatten(out, dim, (b, l)).flatten(start_dim=3)
        return out

    def normal_attention(self, x: torch.Tensor, qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor, dim: int = 0) -> torch.Tensor:
        b, l = x.shape[dim : dim + 2]
        qs, ks, vs = qs.transpose(1, 2), ks.transpose(1, 2), vs.transpose(1, 2)
        x = nn.functional.scaled_dot_product_attention(qs, ks, vs).transpose(-3, -2).flatten(start_dim=-2)
        x = torch.unflatten(x, dim, (b, l)).flatten(start_dim=3)
        return x


class GraphPropagate(nn.Module):
    def __init__(self, Ks: int, gso=None, dropout: float = 0.2) -> None:
        super().__init__()
        self.Ks = Ks
        self.gso = gso
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> list[torch.Tensor]:
        if self.Ks < 1:
            raise ValueError(f"ERROR: Ks must be a positive integer, received {self.Ks}.")
        x_k = x
        x_list = [x]
        for _ in range(1, self.Ks):
            x_k = torch.einsum("thi,btij->bthj", graph, x_k.clone())
            x_list.append(self.dropout(x_k))
        return x_list


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        mlp_ratio: float = 2.0,
        num_heads: int = 8,
        dropout: float = 0.0,
        mask: bool = False,
        kernel: int = 3,
        supports=None,
        order: int = 2,
    ) -> None:
        super().__init__()
        if supports is None or len(supports) == 0:
            base_support = None
        else:
            base_support = supports[0]
        self.locals = GraphPropagate(Ks=order, gso=base_support)
        self.attn = nn.ModuleList(
            [FastAttentionLayer(model_dim, num_heads, mask, kernel=kernel) for _ in range(order)]
        )
        self.pws = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in range(order)])
        for i in range(order):
            nn.init.constant_(self.pws[i].weight, 0)
            nn.init.constant_(self.pws[i].bias, 0)
        self.kernel = kernel
        self.fc = Mlp(
            in_features=model_dim,
            hidden_features=int(model_dim * mlp_ratio),
            act_layer=nn.ReLU,
            drop=dropout,
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = [1, 0.01, 0.001]

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        x_loc = self.locals(x, graph)
        c = x_glo = x
        for i, z in enumerate(x_loc):
            att_outputs = self.attn[i](z)
            x_glo += att_outputs * self.pws[i](c) * self.scale[i]
            c = att_outputs
        x = self.ln1(x + self.dropout(x_glo))
        x = self.ln2(x + self.dropout(self.fc(x)))
        return x


class STGformer(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        input_dim: int = 3,
        output_dim: int = 1,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 12,
        dow_embedding_dim: int = 12,
        spatial_embedding_dim: int = 0,
        adaptive_embedding_dim: int = 12,
        num_heads: int = 4,
        supports=None,
        num_layers: int = 3,
        dropout: float = 0.1,
        mlp_ratio: float = 2.0,
        use_mixed_proj: bool = True,
        dropout_a: float = 0.3,
        kernel_size: Sequence[int] = (1,),
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim

        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        else:
            self.tod_embedding = None

        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        else:
            self.dow_embedding = None

        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
        else:
            self.adaptive_embedding = None

        self.dropout = nn.Dropout(dropout_a)

        k = kernel_size[0]
        self.pooling = nn.AvgPool2d(kernel_size=(1, k), stride=1)

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim,
                    mlp_ratio,
                    num_heads,
                    dropout,
                    kernel=size,
                    supports=supports,
                )
                for size in kernel_size
            ]
        )

        self.encoder_proj = nn.Linear(
            (in_steps - sum(ks - 1 for ks in kernel_size)) * self.model_dim,
            self.model_dim,
        )

        self.kernel_size = k

        self.encoder = nn.ModuleList(
            [
                Mlp(
                    in_features=self.model_dim,
                    hidden_features=int(self.model_dim * mlp_ratio),
                    act_layer=nn.ReLU,
                    drop=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(self.model_dim, out_steps * output_dim)

        self.temporal_proj = nn.Conv2d(self.model_dim, self.model_dim, (1, k), 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        else:
            tod = None
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        else:
            dow = None

        x = x[..., : self.input_dim]
        x = self.input_proj(x)

        features = x.new_zeros(x.shape[:-1] + (0,))
        if self.tod_embedding is not None and tod is not None:
            tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
            features = torch.concat([features, tod_emb], -1)
        if self.dow_embedding is not None and dow is not None:
            dow_emb = self.dow_embedding(dow.long())
            features = torch.concat([features, dow_emb], -1)
        if self.adaptive_embedding is not None:
            adp_emb = self.adaptive_embedding.expand(size=(batch_size, *self.adaptive_embedding.shape))
            features = torch.concat([features, adp_emb], -1)

        x = torch.cat([x, features], dim=-1)

        x = self.temporal_proj(x.transpose(1, 3)).transpose(1, 3)

        if self.adaptive_embedding is not None:
            graph = torch.matmul(self.adaptive_embedding, self.adaptive_embedding.transpose(1, 2))
            graph = self.pooling(graph.transpose(0, 2)).transpose(0, 2)
            graph = F.softmax(F.relu(graph), dim=-1)
        else:
            graph = None

        for attn in self.attn_layers_s:
            x = attn(x, graph)

        x = self.encoder_proj(x.transpose(1, 2).flatten(-2))

        for layer in self.encoder:
            x = x + layer(x)

        out = self.output_proj(x).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)

        out = out.transpose(1, 2)
        out = out.permute(0, 2, 1, 3).squeeze(-1)
        return out


class STGWrapper(nn.Module):
    def __init__(self, core: STGformer) -> None:
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.size(2) == 1:
            x = x.permute(0, 3, 2, 1)  # (B, T_in, 1, N)
            x = x.permute(0, 1, 3, 2)  # (B, T_in, N, 1)
        return self.core(x)


