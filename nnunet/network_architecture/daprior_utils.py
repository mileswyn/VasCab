import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from timm.models.layers import DropPath
import math

logger = logging.getLogger(__name__)

def init_params(module, num_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(num_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

class GraphNodeFeature(nn.Module):
    
    def __init__(self, num_heads, hidden_dim, num_layers=24):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        # self.num_atoms = num_atoms
        
        # self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        # self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        # self.out_degree_encoder = nn.Embedding(
        #     num_out_degree, hidden_dim, padding_idx=0
        # )
        # self.graph_token = nn.Embedding(1, hidden_dim)
        
        # self.apply(lambda module: init_params(module, num_layers=num_layers))
    
    # def forward(self, batched_data):
    def forward(self, x):
        # x, in_degree, out_degree = (
        #     batched_data["x"],
        #     batched_data["in_degree"],
        #     batched_data["out_degree"],
        # )
        # n_graph, n_node = x.size()[:2]  # [B, T, 9]

        # # node feauture + graph token
        # node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        
        # node_feature = (
        #         node_feature
        #         + self.in_degree_encoder(in_degree)  # [n_graph, n_node, n_hidden]
        #         + self.out_degree_encoder(out_degree)  # [n_graph, n_node, n_hidden]
        # )
        
        # graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        # graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        
        return x

class GraphEdgeFeature(nn.Module):
    
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0., num_layers=24):
        super(GraphEdgeFeature, self).__init__()

        inner_dim = dim_head * heads  # dim 
        self.heads = heads
        self.scale = dim_head ** -0.5
        # self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)
        self.to_qk = nn.Linear(dim, inner_dim*2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.apply(lambda module: init_params(module, num_layers=num_layers))

    def b_l_hd__b_h_l_d(self, x, heads):
        b, l, n = x.shape
        h = heads
        d = int(n / h)

        x = x.view(b, l, h, d)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x
    
    def b_l_hd__b_l_h_d(self, x, heads):
        b, l, n = x.shape
        h = heads
        d = int(n / h)

        x = x.view(b, l, h, d).contiguous()
        return x

    def b_h_l_d__b_l_hd(self, x):
        b, h, l, d = x.shape

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, l, -1).contiguous()
        return x
    
    def forward(self, x):
        # q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k = self.to_qk(x).chunk(2, dim=-1)

        # q, k, v = map(lambda t: self.b_l_hd__b_h_l_d(t, self.heads), [q, k, v])
        # attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # q, k, v = map(lambda t: self.b_l_hd__b_l_h_d(t, self.heads), [q, k, v])
        attn = torch.einsum('bqc,bkc->bqk', q, k) * self.scale  # b n n 

        attn = F.softmax(attn, dim=-1)

        # attned = torch.einsum('bhij,bhjd->bhid', attn, v)
        # attned = self.b_h_l_d__b_l_hd(attned)
        
        # attned = self.b_h_l_d__b_l_hd(attned)
        # attned = attned.permute(0, 3, 1, 2).contiguous()

        # attned = torch.einsum('bqk,bkc->bqkc', attn, v) # b n n c
        # attned = self.to_out(attned).permute(0, 3, 1, 2).contiguous()
        # return attned
        return attn

class GraphEmbedding(nn.Module):
    def __init__(self, feat_dim, num_layers, node_dim, num_heads, dropout):
        
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = node_dim
        self.emb_layer_norm = nn.LayerNorm(node_dim)
        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_heads,
            hidden_dim=node_dim,
            num_layers=num_layers,
        )
        
        self.graph_edge_feature = GraphEdgeFeature(
            dim=feat_dim, 
            heads=feat_dim//32, 
            dim_head=32,
        )
 
    def forward(self, x, perturb=None):
        # compute padding mask. This is needed for multi-head attention
        b, c, d, h, w = x.shape
        data_x = x.view(b, c, -1)  # [B, C, d, h, w]
        n_graph = b
        padding_mask = (data_x[:, :, 0]).eq(0)  # B C
        # padding_mask_cls = torch.zeros(  # not mask
        #     n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        # )  # B node_embeds 1
        # padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        # B node_embeds (1+T)
        
        node_embeds = self.graph_node_feature(data_x)
        if perturb is not None:  # perturb is None
            node_embeds[:, 1:, :] += perturb  # token那一列不需要加入扰动
        
        # node_embeds: B node_embeds T node_embeds C

        edge_embeds = self.graph_edge_feature(node_embeds.transpose(-1,-2))  # b n n 
        
        return node_embeds, edge_embeds, padding_mask
    
class GraphPropagationAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = node_dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(node_dim)

        self.qkv = nn.Linear(node_dim, node_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(node_dim, node_dim)
        
        self.expand = nn.Conv2d(1, num_heads, kernel_size=1)
        self.reduce = nn.Conv2d(num_heads, 1, kernel_size=1)
        if edge_dim != node_dim:
            self.fc = nn.Linear(edge_dim, node_dim)
        else:
            self.fc = nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, node_embeds, edge_embeds, da_prior):
        B, C, N = node_embeds.shape
        node_embeds = node_embeds.transpose(-1, -2)

        attned_node = torch.einsum('bnn,bnc->bnc', edge_embeds, node_embeds)
        da_prior = da_prior.permute(0,2,1).contiguous()
        updated_node = torch.einsum('bnc,bca->bnc', attned_node, da_prior)
        node_embeds = updated_node + node_embeds

        # node_embeds = node_embeds.transpose(-1, -2)
        qkv = self.qkv(self.norm(node_embeds)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, n_head, 1+N, 1+N]
        attn_bias = self.expand(edge_embeds.unsqueeze(1)) # [B, C, 1+N, 1+N] -> [B, n_head, 1+N, 1+N]
        attn = attn + attn_bias # [B, n_head, 1+N, 1+N]
        residual = attn
        attn = attn.softmax(dim=-1) # [B, C, N, N]
        attn = self.attn_drop(attn)

        node_embeds = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # node-to-edge propagation
        edge_embeds = self.reduce(attn + residual)

        # edge-to-node propagation
        w = edge_embeds.softmax(dim=-1)
        w = (w * edge_embeds).sum(-1).transpose(-1, -2)
        node_embeds = node_embeds + self.fc(w)
        node_embeds = self.proj(node_embeds)
        node_embeds = self.proj_drop(node_embeds).transpose(-1,-2)

        edge_embeds = edge_embeds.squeeze(1)

        return node_embeds, edge_embeds
    
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., drop_act=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.drop_act = nn.Dropout(drop_act)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop_act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class GPTransBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, feat_dim, num_heads, mlp_ratio=1., qkv_bias=True, drop=0., drop_act=0.,
                 with_cp=False, attn_drop=0., drop_path=0., init_values=None):
        super().__init__()
        self.with_cp = with_cp
        self.norm1 = nn.LayerNorm(node_dim)
        self.gpa = GraphPropagationAttention(node_dim, edge_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                             attn_drop=attn_drop, proj_drop=drop)
        # self.norm2 = nn.LayerNorm(node_dim)
        # self.ffn = FFN(in_features=node_dim, hidden_features=int(node_dim * mlp_ratio),
        #                drop=drop, drop_act=drop_act)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # if init_values is not None:
        #     self.gamma1 = nn.Parameter(init_values * torch.ones((node_dim)), requires_grad=True)
        #     self.gamma2 = nn.Parameter(init_values * torch.ones((node_dim)), requires_grad=True)
        # else:
        #     self.gamma1 = None
        #     self.gamma2 = None

    def forward(self, node_embeds, edge_embeds, padding_mask, da_prior):
        # node_embeds: b c n
        # node_embeds_, edge_embeds_ = self.gpa(self.norm1(node_embeds.permute(0,2,1)), edge_embeds, da_prior)
        node_embeds = self.norm1(node_embeds.permute(0,2,1)).permute(0,2,1)
        node_embeds_, edge_embeds_ = self.gpa(node_embeds, edge_embeds, da_prior)
        # node_embeds_ = node_embeds_ + node_embeds
        edge_embeds = edge_embeds + self.drop_path(edge_embeds_)
        # x = x + self.drop_path(attn)
        # x = x + self.drop_path(self.ffn(self.norm2(x)))
        return node_embeds_, edge_embeds

class GPTrans(nn.Module):
    def __init__(self, feat_dim, encode_prior_dim, num_layers=2, num_heads=8, node_dim=320, 
                 edge_dim=320, layer_scale=1.0, mlp_ratio=1.0, drop_rate=0.0, attn_drop_rate=0.0, 
                 drop_path_rate=0.0, qkv_bias=True, random_feature=False, with_cp=False):
        super(GPTrans, self).__init__()
        # logger.info(f"drop: {drop_rate}, drop_path_rate: {drop_path_rate}, attn_drop_rate: {attn_drop_rate}")
        
        self.random_feature = random_feature
        self.graph_embedding = GraphEmbedding(
            feat_dim=feat_dim,
            num_layers=num_layers,
            node_dim=node_dim,
            num_heads=num_heads,
            dropout=drop_rate,
        )
        self.fc_prior = nn.Linear(encode_prior_dim, feat_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            GPTransBlock(node_dim=node_dim, edge_dim=edge_dim, feat_dim=feat_dim, 
                         num_heads=num_heads, mlp_ratio=mlp_ratio,
                         drop_act=drop_rate, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                         qkv_bias=qkv_bias, init_values=layer_scale, with_cp=with_cp) for i in range(num_layers)
        ])

    def forward(self, x, da_prior, perturb=None):
        b, c, d, h, w = x.shape
        node_embeds, edge_embeds, padding_mask = self.graph_embedding(
            x,
            perturb=perturb,
        )
        # node_embeds: b c n; edge_embeds: b n n
        da_prior = self.fc_prior(da_prior)
        if self.random_feature and self.training:  # 随机扰动 
            node_embeds += torch.rand_like(node_embeds)
            edge_embeds += torch.rand_like(edge_embeds)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)

        for blk in self.blocks:
            node_embeds, edge_embeds = blk(node_embeds,  # [B, 1+N, C]
                                           edge_embeds,  # [B, C, 1+N, 1+N]
                                           padding_mask,
                                           da_prior) # [B, 1+N, 1]
        x = node_embeds.view(b, c, d, h, w).contiguous()
        return x
