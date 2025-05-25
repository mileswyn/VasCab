import torch
import torch.nn as nn
import torch.nn.functional as F
from .domain_prior_conv_layers import BasicBlock, Bottleneck, ConvNormAct
from .domain_prior_trans_layers import Attention, CrossAttention, LayerNorm, Mlp, PreNorm
import pdb
from einops import rearrange

class HierarchyPriorClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
            
        self.norm = nn.LayerNorm(in_dim)
        self.classifier_pred = nn.Sequential(
            Mlp(in_dim=in_dim, out_dim=out_dim),
            Mlp(in_dim=out_dim, out_dim=out_dim)
            )   
        #TODO
        self.dimension_reduction_classifier = nn.Sequential(
            Mlp(in_dim=50, out_dim=2),
            Mlp(in_dim=2, out_dim=2)
            )   
                
        self.classifier_pred.apply(self.init_weights)
        
    def init_weights(self, m): 
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)

    def forward(self, x, prior_list):
        priors = torch.cat(prior_list, dim=2)
        priors = self.norm(priors)
        weights = self.classifier_pred(priors) # B, n, dim
            
        B, C, D, H, W = x.shape
        #x = rearrange(x, 'b c d h w -> b (d h w) c', b=B, c=C, d=D, h=H, w=W)
        x = x.view(B, C, -1) 
        x = x.permute(0, 2, 1).contiguous()
            
        weights = torch.permute(weights, (0, 2, 1)) 

        weights = self.dimension_reduction_classifier(weights)
            
        output = torch.bmm(x, weights)
            
        #output = rearrange(output, 'b (d h w) c -> b c d h w', b=B, c=weights.shape[2], d=D, h=H, w=W)
        c = weights.shape[2]
        output = output.permute(0, 2, 1).contiguous()
        output = output.view(B, c, D, H, W).contiguous()

        return output
    
class ModalityClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(in_dim)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = Mlp(in_dim=in_dim, out_dim=out_dim)

    def forward(self, prior_list):

        priors = torch.cat(prior_list, dim=2)
        priors = self.norm(priors)

        priors = torch.permute(priors, (0, 2, 1))
        priors = self.avg_pool(priors).squeeze(2)


        pred = self.classifier(priors)

        return pred

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], block=BasicBlock, norm=nn.BatchNorm3d):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        pad_size = [i//2 for i in kernel_size]
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=pad_size, bias=False)
        self.conv2 = block(out_ch, out_ch, kernel_size=kernel_size, norm=norm)

    def forward(self, x): 
        out = self.conv1(x)
        out = self.conv2(out)

        return out 


class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, kernel_size=[3,3,3], down_scale=[2,2,2], pool=True, norm=nn.BatchNorm3d):
        super().__init__() 
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(down_scale, int):
            down_scale = [down_scale] * 3

        block_list = []

        if pool:
            block_list.append(nn.MaxPool3d(down_scale))
            block_list.append(block(in_ch, out_ch, kernel_size=kernel_size, norm=norm))
        else:
            block_list.append(block(in_ch, out_ch, stride=down_scale, kernel_size=kernel_size, norm=norm))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, stride=1, kernel_size=kernel_size, norm=norm))

        self.conv = nn.Sequential(*block_list)
    def forward(self, x):
        return self.conv(x)

class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, kernel_size=[3,3,3], up_scale=[2,2,2], norm=nn.BatchNorm3d):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(up_scale, int):
            up_scale = [up_scale] * 3

        self.up_scale = up_scale


        block_list = []

        block_list.append(block(in_ch+out_ch, out_ch, kernel_size=kernel_size, norm=norm))
        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, kernel_size=kernel_size, norm=norm))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        input_dtype = x1.dtype
        # F.interpolate trilinear doesn't support bfloat16, so need to cast to float32 for upsampling then cast back if using amp training
        x1 = F.interpolate(x1.float(), size=x2.shape[2:], mode='trilinear', align_corners=True)
        x1 = x1.to(input_dtype)
        out = torch.cat([x2, x1], dim=1)

        out = self.conv(out)

        return out

# class DualPreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.fn = fn
#     def forward(self, x1, x2, **kwargs):
#         return self.fn(self.norm1(x1), self.norm2(x2), **kwargs)


# class PriorAttentionBlock(nn.Module):
#     def __init__(self, feat_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0.):
#         super().__init__()

#         self.inner_dim = dim_head * heads
#         self.feat_dim = feat_dim
#         self.heads = heads
#         self.scale = dim_head ** (-0.5)
#         self.dim_head = dim_head

#         dim = feat_dim
#         mlp_dim = dim * 4

#         # update priors by aggregating from the feature map
#         self.prior_aggregate_block = DualPreNorm(dim, CrossAttention(dim, heads, dim_head, attn_drop, proj_drop))
#         self.prior_ffn = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))

#         # update the feature map by injecting knowledge from the priors
#         self.feat_aggregate_block = DualPreNorm(dim, CrossAttention(dim, heads, dim_head, attn_drop, proj_drop))
#         self.feat_ffn = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))


#     def forward(self, x1, x2):
#         # x1: image feature map, x2: priors

#         x2 = self.prior_aggregate_block(x2, x1, prior=True) + x2
#         x2 = self.prior_ffn(x2) + x2

#         x1 = self.feat_aggregate_block(x1, x2, prior=False) + x1
#         x1 = self.feat_ffn(x1) + x1

#         return x1, x2


# # class DomainPriorInitFusionLayer(nn.Module):
# #     def __init__(self, feat_dim, prior_dim, block_num=2, task_prior_num=42, modality_prior_num=2, l=10):
# #         super().__init__()
        
# #         # random initialize the priors
# #         self.task_prior = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(task_prior_num+1, prior_dim))) # +1 for null token
# #         self.modality_prior = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(modality_prior_num, l, prior_dim)))

# #         self.attn_layers = nn.ModuleList([])
# #         for i in range(block_num):
# #             self.attn_layers.append(PriorAttentionBlock(feat_dim, heads=feat_dim//32, dim_head=32, attn_drop=0, proj_drop=0))

# #     def forward(self, x, tgt_idx, mod_idx):
# #         # x: image feature map, tgt_idx: target task index, mod_idx: modality index
# #         B, C, D, H, W = x.shape
        
# #         task_prior_list = []
# #         modality_prior_list = []
# #         # prior selection
# #         for i in range(B):
# #             idxs = tgt_idx[i]
# #             task_prior_list.append(torch.stack([self.task_prior[[0], :],self.task_prior[idxs, :]], dim=0).squeeze(1))
# #             modality_prior_list.append(self.modality_prior[mod_idx[i], :, :])
        

# #         task_priors = torch.stack(task_prior_list)
# #         modality_priors = torch.stack(modality_prior_list)
# #         modality_priors = modality_priors.squeeze(1)

# #         priors = torch.cat([task_priors, modality_priors], dim=1)
        
# #         #x = rearrange(x, 'b c d h w -> b (d h w) c', d=D, h=H, w=W)
# #         b, c, d, h, w = x.shape
# #         x = x.view(b, c, -1)
# #         x = x.permute(0, 2, 1).contiguous()

        
# #         for layer in self.attn_layers:
# #             x, priors = layer(x, priors)
        
# #         #x = rearrange(x, 'b (d h w) c -> b c d h w', d=D, h=H, w=W, c=C)
# #         x = x.permute(0, 2, 1)
# #         x = x.view(b, c, d, h, w).contiguous()

# #         return x, priors
# class DomainPriorInitFusionLayer(nn.Module):
#     def __init__(self, feat_dim, prior_dim, block_num=2, task_prior_num=42, modality_prior_num=9, l=50):
#         super().__init__()
        
#         # random initialize the priors
#         self.domain_prior = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(modality_prior_num, l, prior_dim)))

#         self.attn_layers = nn.ModuleList([])
#         for i in range(block_num):
#             self.attn_layers.append(PriorAttentionBlock(feat_dim, heads=feat_dim//32, dim_head=32, attn_drop=0, proj_drop=0))

#     def forward(self, x, mod_idx):
#         # x: image feature map, tgt_idx: target task index, mod_idx: modality index
#         B, C, D, H, W = x.shape

#         domain_prior_list = []
#         # prior selection
#         for i in range(B):
#             # idxs = tgt_idx[i]
#             domain_prior_list.append(self.domain_prior[mod_idx[i], :, :])
        
#         domain_priors = torch.stack(domain_prior_list)
#         domain_priors = domain_priors.squeeze(1)
        
#         #x = rearrange(x, 'b c d h w -> b (d h w) c', d=D, h=H, w=W)
#         b, c, d, h, w = x.shape
#         x = x.view(b, c, -1)
#         x = x.permute(0, 2, 1).contiguous()

        
#         for layer in self.attn_layers:
#             x, domain_priors = layer(x, domain_priors)
        
#         #x = rearrange(x, 'b (d h w) c -> b c d h w', d=D, h=H, w=W, c=C)
#         x = x.permute(0, 2, 1)
#         x = x.view(b, c, d, h, w).contiguous()

#         return x, domain_priors
    
class DualPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm1(x1), self.norm2(x2), **kwargs)
    
class CrossAttention_1(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0.):
        super().__init__()

        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias=False)

        # self.qkv = nn.Linear(dim, dim * 3, bias=True)
        # self.proj = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)


    def b_l_hd__b_h_l_d(self, x, heads):
        b, l, n = x.shape
        h = heads
        d = int(n / h)

        x = x.view(b, l, h, d)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def b_h_l_d__b_l_hd(self, x):
        b, h, l, d = x.shape

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, l, -1).contiguous()
        return x


    def forward(self, x1, x2, prior=True):
        # B, S1, C = x1.shape
        # _, S2, _ = x2.shape

        # multi-head self-attention
        # qkv0 = self.qkv(torch.cat([x1, x2], dim=1))
        # qkv = qkv0.reshape(B, S1+S2, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.reshape(3, B * self.heads, S1+S2, -1).unbind(0)

        # attn = (q * self.scale) @ k.transpose(-2, -1)
        # attn = attn.softmax(dim=-1)
        # x = (attn @ v).view(B, self.heads, S1+S2, -1).permute(0, 2, 1, 3).reshape(B, S1+S2, -1)
        # if prior:
        #     attned1 = self.proj(x)[:,:S1,:]
        # else:
        #     attned1 = self.proj(x)[:,S2:,:]

        # aggregate from x2 to x1
        q = self.to_q(x1)
        k, v = self.to_kv(x2).chunk(2, dim=-1)

        #q, k, v = map(lambda t: rearrange(t, 'b l (heads dim_head) -> b heads l dim_head', heads=self.heads), [q, k, v])
        q, k, v = map(lambda t: self.b_l_hd__b_h_l_d(t, self.heads), [q, k, v])

        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = F.softmax(attn, dim=-1)

        attned = torch.einsum('bhij,bhjd->bhid', attn, v)
        #attned = rearrange(attned, 'b heads l dim_head -> b l (dim_head heads)')
        attned = self.b_h_l_d__b_l_hd(attned)

        attned = self.to_out(attned)

        return attned
    
class CrossAttention_2(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0.):
        super().__init__()

        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias=False)

        # self.qkv = nn.Linear(dim, dim * 3, bias=True)
        # self.proj = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)


    def b_l_hd__b_h_l_d(self, x, heads):
        b, l, n = x.shape
        h = heads
        d = int(n / h)

        x = x.view(b, l, h, d)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def b_h_l_d__b_l_hd(self, x):
        b, h, l, d = x.shape

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, l, -1).contiguous()
        return x


    def forward(self, x1, x2, prior=True):
        # B, S1, C = x1.shape
        # _, S2, _ = x2.shape

        # multi-head self-attention
        # qkv0 = self.qkv(torch.cat([x1, x2], dim=1))
        # qkv = qkv0.reshape(B, S1+S2, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.reshape(3, B * self.heads, S1+S2, -1).unbind(0)

        # attn = (q * self.scale) @ k.transpose(-2, -1)
        # attn = attn.softmax(dim=-1)
        # x = (attn @ v).view(B, self.heads, S1+S2, -1).permute(0, 2, 1, 3).reshape(B, S1+S2, -1)
        # if prior:
        #     attned1 = self.proj(x)[:,:S1,:]
        # else:
        #     attned1 = self.proj(x)[:,S2:,:]

        # x1:featmap; x2: prior
        q = self.to_q(x1)
        k, v = self.to_kv(x2).chunk(2, dim=-1)

        #q, k, v = map(lambda t: rearrange(t, 'b l (heads dim_head) -> b heads l dim_head', heads=self.heads), [q, k, v])
        q, k, v = map(lambda t: self.b_l_hd__b_h_l_d(t, self.heads), [q, k, v])

        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = F.softmax(attn, dim=-1)

        attned = torch.einsum('bhij,bhjd->bhid', attn, v)
        #attned = rearrange(attned, 'b heads l dim_head -> b l (dim_head heads)')
        attned = self.b_h_l_d__b_l_hd(attned)

        attned = self.to_out(attned)

        return attned


class DPAM(nn.Module):
    def __init__(self, feat_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.feat_dim = feat_dim
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head

        dim = feat_dim
        mlp_dim = dim * 4

        # update priors by aggregating from the feature map
        self.prior_aggregate_block = DualPreNorm(dim, CrossAttention_1(dim, heads, dim_head, attn_drop, proj_drop))
        self.prior_ffn = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))

        # update the feature map by injecting knowledge from the priors
        self.feat_aggregate_block = DualPreNorm(dim, CrossAttention_2(dim, heads, dim_head, attn_drop, proj_drop))
        self.feat_ffn = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))


    def forward(self, x1, x2):
        # x1: image feature map, x2: priors

        x2 = self.prior_aggregate_block(x2, x1) + x2 # x1 [2 38400 128] x2 [2 50 128]
        x2 = self.prior_ffn(x2) + x2 # x2 [2 50 128]

        x1 = self.feat_aggregate_block(x1, x2) + x1
        x1 = self.feat_ffn(x1) + x1

        return x1, x2


class TPFM(nn.Module):
    def __init__(self, d, w, h, input_channels=2048):
        super(TPFM, self).__init__()
        self.input_channels = input_channels
        BatchNorm1d = nn.BatchNorm1d
        BatchNorm3d = nn.BatchNorm3d

        self.topo_prompt = nn.Parameter(torch.FloatTensor(torch.zeros((d, w, h, 6))), requires_grad=True)

        self.edge_aggregation_func = nn.Sequential(
            nn.Linear(6, 1),
            BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.vertex_update_func = nn.Sequential(
            nn.Linear(2 * input_channels, input_channels // 2),
            BatchNorm1d(input_channels // 2),
            nn.LeakyReLU(inplace=True),
        )

        self.edge_update_func = nn.Sequential(
            nn.Linear(2 * input_channels, input_channels // 2),
            BatchNorm1d(input_channels // 2),
            nn.LeakyReLU(inplace=True),
        )
        self.update_edge_reduce_func = nn.Sequential(
            nn.Linear(6, 1),
            BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
        )

        self.final_aggregation_layer = nn.Sequential(
            nn.Conv3d(input_channels + input_channels // 2, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm3d(input_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, input):
        x = input
        B, C, D, W, H = x.size()
        
        vertex = input
        edge = torch.stack(
            (
                torch.cat((input[:,:,-1:], input[:,:,:-1]), dim=2),
                torch.cat((input[:,:,1:], input[:,:,:1]), dim=2),
                torch.cat((input[:,:,:,-1:], input[:,:,:,:-1]), dim=3),
                torch.cat((input[:,:,:,1:], input[:,:,:,:1]), dim=3),
                torch.cat((input[:,:,:,:,-1:], input[:,:,:,:,:-1]), dim=4),
                torch.cat((input[:,:,:,:,1:], input[:,:,:,:,:1]), dim=4)
            ), dim=-1 # b c d w h 6
        ) * input.unsqueeze(dim=-1) + self.topo_prompt # b c d w h 6

        aggregated_edge = self.edge_aggregation_func(
            edge.reshape(-1, 6) # bcdwh 6
        ).reshape((B, C, D, W, H)) # b c d w h
        cat_feature_for_vertex = torch.cat((vertex, aggregated_edge), dim=1) # b 2c d w h
        update_vertex = self.vertex_update_func(
            cat_feature_for_vertex.permute(0, 2, 3, 4, 1).reshape((-1, 2 * self.input_channels)) # bdwh 2c
        ).reshape((B, D, W, H, self.input_channels // 2)).permute(0, 4, 1, 2, 3) # b c//2 d w h

        cat_feature_for_edge = torch.cat(
            (
                torch.stack((vertex, vertex, vertex, vertex, vertex, vertex), dim=-1), # b c d w h 6
                edge # b c d w h 6
            ), dim=1
        ).permute(0, 2, 3, 4, 5, 1).reshape((-1, 2 * self.input_channels)) # 6bdwh 2c
        update_edge = self.edge_update_func(cat_feature_for_edge).reshape((B, D, W, H, 6, C//2)).permute(0, 5, 1, 2, 3, 4).reshape((-1, 6)) # bc//2dwh 6
        update_edge_converted = self.update_edge_reduce_func(update_edge).reshape((B, C//2, D, W, H))

        update_feature = update_vertex * update_edge_converted
        output = self.final_aggregation_layer(
            torch.cat((x, update_feature), dim=1)
        )

        return output
    
class DomainPriorInitFusionLayer(nn.Module):
    def __init__(self, feat_dim, prior_dim, feature_size, task_prior_num=42, modality_prior_num=9, l=50):
        super().__init__()
        
        self.d, self.w, self.h = feature_size[0], feature_size[1], feature_size[2]
        # random initialize the priors
        self.domain_prior = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(modality_prior_num, l, prior_dim)))
        # self.topology_prior = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((self.d, self.w, self.h, 6))))

        self.attn_layers = DPAM(feat_dim, heads=feat_dim//32, dim_head=32, attn_drop=0, proj_drop=0)
        self.tp_layers = TPFM(self.d, self.w, self.h, feat_dim)

        self.conv_out = nn.Conv3d(feat_dim*2, feat_dim, 1)

    def forward(self, x, mod_idx):
        # x: image feature map, tgt_idx: target task index, mod_idx: modality index
        B, C, D, H, W = x.shape

        domain_prior_list = []
        # prior selection
        for i in range(B):
            # idxs = tgt_idx[i]
            domain_prior_list.append(self.domain_prior[mod_idx[i], :, :])
        
        domain_priors = torch.stack(domain_prior_list)
        domain_priors = domain_priors.squeeze(1)
        
        #x = rearrange(x, 'b c d h w -> b (d h w) c', d=D, h=H, w=W)
        b, c, d, h, w = x.shape
        
        x_dp, domain_priors = self.attn_layers(x.view(b, c, -1).permute(0, 2, 1).contiguous(), domain_priors)
        x_tp = self.tp_layers(x)
        
        #x = rearrange(x, 'b (d h w) c -> b c d h w', d=D, h=H, w=W, c=C)
        x_dp = x_dp.permute(0, 2, 1)
        x_dp = x_dp.view(b, c, d, h, w).contiguous()

        x = torch.cat([x_dp, x_tp], dim=1)
        x = self.conv_out(x)

        return x, domain_priors
