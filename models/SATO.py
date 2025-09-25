import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class Physics_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Serialized_Attention(nn.Module):
    def __init__(self, patch_size, shift, dim, num_heads, dropout=0.1):
        super(Serialized_Attention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.patch_size = patch_size
        self.shift = shift
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        index = torch.tensor([i for i in range(0, patch_size*shift, shift)], dtype=torch.int64)[None, ...]
        self.group_index = torch.cat([index+i for i in range(shift)], dim=0)
        
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # padding
        pad_size = int((self.patch_size*self.shift) - N % (self.patch_size*self.shift))
        x_pad = torch.cat([x, torch.zeros(B, pad_size, C).to(x.device)], dim=1)

        # index_generation
        index = self.group_index
        index_list = [self.group_index]
        while index[-1,-1].item() < N:
            index = index + (self.patch_size*self.shift)
            index_list.append(index)
        patch_index = torch.cat(index_list, dim=0)
        
        # pad2patch
        x_patch = x_pad[:, patch_index, :] # (B, patch_num, patch_size, C)
        
        # patch attention
        x_patch = rearrange(x_patch, 'b n s c -> (b n) s c')
        B_p, S_p, C = x_patch.shape
        
        qkv = self.qkv_proj(x_patch)
        qkv = qkv.reshape(B_p, S_p, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q_token, k_token, v_token = qkv[0], qkv[1], qkv[2]
        
        dots = torch.matmul(q_token, k_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        attn_token = torch.matmul(attn, v_token)
        
        attn_token = attn_token.transpose(1, 2).reshape(B_p, S_p, self.dim)
        out_token = attn_token
        
        out_token = rearrange(out_token, '(b n) s c -> b n s c', b=B)
        
        # patch2pad
        x_pad[:, patch_index, :] = out_token
        
        return self.out_proj(x_pad[:, :N, :])

    
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class SATO_block(nn.Module):
    """Spatially-AwareTransformer Operator block."""
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            slice_num=32,
            patch_size=20,
            shift=1
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.global_attention = Physics_Attention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                         dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        
        self.local_ln_1 = nn.LayerNorm(hidden_dim)
        self.local_attention = Serialized_Attention(patch_size, shift, hidden_dim, num_heads, dropout=0.1)
        self.local_ln_2 = nn.LayerNorm(hidden_dim)
        self.local_gate = nn.Parameter(torch.tensor([0.0]))

        self.ln_3 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        
        if self.last_layer:
            self.ln_4 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, fx, order, inverse):
        x0 = fx 
        fxn = self.ln_1(fx)

        # global_attention
        fx1 = self.global_attention(fxn)
        
        # local_attention
        fx2 = torch.zeros_like(fx1)
        for i in range(fx2.shape[0]):
            fx2[i] = (fxn - fx1)[i, order[i], :] # serialized
        fx2 = self.local_ln_2(self.local_attention(self.local_ln_1(fx2)))
        for i in range(fx2.shape[0]):
            fx2[i] = fx2[i, inverse[i], :]  # deserialized
        
        fx = self.mlp(self.ln_3(x0 + fx1 + self.local_gate * fx2)) + x0
        
        if self.last_layer:
            return self.mlp2(self.ln_4(fx))
        else:
            return fx
        

class Model(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 slice_num=32,
                 patch_size=20,
                 shift=1,
                 n_iter=1
                 ):
        super(Model, self).__init__()
        self.__name__ = 'SATO'

        self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.n_layers = n_layers
        self.n_iter = n_iter

        self.blocks = nn.ModuleList([SATO_block(num_heads=n_head, hidden_dim=n_hidden,
                                      dropout=dropout,
                                      act=act,
                                      mlp_ratio=mlp_ratio,
                                      slice_num=slice_num,
                                      patch_size=patch_size,
                                      shift=shift,
                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, fx_surf, order, inverse):
        fx_surf = self.preprocess(fx_surf)
        fx_surf = fx_surf + self.placeholder[None, None, :]
        
        if self.n_iter == 1:
            for i in range(self.n_layers):
                serialization_type = int(i % order.shape[1])
                fx_surf = self.blocks[i](fx_surf, order[:, serialization_type, :], inverse[:, serialization_type, :])

        else:
            for _ in range(self.n_iter):
                for i in range(self.n_layers-1):
                    serialization_type = int(i % order.shape[1])
                    fx_surf = self.blocks[i](fx_surf, order[:, serialization_type, :], inverse[:, serialization_type, :])

                i = self.n_layers - 1
                serialization_type = int(i % order.shape[1])
                fx_surf = self.blocks[i](fx_surf, order[:, serialization_type, :], inverse[:, serialization_type, :])

        return fx_surf[0]
