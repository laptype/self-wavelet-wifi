import torch
import torch.nn as nn
from .torch_wavelets_1D import DWT_1D, IDWT_1D


class WaveAttention_test(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.dwt = DWT_1D(wave='haar')
        self.idwt = IDWT_1D(wave='haar')

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        
        x = self.dwt(x)
        x = self.idwt(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__ == '__main__':

    import numpy as np
    # def _pickup_patching(batch_data):
    #     # batch_size, n_channels, seq_len
    #     batch_size, n_channels, seq_len = batch_data.size()
    #     patch_size = 16
    #     assert seq_len % patch_size == 0
    #     batch_data = batch_data.view(batch_size, n_channels, seq_len // patch_size, patch_size)
    #     batch_data = batch_data.permute(0, 2, 1, 3)
    #     batch_data = batch_data.reshape(batch_size, seq_len // patch_size, n_channels * patch_size)
    #     return batch_data
    # inputs = np.ones((2, 30, 1600))
    # inputs = torch.from_numpy(inputs).float().to(torch.device('cuda'))
    # print(inputs.shape)
    # inputs = _pickup_patching(inputs)
    # print(inputs.shape)
    # wave_attn = WaveAttention2(dim=480).to(torch.device('cuda'))
    # wave_attn(inputs)