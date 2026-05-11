import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# ==============================================================================
# 1. CÁC MODULE CƠ BẢN (TIỆN ÍCH)
# ==============================================================================
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm2d(nn.Module):
    """Sử dụng LayerNorm thay cho BatchNorm2d để ổn định với Batch Size nhỏ"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ==============================================================================
# KHỐI PRIOR EXTRACTOR (THAY THẾ CHO GMWT VÀ WAVELET PRIOR)
# ==============================================================================
class LightweightPriorExtractor(nn.Module):
    """
    Khối trích xuất đặc trưng cục bộ (Cạnh, cấu trúc) siêu nhẹ.
    THAY THẾ HOÀN TOÀN CHO GMWT VÀ WAVELET PRIOR.
    """
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()

        # Bước 1: Trích xuất đặc trưng không gian (như cạnh, viền)
        # Sử dụng kernel 3x3 để quét qua ảnh đầu vào.
        self.spatial_extract = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1),
            LayerNorm2d(in_channels * 4),
            nn.GELU()
        )

        # Bước 2: Xử lý sâu hơn với chi phí cực thấp bằng Depthwise Convolution
        hidden_dim = in_channels * 4
        self.depthwise_process = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            LayerNorm2d(hidden_dim),
            nn.GELU()
        )

        # Bước 3: Ép số kênh về đúng out_channels để chuẩn bị Concat với nhánh chính
        self.channel_proj = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x):
        feat = self.spatial_extract(x)
        feat = self.depthwise_process(feat)
        out = self.channel_proj(feat)
        return out


# ==============================================================================
# 2. NHÁNH GLOBAL (ATTENTION DỰA TRÊN CỬA SỔ VÀ KÊNH)
# ==============================================================================
class GlobalAttentionBlock(nn.Module):
    """Trích xuất ngữ cảnh toàn cục qua Spatial & Channel Attention, có DWConv tinh chỉnh Artifact"""
    def __init__(self, dim, window_size=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

        self.smooth_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))
            _, _, H_pad, W_pad = x.shape
        else:
            H_pad, W_pad = H, W

        q, k, v = self.qkv(x).chunk(3, dim=1)

        # 1. Channel-Wise Self-Attention
        q_c, k_c, v_c = q.view(B, C, -1), k.view(B, C, -1), v.view(B, C, -1)
        scale_c = (H_pad * W_pad) ** -0.5
        attn_c = (q_c @ k_c.transpose(-2, -1)) * self.temperature * scale_c
        attn_c = attn_c.softmax(dim=-1)
        y_c = (attn_c @ v_c).view(B, C, H_pad, W_pad)

        # 2. Spatial Window Self-Attention
        q_s = q.view(B, C, H_pad // self.window_size, self.window_size, W_pad // self.window_size, self.window_size)
        q_s = q_s.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.window_size**2, C)

        k_s = k.view(B, C, H_pad // self.window_size, self.window_size, W_pad // self.window_size, self.window_size)
        k_s = k_s.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.window_size**2, C)

        v_s = v.view(B, C, H_pad // self.window_size, self.window_size, W_pad // self.window_size, self.window_size)
        v_s = v_s.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.window_size**2, C)

        attn_s = (q_s @ k_s.transpose(-2, -1)) * (C ** -0.5)
        y_s = (attn_s.softmax(dim=-1) @ v_s)

        y_s = y_s.view(B, H_pad // self.window_size, W_pad // self.window_size, self.window_size, self.window_size, C)
        y_s = y_s.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H_pad, W_pad)

        y_s = self.smooth_conv(y_s)

        # 3. Adaptive Interaction
        y_s_hat = y_s * torch.sigmoid(y_c)
        y_c_hat = y_c * torch.sigmoid(y_s)

        out = self.proj(y_s_hat + y_c_hat)

        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :H, :W]

        return out


class GlobalEncoderStage(nn.Module):
    def __init__(self, in_dim, out_dim, window_size=8, apply_downsample=True):
        super().__init__()
        if apply_downsample:
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)
        else:
            self.downsample = nn.Identity()
            out_dim = in_dim

        self.norm1 = LayerNorm2d(out_dim)
        self.attention = GlobalAttentionBlock(out_dim, window_size)

        self.norm2 = LayerNorm2d(out_dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(out_dim, out_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1)
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.attention(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x


# ==============================================================================
# 3. NHÁNH LOCAL (TRÍCH XUẤT ĐẶC TRƯNG ĐỊA PHƯƠNG BẰNG TÍCH CHẬP)
# ==============================================================================
class LocalInceptionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False), LayerNorm2d(dim))
        self.branch2 = nn.Sequential(nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim, bias=False), LayerNorm2d(dim))
        self.branch3 = nn.Sequential(nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim, bias=False), LayerNorm2d(dim))
        self.branch4 = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), LayerNorm2d(dim))
        self.fusion = nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False)

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return self.fusion(out)


class LocalFeatureStage(nn.Module):
    def __init__(self, in_dim, out_dim, apply_downsample=True):
        super().__init__()
        if apply_downsample:
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)
        else:
            self.downsample = nn.Identity()
            out_dim = in_dim

        self.norm1 = LayerNorm2d(out_dim)
        self.local_extract = LocalInceptionBlock(out_dim)

        self.norm2 = LayerNorm2d(out_dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(out_dim, out_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1)
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.local_extract(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x


# ==============================================================================
# 4. BOTTLENECK FUSION (DUNG HỢP THÔNG TIN TOÀN CỤC VÀ ĐỊA PHƯƠNG)
# ==============================================================================
class CrossAttentionModule(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)
        self.combine_heads = nn.Linear(embed_size, embed_size)

        init.xavier_uniform_(self.query_dense.weight); init.constant_(self.query_dense.bias, 0)
        init.xavier_uniform_(self.key_dense.weight); init.constant_(self.key_dense.bias, 0)
        init.xavier_uniform_(self.value_dense.weight); init.constant_(self.value_dense.bias, 0)
        init.xavier_uniform_(self.combine_heads.weight); init.constant_(self.combine_heads.bias, 0)

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x_query, x_kv):
        B, C, H, W = x_query.size()
        x_q = x_query.flatten(2).transpose(1, 2)
        x_k_v = x_kv.flatten(2).transpose(1, 2)

        query = self.split_heads(self.query_dense(x_q), B)
        key = self.split_heads(self.key_dense(x_k_v), B)
        value = self.split_heads(self.value_dense(x_k_v), B)

        attn_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attention = torch.matmul(attn_weights, value).permute(0, 2, 1, 3).contiguous().reshape(B, -1, self.embed_size)

        output = self.combine_heads(attention)
        return output.transpose(1, 2).reshape(B, C, H, W)


class CrossGatingFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.dwconv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = LayerNorm2d(dim)

        self.proj2 = nn.Conv2d(dim, dim, 1)
        self.dwconv2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = LayerNorm2d(dim)

        self.out_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x1, x2):
        z1 = self.norm1(self.dwconv1(self.proj1(x1)))
        z2 = self.norm2(self.dwconv2(self.proj2(x2)))
        return self.out_proj((z1 * F.gelu(z2)) + (z2 * torch.sigmoid(z1)))


class GlobalLocalFusionModule(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm_local = LayerNorm2d(dim)
        self.norm_global = LayerNorm2d(dim)

        self.cross_attention = CrossAttentionModule(dim, num_heads)
        self.cross_gating = CrossGatingFusion(dim)

    def forward(self, x_local, x_global):
        x_cross = self.cross_attention(x_query=self.norm_local(x_local),
                                       x_kv=self.norm_global(x_global))

        out = self.cross_gating(x_local, x_cross)
        return out + x_local


# ==============================================================================
# 5. REFINEMENT CUỐI (TINH CHỈNH ẢNH TRƯỚC KHI XUẤT RA)
# ==============================================================================
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw_channel = c * DW_Expand

        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channel // 2, dw_channel // 2, 1))
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)

        self.norm2 = LayerNorm2d(c)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv3(self.sg(self.conv2(self.conv1(x))) * self.sca(self.sg(self.conv2(self.conv1(x)))))
        y = inp + x * self.beta

        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class ImageRefinementModule(nn.Module):
    def __init__(self, in_size=32, out_size=3, dim=32, num_blocks=2):
        super().__init__()
        self.intro = nn.Conv2d(in_size, dim, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[NAFBlock(c=dim) for _ in range(num_blocks)])
        self.outro = nn.Conv2d(dim, out_size, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.intro(x)
        feat_refined = self.body(feat) + feat
        return torch.sigmoid(self.outro(feat_refined))
