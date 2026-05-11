import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import (
    LightweightPriorExtractor,
    GlobalEncoderStage, LocalFeatureStage,
    GlobalLocalFusionModule, ImageRefinementModule
)


# ==============================================================================
# MÔ HÌNH CHÍNH: DUAL-BRANCH U-NET VỚI PRIOR EXTRACTOR
# ==============================================================================
class LightweightDualBranchUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_dim=16, num_heads=2, window_size=8):
        super().__init__()

        self.stem = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

        # --- ENCODER ---
        self.enc_g1 = GlobalEncoderStage(base_dim, base_dim * 2, window_size, apply_downsample=True)
        self.enc_l1 = LocalFeatureStage(base_dim, base_dim * 2, apply_downsample=True)

        self.enc_g2 = GlobalEncoderStage(base_dim * 2, base_dim * 4, window_size, apply_downsample=True)
        self.enc_l2 = LocalFeatureStage(base_dim * 2, base_dim * 4, apply_downsample=True)

        # --- PRIOR EXTRACTOR (thay thế GMWT Wavelet Prior) ---
        # Prior cho Stage 1: nhận ảnh RGB (3 kênh), xuất ra số kênh = đầu ra của enc_1 (base_dim * 2)
        self.prior_s1 = LightweightPriorExtractor(in_channels=3, out_channels=base_dim * 2)
        self.reduce_s1_g = nn.Conv2d(base_dim * 2 * 2, base_dim * 2, kernel_size=1)
        self.reduce_s1_l = nn.Conv2d(base_dim * 2 * 2, base_dim * 2, kernel_size=1)

        # Prior cho Stage 2: nhận ảnh RGB (3 kênh), xuất ra số kênh = đầu ra của enc_2 (base_dim * 4)
        self.prior_s2 = LightweightPriorExtractor(in_channels=3, out_channels=base_dim * 4)
        self.reduce_s2_g = nn.Conv2d(base_dim * 4 * 2, base_dim * 4, kernel_size=1)
        self.reduce_s2_l = nn.Conv2d(base_dim * 4 * 2, base_dim * 4, kernel_size=1)

        # --- BOTTLENECK FUSION ---
        self.fusion = GlobalLocalFusionModule(base_dim * 4, num_heads)

        # --- DECODER ---
        # Concat: fused_s4 (base_dim*4) + g2 (base_dim*4) + l2 (base_dim*4) = base_dim * 12
        self.reduce_s3 = nn.Conv2d(base_dim * 12, base_dim * 4, kernel_size=1)
        self.dec_s3 = LocalFeatureStage(base_dim * 4, base_dim * 4, apply_downsample=False)

        self.up_s2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_dim * 4, base_dim * 2, kernel_size=3, padding=1)
        )
        self.reduce_s2 = nn.Conv2d(base_dim * 6, base_dim * 2, kernel_size=1)
        self.dec_s2 = LocalFeatureStage(base_dim * 2, base_dim * 2, apply_downsample=False)

        self.up_s = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_dim * 2, base_dim, kernel_size=3, padding=1)
        )
        self.reduce_s = nn.Conv2d(base_dim * 2, base_dim, kernel_size=1)
        self.dec_s = LocalFeatureStage(base_dim, base_dim, apply_downsample=False)

        # --- REFINEMENT ---
        self.refinement = ImageRefinementModule(in_size=base_dim, out_size=out_channels, dim=base_dim, num_blocks=4)

    def forward(self, x):
        feat_s = self.stem(x)

        # ==========================================
        # STAGE 1: ENCODER & PRIOR INJECTION
        # ==========================================
        g1, l1 = self.enc_g1(feat_s), self.enc_l1(feat_s)

        # Tạo ảnh downsample để khớp kích thước H/2, W/2 với đầu ra của Stage 1
        x_down1 = F.interpolate(x, size=g1.shape[2:], mode='bilinear', align_corners=False)
        prior_1 = self.prior_s1(x_down1)

        # Nối (Concat) Prior vào Nhánh Global & Local rồi Ép kênh
        g1 = self.reduce_s1_g(torch.cat([g1, prior_1], dim=1))
        l1 = self.reduce_s1_l(torch.cat([l1, prior_1], dim=1))

        # ==========================================
        # STAGE 2: ENCODER & PRIOR INJECTION
        # ==========================================
        g2, l2 = self.enc_g2(g1), self.enc_l2(l1)

        # Tạo ảnh downsample để khớp kích thước H/4, W/4 với đầu ra của Stage 2
        x_down2 = F.interpolate(x, size=g2.shape[2:], mode='bilinear', align_corners=False)
        prior_2 = self.prior_s2(x_down2)

        # Nối (Concat) Prior vào Nhánh Global & Local rồi Ép kênh
        g2 = self.reduce_s2_g(torch.cat([g2, prior_2], dim=1))
        l2 = self.reduce_s2_l(torch.cat([l2, prior_2], dim=1))

        # --- BOTTLENECK FUSION ---
        fused_s4 = self.fusion(x_local=l2, x_global=g2)

        # --- DECODER ---
        d3 = self.dec_s3(self.reduce_s3(torch.cat([fused_s4, g2, l2], dim=1)))
        d2 = self.up_s2(d3)
        if d2.shape[2:] != g1.shape[2:]:
            d2 = F.interpolate(d2, size=g1.shape[2:], mode='bilinear', align_corners=False)

        d2 = self.dec_s2(self.reduce_s2(torch.cat([d2, g1, l1], dim=1)))
        d_s = self.up_s(d2)
        if d_s.shape[2:] != feat_s.shape[2:]:
            d_s = F.interpolate(d_s, size=feat_s.shape[2:], mode='bilinear', align_corners=False)

        d_s = self.dec_s(self.reduce_s(torch.cat([d_s, feat_s], dim=1)))

        return self.refinement(d_s)
