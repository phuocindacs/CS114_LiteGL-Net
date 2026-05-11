import os
import torch
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.utils as vutils

try:
    import wandb
except ImportError:
    pass

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.models.model import LightweightDualBranchUNet
from src.train.loss import CombinedLoss


class LYTLightning(pl.LightningModule):
    def __init__(self, learning_rate=2e-4, max_epochs=1000):
        super().__init__()
        self.save_hyperparameters()

        model = LightweightDualBranchUNet(base_dim=18, num_heads=2)
        self.model = torch.compile(model)

        # Loss Model cần device, nên ta khởi tạo lazy trong on_fit_start
        self.criterion = None

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        # LPIPS: net_type='vgg', normalize=True giả định input vào khoảng [0, 1]
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)

    def on_fit_start(self):
        """Khởi tạo loss sau khi module đã được map lên đúng thiết bị GPU"""
        self.criterion = CombinedLoss(self.device)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }

    def training_step(self, batch, batch_idx):
        low_imgs, high_imgs, img_names = batch
        enhanced_imgs = self.forward(low_imgs)

        # Loss có giám sát: Giữa ảnh được dự đoán và ảnh GroundTruth (high_imgs)
        loss = self.criterion(high_imgs, enhanced_imgs)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        low_imgs, high_imgs, img_names = batch
        enhanced_imgs = self.forward(low_imgs)

        val_psnr = self.psnr(enhanced_imgs, high_imgs)
        val_ssim = self.ssim(enhanced_imgs, high_imgs)

        # Do normalize=True ở __init__, input [0, 1] là hợp lệ.
        val_lpips = self.lpips(enhanced_imgs, high_imgs)

        self.log("val_psnr", val_psnr, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_ssim", val_ssim, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_lpips", val_lpips, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            self._save_sample_images(low_imgs, enhanced_imgs, high_imgs, stage="val")

    def test_step(self, batch, batch_idx):
        low_imgs, high_imgs, img_names = batch
        enhanced_imgs = self.forward(low_imgs)

        test_psnr = self.psnr(enhanced_imgs, high_imgs)
        test_ssim = self.ssim(enhanced_imgs, high_imgs)
        test_lpips = self.lpips(enhanced_imgs, high_imgs)

        self.log("test_psnr", test_psnr, on_epoch=True, prog_bar=True)
        self.log("test_ssim", test_ssim, on_epoch=True, prog_bar=True)
        self.log("test_lpips", test_lpips, on_epoch=True, prog_bar=True)

        self._save_sample_images(low_imgs, enhanced_imgs, high_imgs, stage="test", img_name=img_names[0])

    def _save_sample_images(self, low, enh, high, stage, img_name="sample"):
        out_dir = f"visualize/{stage}"
        os.makedirs(out_dir, exist_ok=True)
        grid = torch.cat([low[0], enh[0], high[0]], dim=2)
        grid_to_log = grid.detach().cpu().clamp(0, 1)
        filename = f"epoch_{self.current_epoch}" if stage == "val" else img_name

        vutils.save_image(grid_to_log, os.path.join(out_dir, f"{filename}.png"))

        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log({
                f"Images/{stage}": [wandb.Image(grid_to_log, caption=f"{stage.upper()} - {filename} (Input | Enhanced | GT)")]
            })
