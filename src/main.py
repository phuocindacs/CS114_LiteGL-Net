import argparse
import os
import torch
import pytorch_lightning as pl
from src.data.dataset import LOLDataModule
from src.models.model import DBWUNetLightning

def train(args):
    print(f"Bắt đầu huấn luyện với cấu hình: {args}")
    # Placeholder for actual training logic
    # data_module = LOLDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=2, patch_size=256)
    # model = DBWUNetLightning(learning_rate=args.lr)
    # trainer = pl.Trainer(max_epochs=args.epochs, default_root_dir=args.ckpt_dir)
    # trainer.fit(model, datamodule=data_module)
    print("Training process placeholder... (cần bổ sung trainer.fit)")

def eval_model(args):
    print(f"Bắt đầu đánh giá mô hình với checkpoint: {args.ckpt}")
    if not os.path.exists(args.ckpt):
        print(f"Lỗi: Không tìm thấy file {args.ckpt}")
        return
    
    # Placeholder for actual eval logic
    # model = DBWUNetLightning.load_from_checkpoint(args.ckpt)
    # model.eval()
    print("Eval process placeholder... (cần bổ sung dữ liệu test và chạy mô hình)")

def main():
    parser = argparse.ArgumentParser(description="DB-WUNet CLI for Training and Evaluation")
    subparsers = parser.add_subparsers(dest="command", help="Lệnh thực thi")
    
    # Lệnh train
    parser_train = subparsers.add_parser("train", help="Huấn luyện mô hình")
    parser_train.add_argument("--data_dir", type=str, required=True, help="Đường dẫn đến dataset")
    parser_train.add_argument("--batch_size", type=int, default=1, help="Kích thước batch")
    parser_train.add_argument("--epochs", type=int, default=200, help="Số epoch")
    parser_train.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser_train.add_argument("--ckpt_dir", type=str, default="./ckpts", help="Thư mục lưu checkpoint")
    
    # Lệnh eval
    parser_eval = subparsers.add_parser("eval", help="Đánh giá mô hình")
    parser_eval.add_argument("--ckpt", type=str, required=True, help="Đường dẫn đến file checkpoint (.ckpt)")
    parser_eval.add_argument("--data_dir", type=str, required=True, help="Đường dẫn đến thư mục chứa dữ liệu đánh giá")

    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        eval_model(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
