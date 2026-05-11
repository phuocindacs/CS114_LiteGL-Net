#!/bin/bash

# Kịch bản chạy huấn luyện mẫu
# Đảm bảo bạn đã cài đặt các thư viện trong requirements.txt

# Đường dẫn đến thư mục chứa dữ liệu
DATA_DIR="/đường/dẫn/đến/lol_dataset"

# Thư mục lưu checkpoint
CKPT_DIR="./ckpts"

# Chạy lệnh huấn luyện
python -m src.main train --data_dir "$DATA_DIR" \
                         --batch_size 2 \
                         --epochs 200 \
                         --lr 0.0002 \
                         --ckpt_dir "$CKPT_DIR"
