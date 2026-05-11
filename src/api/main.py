import os
import io
import gc
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from PIL import Image
from src.models.model import LightweightDualBranchUNet

app = FastAPI(title="DB-WUNet Image Enhancement API")

# Mặc định sử dụng ckpt v2 theo yêu cầu
CKPT_PATH = os.path.join(os.path.dirname(__file__), "../../ckpts/lolv2-real.ckpt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(CKPT_PATH):
        print(f"Đang load mô hình từ: {CKPT_PATH}")
        # Trong thực tế, bạn cần lưu ý về cấu trúc state_dict khi load checkpoint Pytorch Lightning.
        # model = DBWUNetLightning.load_from_checkpoint(CKPT_PATH, map_location=device)
        
        model = LightweightDualBranchUNet(base_dim=18, num_heads=2)
        try:
            checkpoint = torch.load(CKPT_PATH, map_location=device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            # Xóa prefix thừa từ checkpoint (torch.compile tạo ra _orig_mod.)
            new_state_dict = {}
            for k, v in state_dict.items():
                # Bóc hết các prefix có thể có
                for prefix in ["model._orig_mod.", "model.model.", "model.", "net.", "_orig_mod."]:
                    if k.startswith(prefix):
                        k = k[len(prefix):]
                        break
                new_state_dict[k] = v
                
            # Load thẳng vào mạng lõi (LightweightDualBranchUNet)
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f"✅ Đã load weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            if missing:
                print(f"Missing keys (sample): {missing[:5]}")
        except Exception as e:
            print(f"❌ LỖI KHI LOAD WEIGHTS (Đang chạy với random weights): {e}")
            
        model.to(device)
        model.eval()
        print("Load mô hình thành công.")
    else:
        print(f"Cảnh báo: Không tìm thấy checkpoint tại {CKPT_PATH}. Vui lòng kiểm tra lại.")

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    if not model:
        return {"error": "Mô hình chưa được tải."}
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        orig_w, orig_h = image.size

        # Giới hạn kích thước tối đa để tránh hết RAM
        MAX_SIDE = 1024
        if max(orig_w, orig_h) > MAX_SIDE:
            scale = MAX_SIDE / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            print(f"⚠️  Ảnh lớn ({orig_w}x{orig_h}), resize về {new_w}x{new_h} để tránh hết RAM.")
        
        # Đảm bảo kích thước chia hết cho 16 (yêu cầu của DWT trong model)
        img_tensor = TF.to_tensor(image).unsqueeze(0).to(device)
        _, _, H, W = img_tensor.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Suy luận
        with torch.no_grad():
            output_tensor = model(img_tensor)
        
        # Giải phóng tensor input ngay sau khi inference xong
        del img_tensor
        
        # Trả về kích thước gốc
        if pad_h > 0 or pad_w > 0:
            output_tensor = output_tensor[:, :, :H, :W]
            
        # Hậu xử lý: chuyển tensor về PIL Image
        output_image = TF.to_pil_image(output_tensor.squeeze(0).cpu().clamp(0, 1))
        del output_tensor
        
        # Chuyển đổi sang byte để trả về
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        result_bytes = img_byte_arr.getvalue()
        
        return Response(content=result_bytes, media_type="image/png")
        
    except Exception as e:
        import traceback
        print(f"❌ API Error: {traceback.format_exc()}")
        return Response(content=f"Server Error: {str(e)}", status_code=500)
    
    finally:
        # Tự động dọn RAM/VRAM sau mỗi request (dù thành công hay lỗi)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 Đã dọn RAM sau request.")

@app.get("/")
def root():
    return {"message": "DB-WUNet API Demo đang chạy. Truy cập /docs để thử nghiệm."}
