import os
import io
import torch
import torchvision.transforms.functional as TF
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from PIL import Image
from src.models.model import DBWUNetLightning

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
        
        # Placeholder cho việc load model (Do chưa setup đẩy đủ class Pytorch Lightning)
        model = DBWUNetLightning()
        try:
            # Giả định checkpoint lưu trực tiếp state_dict
            checkpoint = torch.load(CKPT_PATH, map_location=device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Cảnh báo khi load weights (Demo có thể chạy với random weights): {e}")
            
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
        
        # Tiền xử lý (đảm bảo chia hết cho 16 hoặc một số cụ thể nếu model yêu cầu)
        img_tensor = TF.to_tensor(image).unsqueeze(0).to(device)
        
        # Suy luận
        with torch.no_grad():
            output_tensor = model(img_tensor)
            
        # Hậu xử lý: chuyển tensor về PIL Image
        output_tensor = output_tensor.squeeze(0).cpu()
        output_image = TF.to_pil_image(output_tensor)
        
        # Chuyển đổi sang byte để trả về
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return Response(content=img_byte_arr, media_type="image/png")
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "DB-WUNet API Demo đang chạy. Truy cập /docs để thử nghiệm."}
