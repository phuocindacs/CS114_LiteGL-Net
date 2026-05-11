import streamlit as st
import requests
from PIL import Image
import io

# ==============================================================================
# CẤU HÌNH TRANG
# ==============================================================================
st.set_page_config(
    page_title="LiteGL-Net | Low-Light Image Enhancer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CSS TÙY CHỈNH
# ==============================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
        color: #e0e0e0;
    }

    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .upload-box {
        border: 2px dashed rgba(167,139,250,0.4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: rgba(255,255,255,0.02);
        transition: all 0.3s ease;
    }

    .image-label {
        font-size: 0.85rem;
        color: #64748b;
        text-align: center;
        margin-top: 0.4rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .placeholder-box {
        height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px dashed rgba(167,139,250,0.3);
        border-radius: 12px;
        color: #475569;
        font-size: 1rem;
    }

    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
    }

    .status-ok {
        color: #34d399;
        font-weight: 600;
    }

    .status-err {
        color: #f87171;
        font-weight: 600;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: opacity 0.2s ease;
    }

    div.stButton > button:hover {
        opacity: 0.85;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Cài đặt")
    api_url = st.text_input(
        "FastAPI Server URL",
        value="http://127.0.0.1:8000",
        help="Địa chỉ server FastAPI đang chạy"
    )

    st.markdown("---")
    st.markdown("### 🔗 Kiểm tra kết nối")
    if st.button("Ping Server"):
        try:
            r = requests.get(f"{api_url}/", timeout=3)
            if r.status_code == 200:
                st.markdown('<p class="status-ok">✅ Kết nối thành công!</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="status-err">⚠️ Server phản hồi: {r.status_code}</p>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<p class="status-err">❌ Không kết nối được: {e}</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    **Hướng dẫn sử dụng:**
    1. Đảm bảo FastAPI server đang chạy
    2. Upload ảnh thiếu sáng
    3. Bấm **Enhance** và chờ kết quả
    """)
    st.markdown("---")
    st.caption("LiteGL-Net · Lightweight Global-Local Network")

# ==============================================================================
# NỘI DUNG CHÍNH
# ==============================================================================
st.markdown('<h1 class="main-title">✨ LiteGL-Net Image Enhancer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Nâng sáng ảnh thiếu sáng bằng mô hình LiteGL-Net</p>', unsafe_allow_html=True)

# Dùng session_state để lưu ảnh output giữa các lần rerender
if "output_bytes" not in st.session_state:
    st.session_state.output_bytes = None
if "output_filename" not in st.session_state:
    st.session_state.output_filename = None

uploaded_file = st.file_uploader(
    "Chọn ảnh thiếu sáng để nâng cấp",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)

# Khi đổi file thì xóa kết quả cũ
if uploaded_file is None:
    st.session_state.output_bytes = None
    st.session_state.output_filename = None

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    w, h = input_image.size

    st.markdown("---")

    # --- Nút Enhance ---
    btn_col, info_col = st.columns([1, 3])
    with btn_col:
        enhance_clicked = st.button("🚀 Enhance!", use_container_width=True)

    if enhance_clicked:
        with st.spinner("⏳ Đang xử lý ảnh..."):
            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
                response = requests.post(f"{api_url}/enhance", files=files, timeout=120)

                if response.status_code == 200:
                    st.session_state.output_bytes = response.content
                    st.session_state.output_filename = f"enhanced_{uploaded_file.name.rsplit('.', 1)[0]}.png"
                    st.success(f"✅ Hoàn tất! Kích thước ảnh gốc: {w}×{h}px")
                else:
                    err = response.json().get("error", "Lỗi không xác định")
                    st.error(f"❌ Server lỗi: {err}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Không kết nối được với FastAPI server. Hãy kiểm tra server đang chạy chưa.")
            except requests.exceptions.Timeout:
                st.error("⏱️ Server xử lý quá lâu (timeout 120s). Thử ảnh nhỏ hơn.")
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

    # --- Nút tải về (hiện khi đã có kết quả) ---
    if st.session_state.output_bytes is not None:
        with info_col:
            st.download_button(
                label="⬇️ Tải ảnh về",
                data=st.session_state.output_bytes,
                file_name=st.session_state.output_filename,
                mime="image/png",
                use_container_width=True
            )

    st.markdown("---")

    # --- Hiển thị ảnh hai cột ---
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(input_image, use_container_width=True)
        st.markdown('<p class="image-label">📥 Ảnh gốc (Input)</p>', unsafe_allow_html=True)

    with col2:
        if st.session_state.output_bytes is not None:
            output_image = Image.open(io.BytesIO(st.session_state.output_bytes)).convert("RGB")
            st.image(output_image, use_container_width=True)
            st.markdown('<p class="image-label">📤 Ảnh đã nâng sáng (Output)</p>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="placeholder-box">⬆ Bấm Enhance để xem kết quả</div>',
                unsafe_allow_html=True
            )

else:
    st.markdown("""
    <div class="upload-box">
        <div style="font-size:3rem;">🌃</div>
        <div style="color:#94a3b8; margin-top:0.5rem;">Kéo thả ảnh vào đây hoặc dùng nút <b>Browse files</b> phía trên</div>
        <div style="color:#475569; font-size:0.8rem; margin-top:0.4rem;">Hỗ trợ: JPG, PNG · Tối đa ~1024px cạnh dài sẽ được tự resize</div>
    </div>
    """, unsafe_allow_html=True)
