import qrcode

# ⭐️ 1. 여기에 님의 Streamlit 공용 URL을 붙여넣으세요.
streamlit_url = "https://cyberpunk-rag-agent-7snqyiqvjb4eokgcnvebh7.streamlit.app/" 

# 2. QR 코드 이미지 생성
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(streamlit_url)
qr.make(fit=True)

# 3. 이미지 파일로 저장
img = qr.make_image(fill_color="black", back_color="white")
img.save("cyberpunk_rag_qr_code.png")

print("✅ QR 코드가 'cyberpunk_rag_qr_code.png' 파일로 저장되었습니다.")