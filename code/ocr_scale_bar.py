import cv2
import pytesseract
import re

# 读取临时图像
img = cv2.imread('temp/temp_scale_mark.tif', cv2.IMREAD_GRAYSCALE)

if img is not None:
    # 二值化处理
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 使用 Tesseract OCR
    try:
        text = pytesseract.image_to_string(binary_img, config='--psm 7')
        text = text.strip()
        
        # 清理识别结果
        text = re.sub(r'\s+', '', text)
        
        print(text)
    except Exception as e:
        print(f"OCR 错误: {e}")
else:
    print("无法读取图像")