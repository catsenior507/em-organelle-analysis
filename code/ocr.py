#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR脚本，用于识别图像中的文字
"""

import sys
import cv2
import pytesseract
import json

# 配置Tesseract OCR路径（如果系统环境变量中没有配置）
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_image(image_path):
    """
    对图像进行OCR识别
    :param image_path: 图像文件路径
    :return: 识别到的文字
    """
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "text": "", "error": "无法读取图像"}
        
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 二值化处理
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 膨胀操作，增强文字连通性
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)
        
        # 进行OCR识别，限制只识别英文字母和数字
        try:
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
            text = pytesseract.image_to_string(binary, config=custom_config)
            
            # 去除空白字符
            text = text.strip()
            
            return {"success": True, "text": text, "error": ""}
        except pytesseract.pytesseract.TesseractNotFoundError:
            return {"success": False, "text": "", "error": "Tesseract OCR 未安装或路径配置错误"}
        except Exception as ocr_error:
            return {"success": False, "text": "", "error": "OCR识别失败: {0}".format(str(ocr_error))}
    except Exception as e:
        return {"success": False, "text": "", "error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"success": False, "text": "", "error": "参数错误，需要提供图像文件路径"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = ocr_image(image_path)
    print(json.dumps(result, ensure_ascii=False))
