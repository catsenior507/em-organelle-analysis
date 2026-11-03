#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⛔ 已废弃 —— 本文件已被 `tools/detr/` 取代，请勿再运行。

为什么废弃（三条都是"跑不通"级别，不只是"效果差"）
--------------------------------------------------
1. **第一次前向就崩**：`collate_fn`（本文件 L466-480）返回的 `targets` 是
   **list**，而 `compute_loss`（L299-301）按 **dict** 索引
   （`targets['boxes'][i]`）→ `TypeError`。本脚本从未成功训练过。
2. **图像被双重归一化**：`Dataset.__getitem__`（L80-85）先做
   `Resize+ToTensor+Normalize(ImageNet)`，再把 tensor 交给
   `DetrImageProcessor`（L84）二次归一化；处理器还会把短边缩到 800 px，
   与声明的 `img_size=256` 完全脱节。
3. **监督信号本身是坏的**：数据来自 `data/change_feature_json`，其中
   7404 个区域的 `region_pixels` 已被压成 **1 个点**（见
   `docs/AUDIT-REPORT.md` §2.2 CRIT-12），特征则来自被截断到 1000 点的像素表。

另外与 `进度3.pdf` 描述的方案不符：文档写"匈牙利算法 + GIoU + 加权交叉熵"，
这里实现的是 `torch.topk` 贪心匹配（L318）+ MSE（L322-333），没有 GIoU。

替代实现
--------
    tools/detr/dataset.py    数据管线（单次预处理、按图分组划分）
    tools/detr/losses.py     匈牙利匹配 + 加权 CE + L1 + GIoU + 属性 MSE
    tools/detr/model.py      DetrRegionModel（复用预训练 DetrModel）
    tools/detr/train.py      训练/评估
    tools/detr/selftest.py   12 项 CPU 自检（无需预训练权重）
    tools/detr/README.md     服务器运行手册

原文件内容保存在 git 提交 `aeef301` 中：`git show aeef301:code/detr_region_feature.py`

--------------------------------------------------------------------------------
以下为原始文件内容（保留仅供参考，不再维护）。
--------------------------------------------------------------------------------

DETR-based Region Detection and Feature Regression Script
功能：使用 DETR（Detection Transformer）实现区域检测+特征回归
输入：增强后的图片和对应的JSON标注
输出：检测到的regions及其14维特征
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers.modeling_outputs import BaseModelOutput

# --------------------------
# 配置参数
# --------------------------
CONFIG = {
    'data_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
    'img_size': 256,
    'batch_size': 4,
    'num_epochs': 10,
    'lr': 1e-4,
    'num_queries': 10,  # 最多检测10个regions
    'feature_dim': 14,   # 14维特征
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# --------------------------
# 数据集类
# --------------------------
class RegionFeatureDataset(Dataset):
    """
    数据集类：加载图片和对应的region标注
    """
    def __init__(self, img_dir, json_dir, processor, transform=None):
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.processor = processor
        self.transform = transform
        
        # 获取所有图片文件
        self.img_files = []
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                self.img_files.append(f)
        
        print(f"找到 {len(self.img_files)} 张图片")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        
        # 获取对应的JSON文件名
        img_name = os.path.splitext(img_file)[0]
        json_file = f'region_analysis__{img_name}.json'
        json_path = os.path.join(self.json_dir, json_file)
        
        # 读取图片
        image = Image.open(img_path).convert("RGB")
        
        # 读取JSON标注
        targets = self._load_json_annotation(json_path, image.size)
        
        # 预处理
        if self.transform:
            image = self.transform(image)
        
        # 使用DETR处理器
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        
        return {
            'pixel_values': pixel_values,
            'targets': targets,
            'image_name': img_name
        }

    def _load_json_annotation(self, json_path, image_size):
        """加载JSON标注并转换为DETR格式"""
        if not os.path.exists(json_path):
            print(f"警告：未找到JSON文件 {json_path}")
            return self._create_empty_target()
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        regions = data.get('regions', [])
        if not regions:
            return self._create_empty_target()
        
        # 解析regions
        boxes = []
        features = []
        labels = []
        
        for region in regions:
            region_id = region.get('region_id', 0)
            region_pixels = region.get('region_pixels', [])
            
            if not region_pixels or len(region_pixels) < 2:
                continue
            
            # 将像素列表转换为坐标数组
            coords = np.array(region_pixels).reshape(-1, 2)
            
            if len(coords) < 1:
                continue
            
            # 计算边界框 (xmin, ymin, xmax, ymax)
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            xmin = float(np.min(x_coords))
            ymin = float(np.min(y_coords))
            xmax = float(np.max(x_coords))
            ymax = float(np.max(y_coords))
            
            # 归一化到 [0, 1]
            img_width, img_height = image_size
            xmin_norm = xmin / img_width
            ymin_norm = ymin / img_height
            xmax_norm = xmax / img_width
            ymax_norm = ymax / img_height
            
            boxes.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])
            
            # 计算14维特征
            feat = self._compute_features(region, coords, img_width, img_height)
            features.append(feat)
            
            # 类别标签（统一为1，表示region）
            labels.append(1)
        
        # 转换为tensor
        targets = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty(0, 4),
            'features': torch.tensor(features, dtype=torch.float32) if features else torch.empty(0, CONFIG['feature_dim']),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.empty(0, dtype=torch.long),
            'image_id': torch.tensor([idx]),
            'area': torch.tensor([(box[2]-box[0])*(box[3]-box[1]) for box in boxes], dtype=torch.float32) if boxes else torch.empty(0),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64) if boxes else torch.empty(0)
        }
        
        return targets

    def _compute_features(self, region, coords, img_width, img_height):
        """计算14维区域特征"""
        features = []
        
        # 1-6: 位置特征 (min_x, max_x, min_y, max_y, width, height)
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        min_x = float(np.min(x_coords)) / img_width
        max_x = float(np.max(x_coords)) / img_width
        min_y = float(np.min(y_coords)) / img_height
        max_y = float(np.max(y_coords)) / img_height
        width = max_x - min_x
        height = max_y - min_y
        
        features.extend([min_x, max_x, min_y, max_y, width, height])
        
        # 7-8: 中心特征 (center_x, center_y)
        center_x = float(np.mean(x_coords)) / img_width
        center_y = float(np.mean(y_coords)) / img_height
        features.extend([center_x, center_y])
        
        # 9-10: 轴特征 (major_axis, minor_axis)
        if len(coords) >= 2:
            coords_centered = coords - np.mean(coords, axis=0)
            cov_matrix = np.cov(coords_centered.T)
            eigenvalues, _ = np.linalg.eig(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]
            major_axis = 2 * np.sqrt(eigenvalues[0]) / max(img_width, img_height)
            minor_axis = 2 * np.sqrt(eigenvalues[1]) / max(img_width, img_height) if len(eigenvalues) > 1 else 0
        else:
            major_axis = 0
            minor_axis = 0
        
        features.extend([major_axis, minor_axis])
        
        # 11: 粗糙度 (roughness) - 简化计算
        roughness = len(coords) / ((max_x - min_x) * (max_y - min_y) * img_width * img_height) if width * height > 0 else 0
        features.append(roughness)
        
        # 12: 复杂度 (complexity) - 使用像素分布
        complexity = np.std(coords) / max(img_width, img_height) if len(coords) > 1 else 0
        features.append(complexity)
        
        # 13-14: 灰度特征 (mean, std) - 使用黄色像素比例代替
        yellow_pixels = region.get('yellow_pixels', 0)
        total_pixels = region.get('total_pixels', len(coords))
        mean_gray = float(yellow_pixels) / total_pixels if total_pixels > 0 else 0
        std_gray = 0.1  # 默认值
        
        features.extend([mean_gray, std_gray])
        
        return features

    def _create_empty_target(self):
        """创建空目标"""
        return {
            'boxes': torch.empty(0, 4),
            'features': torch.empty(0, CONFIG['feature_dim']),
            'labels': torch.empty(0, dtype=torch.long),
            'image_id': torch.tensor([0]),
            'area': torch.empty(0),
            'iscrowd': torch.empty(0)
        }

# --------------------------
# 自定义DETR模型
# --------------------------
class DetrForRegionFeature(nn.Module):
    """
    自定义DETR模型，添加特征回归头
    输出：检测框 + 14维特征
    """
    def __init__(self, num_queries=10, feature_dim=14):
        super().__init__()
        
        # 加载预训练DETR模型
        self.detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        
        # 修改分类头（只有一个类别：region）
        self.detr.class_labels_classifier = nn.Linear(
            self.detr.class_labels_classifier.in_features, 2  # 0=背景, 1=region
        )
        
        # 添加特征回归头
        self.feature_head = nn.Sequential(
            nn.Linear(self.detr.config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
        
        # 修改查询数量
        self.detr.model.decoder.num_queries = num_queries
        self.detr.model.decoder.query_embed = nn.Embedding(num_queries, self.detr.config.hidden_dim)
        
        self.num_queries = num_queries
        self.feature_dim = feature_dim

    def forward(self, pixel_values, pixel_mask=None, targets=None):
        """
        前向传播
        """
        # 获取DETR的基础输出
        outputs = self.detr.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask
        )
        
        # 提取解码器输出
        decoder_output = outputs.last_hidden_state  # [batch_size, num_queries, hidden_dim]
        
        # 预测分类和边界框
        logits = self.detr.class_labels_classifier(decoder_output)  # [batch_size, num_queries, num_classes]
        pred_boxes = self.detr.bbox_predictor(decoder_output).sigmoid()  # [batch_size, num_queries, 4]
        
        # 预测14维特征
        pred_features = self.feature_head(decoder_output)  # [batch_size, num_queries, feature_dim]
        
        # 如果有目标，计算损失
        loss = None
        if targets is not None:
            loss = self.compute_loss(logits, pred_boxes, pred_features, targets)
        
        return {
            'logits': logits,
            'pred_boxes': pred_boxes,
            'pred_features': pred_features,
            'loss': loss
        }

    def compute_loss(self, logits, pred_boxes, pred_features, targets):
        """
        计算综合损失
        """
        device = logits.device
        batch_size = logits.size(0)
        
        total_loss = 0.0
        
        for i in range(batch_size):
            target_boxes = targets['boxes'][i].to(device)
            target_features = targets['features'][i].to(device)
            target_labels = targets['labels'][i].to(device)
            
            num_targets = len(target_labels)
            
            if num_targets == 0:
                # 没有目标，只计算背景分类损失
                bg_logits = logits[i, :, 0]  # 背景类概率
                loss_cls = F.binary_cross_entropy_with_logits(bg_logits, torch.ones_like(bg_logits))
                total_loss += loss_cls
                continue
            
            # 匹配预测和目标（简化的匈牙利匹配）
            # 使用IOU匹配
            pred_box = pred_boxes[i]
            iou_matrix = self._compute_iou_matrix(pred_box, target_boxes)
            
            # 选择IOU最高的前num_targets个预测
            _, indices = torch.topk(iou_matrix.max(dim=1).values, min(num_targets, self.num_queries))
            
            # 分类损失
            selected_logits = logits[i, indices]
            loss_cls = F.cross_entropy(selected_logits, target_labels[:len(indices)])
            
            # 边界框损失
            selected_boxes = pred_box[indices]
            loss_bbox = F.mse_loss(selected_boxes, target_boxes[:len(indices)])
            
            # 特征回归损失
            selected_features = pred_features[i, indices]
            loss_features = F.mse_loss(selected_features, target_features[:len(indices)])
            
            # 综合损失
            loss = loss_cls + loss_bbox + 0.1 * loss_features
            total_loss += loss
        
        return total_loss / batch_size

    def _compute_iou_matrix(self, pred_boxes, target_boxes):
        """计算IOU矩阵"""
        # pred_boxes: [num_queries, 4]
        # target_boxes: [num_targets, 4]
        # 返回: [num_queries, num_targets]
        
        px1, py1, px2, py2 = pred_boxes.unbind(1)
        tx1, ty1, tx2, ty2 = target_boxes.unbind(1)
        
        # 计算交集
        inter_x1 = torch.max(px1.unsqueeze(1), tx1.unsqueeze(0))
        inter_y1 = torch.max(py1.unsqueeze(1), ty1.unsqueeze(0))
        inter_x2 = torch.min(px2.unsqueeze(1), tx2.unsqueeze(0))
        inter_y2 = torch.min(py2.unsqueeze(1), ty2.unsqueeze(0))
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集
        pred_area = (px2 - px1) * (py2 - py1)
        target_area = (tx2 - tx1) * (ty2 - ty1)
        union_area = pred_area.unsqueeze(1) + target_area.unsqueeze(0) - inter_area
        
        # IOU
        iou = inter_area / (union_area + 1e-6)
        
        return iou

# --------------------------
# 训练函数
# --------------------------
def train_model():
    """训练DETR区域特征模型"""
    print("="*50)
    print("DETR区域检测与特征回归训练")
    print("="*50)
    
    # 初始化处理器
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    img_dir = os.path.join(CONFIG['data_path'], 'change')
    json_dir = os.path.join(CONFIG['data_path'], 'change_feature_json')
    
    dataset = RegionFeatureDataset(img_dir, json_dir, processor, transform)
    
    if len(dataset) == 0:
        print("错误：未找到训练数据")
        return
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], collate_fn=collate_fn)
    
    # 创建模型
    model = DetrForRegionFeature(
        num_queries=CONFIG['num_queries'],
        feature_dim=CONFIG['feature_dim']
    ).to(CONFIG['device'])
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    # 训练循环
    best_val_loss = float('inf')
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'change_mode', 'detr_region_feature.pth')
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
        
        # 训练
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            pixel_values = batch['pixel_values'].to(CONFIG['device'])
            targets = {k: v.to(CONFIG['device']) if isinstance(v, torch.Tensor) else v for k, v in batch['targets'].items()}
            
            outputs = model(pixel_values, targets=targets)
            loss = outputs['loss']
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"  训练损失: {avg_train_loss:.4f}")
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(CONFIG['device'])
                targets = {k: v.to(CONFIG['device']) if isinstance(v, torch.Tensor) else v for k, v in batch['targets'].items()}
                
                outputs = model(pixel_values, targets=targets)
                loss = outputs['loss']
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"  验证损失: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  保存最佳模型到 {model_save_path}")
    
    print("\n训练完成！")

def collate_fn(batch):
    """自定义collate函数"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    image_names = [item['image_name'] for item in batch]
    
    # 合并targets
    targets = []
    for item in batch:
        targets.append(item['targets'])
    
    return {
        'pixel_values': pixel_values,
        'targets': targets,
        'image_names': image_names
    }

# --------------------------
# 预测函数
# --------------------------
def predict(model_path, image_path, output_path):
    """使用训练好的模型进行预测"""
    print(f"\n预测图片: {image_path}")
    
    # 加载处理器
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # 加载模型
    model = DetrForRegionFeature(
        num_queries=CONFIG['num_queries'],
        feature_dim=CONFIG['feature_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
    model.to(CONFIG['device'])
    model.eval()
    
    # 加载图片
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pixel_values = transform(image).unsqueeze(0).to(CONFIG['device'])
    
    # 预测
    with torch.no_grad():
        outputs = model(pixel_values)
    
    # 解析结果
    logits = outputs['logits']
    pred_boxes = outputs['pred_boxes']
    pred_features = outputs['pred_features']
    
    # 获取置信度大于阈值的预测
    confidence_threshold = 0.5
    results = []
    
    for i in range(CONFIG['num_queries']):
        prob = F.softmax(logits[0, i], dim=-1)[1].item()  # region类别的概率
        if prob > confidence_threshold:
            # 反归一化边界框
            box = pred_boxes[0, i].cpu().numpy()
            box[0] *= original_size[0]
            box[1] *= original_size[1]
            box[2] *= original_size[0]
            box[3] *= original_size[1]
            
            # 获取特征
            features = pred_features[0, i].cpu().numpy()
            
            results.append({
                'region_id': i + 1,
                'confidence': prob,
                'bbox': {
                    'xmin': float(box[0]),
                    'ymin': float(box[1]),
                    'xmax': float(box[2]),
                    'ymax': float(box[3])
                },
                'features': {
                    'position': {
                        'min_x': float(features[0] * original_size[0]),
                        'max_x': float(features[1] * original_size[0]),
                        'min_y': float(features[2] * original_size[1]),
                        'max_y': float(features[3] * original_size[1]),
                        'width': float(features[4] * original_size[0]),
                        'height': float(features[5] * original_size[1])
                    },
                    'center': {
                        'x': float(features[6] * original_size[0]),
                        'y': float(features[7] * original_size[1])
                    },
                    'axes': {
                        'major_axis': float(features[8] * max(original_size)),
                        'minor_axis': float(features[9] * max(original_size))
                    },
                    'roughness': float(features[10]),
                    'complexity': float(features[11]),
                    'gray_level': {
                        'mean': float(features[12]),
                        'std': float(features[13])
                    }
                }
            })
    
    # 保存结果
    os.makedirs(output_path, exist_ok=True)
    result_file = os.path.join(output_path, f"prediction_{os.path.basename(image_path)}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"预测完成，结果保存到: {result_file}")
    print(f"检测到 {len(results)} 个regions")
    
    return results

# --------------------------
# 主函数
# --------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DETR区域检测与特征回归')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='运行模式')
    parser.add_argument('--image-path', help='预测时的图片路径')
    parser.add_argument('--output-path', default='./predictions', help='预测结果保存路径')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model()
    elif args.mode == 'predict':
        if not args.image_path:
            print("错误：预测模式需要提供 --image-path 参数")
            sys.exit(1)
        
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'change_mode', 'detr_region_feature.pth')
        if not os.path.exists(model_path):
            print(f"错误：未找到模型文件 {model_path}")
            sys.exit(1)
        
        predict(model_path, args.image_path, args.output_path)
