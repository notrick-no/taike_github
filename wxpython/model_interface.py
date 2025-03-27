from typing import List, Dict, Tuple, Union
import numpy as np
import random
from dataclasses import dataclass
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """检测结果数据类"""
    valid: bool
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float

class Model1:
    """零件定位模型模拟类
    
    用于检测视频帧中的零件位置，返回边界框坐标和置信度。
    当前为模拟实现，使用随机生成的数据。
    
    Attributes:
        min_confidence (float): 最小置信度阈值
    """
    
    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        logger.info("Model1初始化完成")
    
    def detect_and_crop(self, frame: np.ndarray) -> Tuple[List[DetectionResult], List[np.ndarray]]:
        """检测零件位置并返回裁剪区域
        
        Args:
            frame (np.ndarray): 输入视频帧，shape为(H, W, C)
            
        Returns:
            Tuple[List[DetectionResult], List[np.ndarray]]: 
                - 检测结果列表，每个元素包含valid、bbox和confidence
                - 裁剪后的零件图像列表
        """
        if frame is None or len(frame.shape) != 3:
            logger.error("输入帧格式错误")
            return [], []
            
        height, width = frame.shape[:2]
        results = []
        cropped_images = []
        
        # 模拟检测1-3个零件
        num_parts = random.randint(1, 3)
        
        for _ in range(num_parts):
            # 生成随机置信度
            confidence = random.uniform(0.3, 0.95)
            
            # 生成随机边界框坐标
            box_width = random.randint(100, 300)
            box_height = random.randint(100, 300)
            x1 = random.randint(0, width - box_width)
            y1 = random.randint(0, height - box_height)
            x2 = x1 + box_width
            y2 = y1 + box_height
            
            # 创建检测结果
            result = DetectionResult(
                valid=confidence >= self.min_confidence,
                bbox=[x1, y1, x2, y2],
                confidence=confidence
            )
            
            if result.valid:
                # 裁剪零件区域
                cropped = frame[y1:y2, x1:x2].copy()
                cropped_images.append(cropped)
            
            results.append(result)
            
        logger.info(f"检测到 {len(results)} 个零件，其中有效 {len(cropped_images)} 个")
        return results, cropped_images

class Model2:
    """缺陷分类模型模拟类
    
    用于对零件切片进行缺陷分类，返回缺陷类型和热力图。
    当前为模拟实现，使用随机生成的数据。
    
    Attributes:
        num_classes (int): 缺陷类别数量（包括正常类别）
        heatmap_size (Tuple[int, int]): 热力图尺寸
    """
    
    def __init__(self, num_classes: int = 6, heatmap_size: Tuple[int, int] = (6, 9)):
        self.num_classes = num_classes  # 0:正常, 1-5:缺陷类型
        self.heatmap_size = heatmap_size
        logger.info("Model2初始化完成")
    
    def classify_slices(self, slices: List[np.ndarray]) -> Dict[str, Union[List[int], np.ndarray]]:
        """对切片进行缺陷分类
        
        Args:
            slices (List[np.ndarray]): 54个切片图像列表
            
        Returns:
            Dict[str, Union[List[int], np.ndarray]]: 
                - defect_types: 54个切片的缺陷类型列表
                - heatmap: 6x9的热力图矩阵
        """
        if not slices or len(slices) != 54:
            logger.error("输入切片数量错误")
            return {'defect_types': [], 'heatmap': np.zeros(self.heatmap_size)}
        
        # 生成随机缺陷类型
        defect_types = [random.randint(0, self.num_classes-1) for _ in range(54)]
        
        # 生成随机热力图
        heatmap = np.random.rand(*self.heatmap_size)
        
        # 根据缺陷类型调整热力图值
        for i, defect_type in enumerate(defect_types):
            if defect_type > 0:  # 如果是缺陷
                row = i // 9
                col = i % 9
                heatmap[row, col] = random.uniform(0.7, 1.0)
        
        logger.info(f"分类完成，缺陷类型分布: {[defect_types.count(i) for i in range(self.num_classes)]}")
        return {
            'defect_types': defect_types,
            'heatmap': heatmap
        } 