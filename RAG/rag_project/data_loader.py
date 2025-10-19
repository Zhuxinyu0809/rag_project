"""
数据加载模块 - data_loader.py

负责加载和管理HQ-small数据集
"""

from typing import List, Dict
from datasets import load_dataset
from config import DATASET_CONFIG


class DataLoader:
    """HQ-small数据集加载器"""
    
    def __init__(self):
        """初始化并加载数据集"""
        print("📚 正在加载数据集...")
        
        self.dataset = load_dataset(
            DATASET_CONFIG["name"],
            cache_dir=DATASET_CONFIG.get("cache_dir")
        )
        
        # 获取各个数据集分割
        self.train_data = self.dataset.get('train')
        self.val_data = self.dataset.get('validation')
        self.test_data = self.dataset.get('test')
        self.collection = self.dataset['collection']
        
        print(f"✅ 数据加载完成！")
        if self.train_data:
            print(f"   - 训练集: {len(self.train_data)} 条")
        if self.val_data:
            print(f"   - 验证集: {len(self.val_data)} 条")
        if self.test_data:
            print(f"   - 测试集: {len(self.test_data)} 条")
        print(f"   - 文档库: {len(self.collection)} 条")
    
    def get_documents(self) -> List[Dict]:
        """
        获取所有文档
        
        Returns:
            文档列表，每个文档包含id和text
        """
        return [
            {
                'id': item['id'],
                'text': item['text']
            }
            for item in self.collection
        ]
    
    def get_train_samples(self, n: int = None) -> List[Dict]:
        """
        获取训练集样本
        
        Args:
            n: 要获取的样本数量，None表示全部
        
        Returns:
            训练样本列表
        """
        if not self.train_data:
            return []
        
        samples = self.train_data if n is None else self.train_data.select(range(min(n, len(self.train_data))))
        
        return [
            {
                'id': item['id'],
                'question': item['text'],
                'answer': item['answer'],
                'supporting_ids': item['supporting_ids']
            }
            for item in samples
        ]
    
    def get_validation_samples(self, n: int = None) -> List[Dict]:
        """
        获取验证集样本
        
        Args:
            n: 要获取的样本数量，None表示全部
        
        Returns:
            验证样本列表
        """
        if not self.val_data:
            return []
        
        samples = self.val_data if n is None else self.val_data.select(range(min(n, len(self.val_data))))
        
        return [
            {
                'id': item['id'],
                'question': item['text'],
                'answer': item['answer'],
                'supporting_ids': item['supporting_ids']
            }
            for item in samples
        ]
    
    def get_test_samples(self, n: int = None) -> List[Dict]:
        """
        获取测试集样本
        
        Args:
            n: 要获取的样本数量，None表示全部
        
        Returns:
            测试样本列表
        """
        if not self.test_data:
            return []
        
        samples = self.test_data if n is None else self.test_data.select(range(min(n, len(self.test_data))))
        
        return [
            {
                'id': item['id'],
                'question': item['text']
            }
            for item in samples
        ]
    
    def get_document_by_id(self, doc_id: str) -> Dict:
        """
        根据ID获取单个文档
        
        Args:
            doc_id: 文档ID
        
        Returns:
            文档字典
        """
        for item in self.collection:
            if item['id'] == doc_id:
                return {
                    'id': item['id'],
                    'text': item['text']
                }
        return None