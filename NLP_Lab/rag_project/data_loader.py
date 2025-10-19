"""
数据加载模块 - 负责加载和管理数据集
"""

from typing import List, Dict
from datasets import load_dataset


class DataLoader:
    """数据加载器 - 从HuggingFace加载数据集"""
    
    def __init__(self, dataset_name: str = None, split: str = None):
        """
        初始化数据加载器
        
        Args:
            dataset_name: HuggingFace数据集名称
            split: 数据集分割名称
        """
        # 导入配置（延迟导入避免循环依赖）
        from config import DATASET_CONFIG
        
        if dataset_name is None:
            dataset_name = DATASET_CONFIG["name"]
        if split is None:
            split = DATASET_CONFIG["split"]
        
        print("📚 正在加载数据集...")
        print(f"   数据集: {dataset_name}")
        print(f"   分割: {split}")
        
        self.dataset = load_dataset(dataset_name)
        self.collection = self.dataset[split]
        
        print(f"✅ 数据加载完成！文档库: {len(self.collection)} 条")
    
    def get_documents(self) -> List[Dict]:
        """
        获取所有文档
        
        Returns:
            文档列表，每个文档包含 'id' 和 'text' 字段
        """
        return [
            {
                'id': item['id'], 
                'text': item['text']
            } 
            for item in self.collection
        ]
    
    def get_document_count(self) -> int:
        """获取文档总数"""
        return len(self.collection)
    
    def get_document_by_id(self, doc_id: str) -> Dict:
        """
        根据ID获取单个文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档字典，如果未找到返回None
        """
        for item in self.collection:
            if item['id'] == doc_id:
                return {'id': item['id'], 'text': item['text']}
        return None


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("测试 DataLoader 模块")
    print("="*70)
    
    # 创建数据加载器
    loader = DataLoader()
    
    # 获取文档
    documents = loader.get_documents()
    print(f"\n文档总数: {loader.get_document_count()}")
    print(f"\n前3个文档:")
    for doc in documents[:3]:
        print(f"\nID: {doc['id']}")
        print(f"文本: {doc['text'][:100]}...")