# text_tokenizer.py (修复版)

import re
from typing import List, Union

import Stemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- NLTK 数据下载检查 (可选, 但建议) ---
try:
    stopwords.words('english')
except LookupError:
    print("正在下载 NLTK 'stopwords' 数据...")
    import nltk
    nltk.download('stopwords', quiet=True)
try:
    word_tokenize("test")
except LookupError:
    print("正在下载 NLTK 'punkt' 数据...")
    import nltk
    nltk.download('punkt', quiet=True)
# -----------------------------------------


class TextTokenizer:
    """
    一个自定义的文本分词器，用于文本清洗、分词、停用词移除和词干提取。
    """
    def __init__(self, language: str = "english"):
        """
        初始化分词器。
        
        Args:
            language (str): 用于词干提取和停用词的语言。
        """
        print(f"🔧 正在初始化自定义文本分词器 (语言: {language})...")
        self.stemmer = Stemmer.Stemmer(language)
        # 使用集合(set)以获得更快的查找速度
        self.stopwords = set(stopwords.words(language))
        print("✅ 自定义文本分词器初始化完成！")

    def _process_single_text(self, text: str) -> List[str]:
        """
        对单个字符串执行完整的处理流程。

        1. 转换为小写。
        2. 移除标点符号和数字。
        3. 分词。
        4. 移除停用词。
        5. 词干提取。
        """
        # 1. 转换为小写
        text = text.lower()
        
        # 2. 移除标点符号和数字 (只保留字母和空格)
        text = re.sub(r'[^a-z\s]', '', text)
        
        # 3. 分词
        tokens = word_tokenize(text)
        
        # 4. 移除停用词（在词干提取前）
        tokens = [token for token in tokens if token not in self.stopwords and len(token) > 1]
        
        # 5. 词干提取 - 使用 stemWords (批量处理)
        # ✅ 修复：使用 stemWords 而不是 stem
        if tokens:  # 只有当tokens非空时才调用
            processed_tokens = self.stemmer.stemWords(tokens)
        else:
            processed_tokens = []
        
        return processed_tokens

    def process(self, text_input: Union[str, List[str]]) -> List[List[str]]:
        """
        处理单个字符串或字符串列表，并返回 bm25s 库所需的格式。

        Args:
            text_input: 单个查询字符串或文档字符串列表。

        Returns:
            一个包含处理后 token 列表的列表 (List[List[str]])。
        """
        if isinstance(text_input, str):
            # 如果输入是单个字符串, 返回包含一个列表的列表
            return [self._process_single_text(text_input)]
        
        if isinstance(text_input, list):
            # 如果输入是列表, 对其中每个字符串进行处理
            return [self._process_single_text(doc) for doc in text_input]
        
        raise TypeError("输入必须是字符串 (str) 或字符串列表 (List[str])")