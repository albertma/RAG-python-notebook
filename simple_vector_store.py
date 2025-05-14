import logging
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVectorStore:
    def __init__(self):
        logger.info("初始化 向量存储")
        self.vectors = [] # 向量
        self.texts = [] # 文本块
        self.metadata = [] # 元数据

    def add_item(self, text, embedding, metadata):
        """
        添加一个向量
        """
        logger.info(f"添加向量: {embedding}")
        self.vectors.append(embedding)
        self.texts.append(text)
        self.metadata.append(metadata)

    def add_items(self, items, embeddings):
        """
        添加多个向量
        """
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            self.add_item(
                text=item["text"], 
                embedding = embedding, 
                metadata= {**item.get("metadata",{}), "index": i}
                )

    def similarity_search_with_scores(self, query_vector, top_k=5) -> List[dict]:
        """
        相似度搜索
        query_vector: 查询向量
        top_k: 返回的文本块数量
        return: 包含文本块、元数据和相似度的列表
        """
        logger.info(f"通过余弦相似度搜索 {top_k} 个文本块")
        if not self.vectors:
            logger.warning("没有向量在存储中")
            return []
        # 计算余弦相似度
        #query_vector = np.array(query_vector)
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = cosine_similarity([vector], [query_vector])[0][0]
            similarities.append((i,similarity))
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回相似度最高的 top_k 个向量
        results = []
        for i in range(min(top_k, len(similarities))):
            index = similarities[i][0] # 文本块索引
            results.append({
                "text": self.texts[index],
                "metadata": self.metadata[index],
                "similarity": similarities[i][1]
            })
        # 返回相似度最高的 top_k 个向量
        return results
    
    def get_all_documents(self):
        """
        获取所有文本块
        """
        return [
            {
                "text": self.texts[i],
                "metadata": self.metadata[i]
            }
            for i in range(len(self.texts))
        ]
    
    