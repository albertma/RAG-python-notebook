import os

from typing import Any, Dict, List
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI
import re
import logging
import simple_vector_store
import fitz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="https://api.siliconflow.cn/v1/",
    api_key=os.getenv("SILLICONFLOW_API_KEY")
)

def get_pdf_text(pdf_path: str) -> str:
    """
    从 PDF 文件中提取文本
    """
    
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[dict]:
    """ 
    将文本按 chunk_size 分割成多个块，并重叠 chunk_overlap
    """
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i+chunk_size]
        if chunk:
            chunk_data = {
                "text": chunk,
                "metadata": {
                    "start_index": i,
                    "end_index": i+len(chunk)
                }
            }
            chunks.append(chunk_data)
    print(f"块数量：{len(chunks)}")
    return chunks

def clean_text(text: str) -> str:
    """
    清洗文本，去除标点符号和特殊字符
    """
    print("清洗文本，替换制表符和换行为空格，去除空格")
    # 将多个空白字符（包括换行符和制表符）替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 替换制表符和换行符为空格
    text = text.replace('\\t', ' ')
    text = text.replace('\\n', ' ')
    
    # 去除空格
    text = ' '.join(text.split())
    
    return text

def create_embeddings(chunks, model_name: str = "BAAI/bge-m3") -> List[np.ndarray]:
    """
    创建文本嵌入
    """
    print("创建文本嵌入")
    input_texts = chunks if isinstance(chunks, list) else [chunks]
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i+batch_size]
        response = client.embeddings.create(
            model=model_name,
            input=batch    
        )
        batch_embeddings = [np.array(item.embedding) for item in response.data]
        all_embeddings.extend(batch_embeddings)
    if isinstance(chunks, str):
        return all_embeddings[0]
    return all_embeddings

# 什么是 BM25 索引？
# BM25 是一种用于信息检索的算法，它通过计算查询与文档之间的相似度来排序搜索结果。
# 它将查询转换为词袋模型，并使用词频和逆文档频率来计算每个词的权重，然后加权求和得到文档的得分。
# 它是一种基于词袋模型的算法，适用于短文本的搜索。

def create_bm25_index(chunks: List[dict]) -> BM25Okapi:
    """
    创建 BM25 索引
    """
    print("创建 BM25 索引")
    
    texts = [chunk["text"] for chunk in chunks]
    # 将每个块文本转换为词袋模型
    tokenized_docs = [text.split() for text in texts]
    # 创建 BM25 索引
    bm25 = BM25Okapi(tokenized_docs)
    logger.info("创建 BM25 索引完成, 文件数量: %d", len(texts))
    return bm25


def search_bm25(bm25: BM25Okapi, chunks: List[dict], query: str, top_k=5) -> list:
    """
    使用 BM25 索引搜索
    """
    print("使用 BM25 索引搜索")
    # 将查询转换为词袋模型
    tokenized_query = query.split()
    # 获取 BM25 索引的得分
    scores = bm25.get_scores(tokenized_query)
    # 将结果添加到结果列表中
    results = []
    for i, score in enumerate(scores):
        # 获取块的元数据
        metadata = chunks[i]["metadata"].copy()
        # 添加块索引
        metadata["index"] = i
        results.append({
            "text": chunks[i]["text"],
            "metadata": metadata,
            "bm25_score": score
        })
    # 根据得分排序, 得分最高的块排在前面
    results.sort(key=lambda x: x["bm25_score"], reverse=True)
    return results[:top_k]

def fusion_retrieval(chunks: list, query: str, 
               vector_store: simple_vector_store.SimpleVectorStore, 
               bm25_index: BM25Okapi, top_k=5, alpha=0.5) -> List[Dict[str, Any]]:
    """
    融合向量索引和 BM25 索引
    """
    logger.info("融合向量索引和 BM25 索引")
    query_embedding = create_embeddings([query])[0]
    # 使用向量索引搜索
    vector_results = vector_store.similarity_search_with_scores(query_embedding, len(chunks))
    # 使用 BM25 索引搜索
    bm25_results = search_bm25(bm25_index, chunks, query, len(chunks))
    # 将结果转换为字典,字典的key为索引,value为相似度或得分
    vector_stores_dict = {result["metadata"]["index"]: result["similarity"] for result in vector_results}
    bm25_results_dict = {result["metadata"]["index"]: result["bm25_score"] for result in bm25_results}
    # 融合结果
    all_docs = vector_store.get_all_documents()
    combined_results = []
    for i,doc in enumerate(all_docs):
        # 计算融合得分
        vector_score = vector_stores_dict.get(i, 0.0)
        bm25_score = bm25_results_dict.get(i, 0.0)
        logger.info(f"vector_score: {vector_score}, bm25_score: {bm25_score}")
        # 添加到结果列表
        combined_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "vector_score": vector_score,
            "bm25_score": bm25_score,
            "index": i
        })
    # 计算得分
    vector_scores = np.array([result["vector_score"] for result in combined_results])
    bm25_scores = np.array([result["bm25_score"] for result in combined_results])
    
    # 归一化得分
    normalized_vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))
    normalized_bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
    
    # 融合得分
    combined_scores = alpha * normalized_vector_scores + (1 - alpha) * normalized_bm25_scores
    # 添加融合得分
    for i,result in enumerate(combined_results):
        result["combined_score"] = combined_scores[i]
    # 根据得分排序, 得分最高的块排在前面
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
    logger.info(f"融合得分后块数: {len(combined_results)}")
    # 返回得分最高的 top_k 个块
    top_k_results = combined_results[:top_k]
    logger.info(f"融合得分最高的 {top_k} 个块数, 总块数: {len(chunks)}")
    # 返回得分最高的 top_k 个块
    return top_k_results
    
    
def process_document(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> tuple:
    """
    处理文档
    pdf_path: 文档路径
    chunk_size: 块大小
    chunk_overlap: 块重叠
    return: 文本块, 向量索引, BM25 索引
    """
    logger.info(f"开始处理文档: {pdf_path}")
    # 获取文档文本
    if not os.path.exists(pdf_path):
        logger.error(f"文件不存在: {pdf_path}")
        return
    # 获取文档文本
    logger.info(f"获取文档文本")
    text = get_pdf_text(pdf_path)
    # 清洗文本
    logger.info(f"清洗文本")
    cleaned_text = clean_text(text)
    # 分割文本
    logger.info(f"分割文本")
    chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
    # 创建向量索引
    logger.info(f"创建向量索引")
    chunks_texts = [chunk["text"] for chunk in chunks]
    chunks_embeddings = create_embeddings(chunks_texts)
    vector_store = simple_vector_store.SimpleVectorStore()
    vector_store.add_items(chunks, chunks_embeddings)
    # 创建 BM25 索引
    logger.info(f"创建 BM25 索引")
    bm25_index = create_bm25_index(chunks)
    # 返回结果: 文本块, 向量索引, BM25 索引
    logger.info(f"返回结果: 文本块, 向量索引, BM25 索引")
    return chunks, vector_store, bm25_index

def generate_response(query:str, context:str) -> str:
    """
    生成响应
    query: 用户查询
    context: 上下文
    return: 响应
    """
    logger.info(f"生成响应")
    system_prompt = """ You are a helpful AI assistant that can answer questions based on the context.
    If the context does not contain relevant information to answer the question fully, 
    acknowledge this limitation.
    """
    user_prompt = f"""
    Context:
    {context}
    Question:
    {query}
    please answer the question based on the context.
    """
    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature = 0.1
    )
    return response.choices[0].message.content

def answer_with_fusion_rag(query:str, chunks:List[dict], vector_store:simple_vector_store.SimpleVectorStore, 
                           bm25_index:BM25Okapi, top_k:int = 5, alpha:float = 0.5) -> str:
    """
    使用融合向量索引和 BM25 索引回答问题
    """
    logger.info(f"使用融合向量索引和 BM25 索引回答问题")
    # 融合向量索引和 BM25 索引
    top_k_results = fusion_retrieval(chunks, query, vector_store, bm25_index, top_k=top_k, alpha=alpha)
    # 生成响应
    response = generate_response(query, top_k_results)
    return {
        "query": query,
        "response": response,
        "top_k_results": top_k_results
    }

def vector_only_retrieval(query:str, 
                          vector_store:simple_vector_store.SimpleVectorStore, 
                          top_k:int = 5) -> Dict[str, Any]:
    """
    只使用向量索引回答问题
    """
    logger.info(f"只使用向量索引回答问题")
    query_embedding = create_embeddings([query])[0]
    # 使用向量索引搜索
    vector_results = vector_store.similarity_search_with_scores(query_embedding, top_k)
    context = "\n\n----\n\n".join([result["text"] for result in vector_results])
    # 生成响应
    response = generate_response(query, context=context)
    return {
        "query": query,
        "response": response,
        "top_k_results": vector_results
    }

def bm25_only_retrieval(query:str, chunks:List[dict], bm25_index:BM25Okapi, top_k:int = 5) -> Dict[str, Any]:
    """
    只使用 BM25 索引回答问题
    """
    logger.info(f"只使用 BM25 索引回答问题")
    # 使用 BM25 索引搜索
    bm25_results = search_bm25(bm25_index, chunks, query, top_k)
    context = "\n\n----\n\n".join([result["text"] for result in bm25_results])
    # 生成响应
    response = generate_response(query, context=context)
    return {
        "query": query,
        "response": response,
        "top_k_results": bm25_results
    }
    
def evaluate_response(query:str, bm25_response:str, 
                      vector_response:str, fusion_response:str, 
                      reference_response:str = None) -> float:
        """
        评估响应
        """
        logger.info(f"评估响应")
        system_prompt = """
        You are an expert evaluator of RAG systems. Compare responses from three different retrieval approaches:
        1. Vector-based retrieval: Uses semantic similarity for document retrieval
        2. BM25 keyword retrieval: Uses keyword matching for document retrieval
        3. Fusion retrieval: Combines both vector and keyword approaches

        Evaluate the responses based on:
        - Relevance to the query
        - Factual correctness
        - Comprehensiveness
        - Clarity and coherence"""

        user_prompt = f"""
        Question:
        {query}
    
        Vector-based retrieval response:
        {vector_response}
        BM25 keyword retrieval response:
        {bm25_response}
        Fusion Retrieval response:
        {fusion_response}
        """
        if reference_response:
            user_prompt += f"""
            Reference Answer:
            {reference_response}
            """
        response = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature = 0.0
        )
        return response.choices[0].message.content
    
def compare_retrieval_methods(query:str, chunks:List[dict], 
                                  vector_store:simple_vector_store.SimpleVectorStore, 
                                  bm25_index:BM25Okapi, top_k:int = 5, 
                                  alpha:float = 0.5, reference_answer:str = None) -> Dict[str, Any]:
        """
        比较不同检索方法的性能
        query: 用户查询
        chunks: 文本块
        vector_store: 向量索引
        bm25_index: BM25 索引
        top_k: 返回的文本块数量
        alpha: 融合得分权重
        reference_answer: 参考答案
        """
        logger.info(f"\n========== 比较不同检索方法的性能 ==========")
        # 融合向量索引和 BM25 索引
        fusion_results = answer_with_fusion_rag(query, chunks, vector_store, bm25_index, top_k, alpha)
        # 只使用向量索引
        vector_results = vector_only_retrieval(query, vector_store, top_k)
        # 只使用 BM25 索引
        bm25_results = bm25_only_retrieval(query, chunks, bm25_index, top_k)
        
        # 评估响应
       
        compare_result_str = evaluate_response(query, 
                                           bm25_results["response"], 
                                           vector_results["response"], 
                                           fusion_results["response"], 
                                           reference_answer)
        return {
            "query": query,
            "compare_result": compare_result_str,
            "vector_results": vector_results,
            "bm25_results": bm25_results,
            "fusion_results": fusion_results
        }
def generate_overall_analysis(results:List[Dict[str, Any]]) -> str:
    """
    生成总体分析
    """
    logger.info(f"生成总体分析")
    system_prompt = """You are an expert at evaluating information retrieval systems. 
    Based on multiple test queries, provide an overall analysis comparing three retrieval approaches:
    1. Vector-based retrieval (semantic similarity)
    2. BM25 keyword retrieval (keyword matching)
    3. Fusion retrieval (combination of both)

    Focus on:
    1. Types of queries where each approach performs best
    2. Overall strengths and weaknesses of each approach
    3. How fusion retrieval balances the trade-offs
    4. Recommendations for when to use each approach"""
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"""
        Query {i+1}: {result['query']}
        Comparison:{result['compare_result'][:200]}...\n\n
        """
    user_prompt = f""" Based on the following evaluations of three retrieval approaches across {len(results)} queries,
    provide an overall analysis comparing the three approaches.
    {evaluations_summary}
    Please provide a comprehensive analysis of vector based retrieval, BM25 retrieval, and fusion retrieval,
    highlighting their strengths and weaknesses, and when each approach is most effective.
    """
    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature = 0.0
    )
    return response.choices[0].message.content

def evaluate_fusion_retrieval(pdf_path:str, test_queries:List[str], 
                              reference_answers:List[str] =None, 
                              top_k:int = 5, alpha:float = 0.5):
    """
    评估融合检索的性能
    pdf_path: 文档路径
    test_queries: 测试查询
    reference_answer: 参考答案
    top_k: 返回的文本块数量
    alpha: 融合得分权重
    """
    logger.info(f"评估融合检索的性能")
    # 处理文档
    chunks, vector_store, bm25_index = process_document(pdf_path)
    # 评估融合检索的性能
    results = []
    for i, query in enumerate(test_queries):
        logger.info(f"评估第 {i+1}/{len(test_queries)} 个查询")
        logger.info(f"查询: {query}")
        
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        compare_results = compare_retrieval_methods(query, 
                                                    chunks, 
                                                    vector_store, 
                                                    bm25_index, 
                                                    top_k, alpha, 
                                                    reference)
        
        results.append(compare_results)
        logger.info(f"==== 向量检索 ================================================")
        logger.info(f"向量检索生成响应: {compare_results['vector_results']['response']}")
        logger.info(f"向量检索索引上下文数量: {len(compare_results['vector_results']['top_k_results'])}")
        logger.info(f"==== BM25 检索 ================================================")
        logger.info(f"BM25 检索生成响应: {compare_results['bm25_results']['response']}")
        logger.info(f"BM25 检索索引上下文数量: {len(compare_results['bm25_results']['top_k_results'])}")
        
        logger.info(f"==== 融合检索 ================================================")
        logger.info(f"融合检索生成响应: {compare_results['fusion_results']['response']}")
        logger.info(f"融合检索索引上下文数量: {len(compare_results['fusion_results']['top_k_results'])}")
       
        logger.info(f"========== 评估结果 ==========")
        logger.info(f"评估结果: {compare_results['compare_result']}")
    logger.info(f"========== 生成总体分析 ==========")
    overall_analysis = generate_overall_analysis(results)
    logger.info(f"总体分析:\n \n {overall_analysis}")
    return {
        "overall_analysis": overall_analysis,
        "results": results
    }
    
def main():
    """
    主函数
    """
    logger.info(f"开始主函数")
    # 评估融合检索的性能
    evaluation_result = evaluate_fusion_retrieval(
        pdf_path="data/AI_Information.pdf",
        test_queries = [
            "What are the main applications of transformer models in natural language processing?"  # AI-specific query
        ],
        reference_answers = [
            "Transformer models have revolutionized natural language processing with applications including machine translation, text summarization, question answering, sentiment analysis, and text generation. They excel at capturing long-range dependencies in text and have become the foundation for models like BERT, GPT, and T5."
        ],
        top_k = 5,
        alpha = 0.5
    )
    logger.info(f"评估融合检索的性能完成")
if __name__ == "__main__":
    main()
