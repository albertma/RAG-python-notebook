# RAG-python-notebook

A RAG notebook projects, Which is used in China

## 设置环境变量

使用SILLICONFLOW

```.zshrc
export  SILLICONFLOW_API_KEY=sk-xxxxxxxxx
```

## 运行notebook

```Shell
$pip install -r requirements.txt
$jupyter notebook
```

## RAG Fusion

使用

1. 余弦向量
2. BM25 索引
3. 融合前两者，权重各50%

获取与问题相关度前5个最高的，进行上下文构建，然后生成对应的答案。结果看***fusion_rag_result.md***

```shell
$python funsion_rag.py 
```
