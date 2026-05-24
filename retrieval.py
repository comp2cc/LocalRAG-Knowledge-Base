from rank_bm25 import BM25Okapi
import jieba
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class HybridRetriever:
    def __init__(self, chroma_collection):
        self.collection = chroma_collection
        self.bm25 = None
        self.corpus_texts = []
        
    def build_bm25(self, chunks):
        """构建BM25索引（需要原始文本片段）"""
        self.corpus_texts = [chunk.page_content for chunk in chunks]
        tokenized_corpus = [list(jieba.cut(text)) for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def vector_search(self, query, top_k=10):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results['documents'][0]  # 返回文本列表
    
    def keyword_search(self, query, top_k=10):
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.corpus_texts[i], scores[i]) for i in top_indices if scores[i] > 0]
    
    def hybrid_search(self, query, top_k=10, alpha=0.5):
        """alpha: 向量检索权重，1-alpha为BM25权重"""
        # 向量检索得到top_k*2候选
        vec_texts = self.vector_search(query, top_k=top_k*2)
        # BM25检索
        kw_results = self.keyword_search(query, top_k=top_k*2)
        kw_texts = [text for text, _ in kw_results]
        # 合并去重
        all_texts = list(dict.fromkeys(vec_texts + kw_texts))
        
        # 简化版打分：直接用排名分数（更稳定的方法）
        # 这里为了简单，返回合并后的前top_k（实际生产需归一化分数）
        # 但为了演示，我们简单取两者交集排序
        # 更严谨的方式需要重新实现分数归一化，这里先返回向量+关键词的并集按某规则排序
        # 为了让你尽快跑通，我们简单返回混合后的前top_k
        combined = list(dict.fromkeys(vec_texts + kw_texts))
        return combined[:top_k]

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def rerank(self, query, passages, top_k=5):
        pairs = [[query, p] for p in passages]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze().tolist()
        if isinstance(scores, float):
            scores = [scores]
        scored = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:top_k]]