import os
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, faithfulness, answer_relevancy
from datasets import Dataset
from src.retrieval import HybridRetriever, Reranker
from src.embedding import get_chroma_collection
from src.generation import LLMGenerator
from src.ingestion import load_documents, chunk_documents
import tempfile
import uuid

# 准备测试数据集（手动标注几个问题和对应的标准答案以及预期上下文）
test_questions = [
    {
        "question": "什么是RAG？",
        "ground_truth": "RAG是检索增强生成，结合检索和生成的大模型技术。",
        # 预期上下文可以留空，评估时会自动用检索结果
    },
    {
        "question": "混合检索有什么好处？",
        "ground_truth": "混合检索同时使用向量和关键词，提升召回率。",
    }
]

def build_index_from_test_docs():
    """使用 sample.txt 或测试文档建立临时索引"""
    collection = get_chroma_collection(persist_dir="./chroma_db_eval", collection_name="eval")
    # 清空旧数据
    try:
        collection.delete(collection.get()['ids'])
    except:
        pass
    # 加载测试文档
    test_file = "data/sample.txt"
    if not os.path.exists(test_file):
        print("请准备测试文档 data/sample.txt")
        return None, None
    docs = load_documents([test_file])
    chunks = chunk_documents(docs)
    ids = [str(uuid.uuid4()) for _ in chunks]
    from src.embedding import add_documents_to_chroma
    add_documents_to_chroma(collection, chunks, ids)
    retriever = HybridRetriever(collection)
    retriever.build_bm25(chunks)
    return retriever, chunks

def run_evaluation():
    retriever, _ = build_index_from_test_docs()
    if retriever is None:
        return
    reranker = Reranker()
    llm = LLMGenerator()
    
    results = []
    for item in test_questions:
        q = item["question"]
        candidates = retriever.hybrid_search(q, top_k=5)
        reranked = reranker.rerank(q, candidates, top_k=3)
        generated = llm.generate(q, reranked)
        results.append({
            "question": q,
            "answer": generated,
            "contexts": reranked,
            "ground_truth": item["ground_truth"]
        })
    
    dataset = Dataset.from_list(results)
    scores = evaluate(dataset, metrics=[context_recall, context_precision, faithfulness, answer_relevancy])
    print(scores)
    # 保存结果
    scores.to_pandas().to_csv("ragas_scores.csv", index=False)
    print("评估结果已保存到 ragas_scores.csv")

if __name__ == "__main__":
    run_evaluation()