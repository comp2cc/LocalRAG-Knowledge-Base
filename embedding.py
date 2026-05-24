import chromadb
from chromadb.utils import embedding_functions

def get_chroma_collection(persist_dir="./chroma_db", collection_name="rag_docs"):
    """获取或创建Chroma集合，自动使用BGE embedding函数"""
    client = chromadb.PersistentClient(path=persist_dir)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5"
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def add_documents_to_chroma(collection, chunks, ids):
    """将分块后的文档添加到Chroma"""
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )