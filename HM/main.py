from elasticsearch_class import elasticsearch
from embedding_class import embedding
from retrieval_class import retrieval

def main():
    # ElasticSearch 인스턴스 생성
    es_instance = elasticsearch()
    es_instance.create_client()

    # embedding 인스턴스 생성
    embed_instance = embedding()

    # retrieval 인스턴스 생성
    retrieval_instance = retrieval(es_instance, embed_instance)

    # 인덱스 생성 예시
    index_name = "test"
    settings = {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
    mappings = {
        "properties": {
            "content": {"type": "text"},
            "embeddings": {"type": "dense_vector", "dims": 768}
        }
    }
    retrieval_instance.create_es_index(index_name, settings, mappings)

    # 문서 추가 예시
    docs = [
        {"id": 1, "content": "이것은 첫 번째 문서입니다."},
        {"id": 2, "content": "두 번째 문서의 내용입니다."},
        {"id": 3, "content": "세 번째 문서는 조금 더 길 수 있습니다."},
    ]
    embeddings = embed_instance.get_embeddings_in_batches(docs)
    for doc, emb in zip(docs, embeddings):
        doc["embeddings"] = emb.tolist()
    
    retrieval_instance.bulk_add(index_name, docs)

    # 검색 예시
    query = "문서"
    sparse_results = retrieval_instance.sparse_retrieve(query, 2)
    dense_results = retrieval_instance.dense_retrieve(query, 2)
    dense_results_2 = retrieval_instance.dense_retrieve_2(query, sparse_results['hits']['hits'], 2)

    print("Sparse 검색 결과:", sparse_results)
    print("Dense 검색 결과:", dense_results)
    print("Dense 검색 결과 2:", dense_results_2)

if __name__ == "__main__":
    main()