from elasticsearch_class import elasticsearch
from embedding_class import embedding

def main():
    # ElasticSearch 인스턴스 생성
    es_instance = elasticsearch()

    # 데몬 상태 확인
    print("ElasticSearch 데몬 상태 확인:")
    es_instance.check_daemon()

    # 클라이언트 생성
    print("\nElasticSearch 클라이언트 생성:")
    es_instance.create_client()

    # 정보 확인
    print("\nElasticSearch 클라이언트 정보:")
    info = es_instance.get_info()
    print(info)

    # embedding 인스턴스 생성
    embed_instance = embedding()

    # 예시 문서 리스트 생성
    docs = [
        {"id": 1, "content": "이것은 첫 번째 문서입니다."},
        {"id": 2, "content": "두 번째 문서의 내용입니다."},
        {"id": 3, "content": "세 번째 문서는 조금 더 길 수 있습니다."},
        # ... 더 많은 문서 추가 ...
    ]

    # 배치 단위로 임베딩 생성
    embeddings = embed_instance.get_embeddings_in_batches(docs, batch_size=2)

    # ElasticSearch에 문서와 임베딩 인덱싱
    index_name = "documents"
    es_instance.create_index(index_name)
    
    for doc, emb in zip(docs, embeddings):
        doc["embedding"] = emb.tolist()  # numpy array를 list로 변환
        es_instance.index_document(index_name, doc)

    print(f"\n{len(docs)}개의 문서가 ElasticSearch에 인덱싱되었습니다.")

if __name__ == "__main__":
    main()