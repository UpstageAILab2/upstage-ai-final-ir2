import json
from elasticsearch_class import elasticsearch
from embedding_class import embedding
from retrieval_class import retrieval
from config import Config
from rag_class import RAG

def eval_rag(rag_instance, eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = rag_instance.answer_question([{"role": "user", "content": j["msg"]}])
            print(f'Answer: {response["answer"]}\n')

            output = {
                "eval_id": j["eval_id"], 
                "standalone_query": response["standalone_query"], 
                "topk": response["topk"], 
                "answer": response["answer"], 
                "references": response["references"]
            }
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

def main():
    # ElasticSearch 인스턴스 생성
    es_instance = elasticsearch()
    es_instance.create_client()

    # embedding 인스턴스 생성
    embed_instance = embedding()

    # retrieval 인스턴스 생성
    retrieval_instance = retrieval(es_instance, embed_instance)

    # Config 클래스에서 설정과 매핑 가져오기
    settings = Config.get_settings()
    mappings = Config.get_mappings()

    # 'test' 인덱스 생성
    retrieval_instance.create_es_index("test", settings, mappings)

    # 문서의 content 필드에 대한 임베딩 생성 
    index_docs = []
    with open("/data/ephemeral/home/upstage-ai-final-ir2/HM/data/documents.jsonl") as f:
        docs = [json.loads(line) for line in f]
    embeddings = retrieval_instance.get_embeddings_in_batches(docs)

    # 생성한 임베딩을 색인할 필드로 추가
    for doc, embedding in zip(docs, embeddings):
        doc["embeddings"] = embedding.tolist()
        index_docs.append(doc)

    # 'test' 인덱스에 대량 문서 추가
    ret = retrieval_instance.bulk_add("test", index_docs)

    # 색인이 잘 되었는지 확인 (색인된 총 문서수가 출력되어야 함)
    print(f"Indexed documents: {ret}")

    # RAG 인스턴스 생성
    rag_instance = RAG(api_key="sk-proj-FCdLR9oK", retrieval_instance=retrieval_instance)

    # 평가 실행
    eval_rag(rag_instance, "/data/ephemeral/home/upstage-ai-final-ir2/HM/data/eval.jsonl", "sample_submission_3.csv")

    # 검색 예시 (선택적)
    query = "문서"
    sparse_results = retrieval_instance.sparse_retrieve(query, 2)
    dense_results = retrieval_instance.dense_retrieve(query, 2)
    dense_results_2 = retrieval_instance.dense_retrieve_2(query, sparse_results['hits']['hits'], 2)

    print("Sparse 검색 결과:", sparse_results)
    print("Dense 검색 결과:", dense_results)
    print("Dense 검색 결과 2:", dense_results_2)

if __name__ == "__main__":
    main()