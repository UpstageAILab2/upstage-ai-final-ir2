from elasticsearch import helpers

class retrieval:
    def __init__(self, es_instance, embed_instance):
        self.es_instance = es_instance
        self.embed = embed_instance

    def create_es_index(self, index, settings, mappings):
        if self.es_instance.es.indices.exists(index=index):
            self.es_instance.es.indices.delete(index=index)
        self.es_instance.es.indices.create(index=index, settings=settings, mappings=mappings)
        print(f"인덱스 '{index}'가 생성되었습니다.")

    def delete_es_index(self, index):
        self.es_instance.es.indices.delete(index=index)
        print(f"인덱스 '{index}'가 삭제되었습니다.")

    def bulk_add(self, index, docs):
        actions = [{'_index': index, '_source': doc} for doc in docs]
        return helpers.bulk(self.es_instance.es, actions)

    def index_document(self, index_name, document):
        result = self.es_instance.es.index(index=index_name, body=document)
        return result

    def sparse_retrieve(self, query_str, size, index='test'):
        query = {"match": {"content": {"query": query_str}}}
        return self.es_instance.es.search(index=index, query=query, size=size, sort='_score')

    def dense_retrieve(self, query_str, size, index='test'):
        query_embedding = self.embed.get_embedding([query_str])[0]
        knn = {"field": "embeddings",
               "query_vector": query_embedding.tolist(),
               "k": size,
               "num_candidates": 100}
        return self.es_instance.es.search(index=index, knn=knn)

    def dense_retrieve_2(self, query_str, documents, size, index='test'):
        query_embedding = self.embed.get_embedding([query_str])[0]
        document_ids = [doc['_id'] for doc in documents]
        
        knn = {
            "field": "embeddings",
            "query_vector": query_embedding.tolist(),
            "k": size,
            "num_candidates": len(documents)
        }
        
        body = {
            "knn": knn,
            "_source": {"includes": ["content", "embeddings"]},
            "query": {"ids": {"values": document_ids}}
        }
        
        return self.es_instance.es.search(index=index, body=body)