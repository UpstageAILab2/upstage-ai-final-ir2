from sentence_transformers import SentenceTransformer

class embedding:
    def __init__(self, model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, sentences):
        return self.model.encode(sentences)

    def get_embeddings_in_batches(self, docs, batch_size=100):
        batch_embeddings = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i: i+batch_size]
            contents = [doc["content"] for doc in batch]
            embeddings = self.get_embedding(contents)
            batch_embeddings.extend(embeddings)
            print(f'batch {i}')
        return batch_embeddings