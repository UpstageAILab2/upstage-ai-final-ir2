class Config:
    @staticmethod
    def get_settings():
        return {
            "analysis": {
                "analyzer": {
                    "nori": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "decompound_mode": "mixed",
                        "filter": ["nori_posfilter"]
                    }
                },
                "filter": {
                    "nori_posfilter": {
                        "type": "nori_part_of_speech",
                        "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
                    }
                }
            }
        }

    @staticmethod
    def get_mappings(vector_dims=768):
        return {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "nori"
                },
                "embeddings": {
                    "type": "dense_vector",
                    "dims": vector_dims,
                    "index": True,
                    "similarity": "l2_norm"
                }
            }
        }

    @staticmethod
    def get_index_config(vector_dims=768):
        return {
            "settings": Config.get_settings(),
            "mappings": Config.get_mappings(vector_dims)
        }