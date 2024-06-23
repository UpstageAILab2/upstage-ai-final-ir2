import os
import json
from HM.elasticsearch_class import Elasticsearch, helpers
from subprocess import Popen, PIPE, STDOUT
import time

class elasticsearch:
    def __init__(self, username='elastic', password='8YoPc7sP_W-uBUDkXV73', 
                 ca_certs="/data/ephemeral/home/upstage-ai-final-ir2/HM/elasticsearch-8.8.0/config/certs/http_ca.crt"):
        self.es_username = username
        self.es_password = password
        self.ca_certs = ca_certs
        self.es = None
    
    def check_daemon(self):
        result = os.popen('ps -ef | grep elasticsearch').read()
        print("ElasticSearch 데몬 상태:")
        print(result)

    def create_client(self):
        self.es = Elasticsearch(['https://localhost:9200'], 
                                basic_auth=(self.es_username, self.es_password),
                                ca_certs=self.ca_certs, 
                                verify_certs=False)

    def get_info(self):
        if self.es:
            return self.es.info()
        else:
            print("ElasticSearch 클라이언트가 생성되지 않았습니다.")
            return None
    
    
    def create_index(self, index_name):
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name)
            print(f"인덱스 '{index_name}'가 생성되었습니다.")
        else:
            print(f"인덱스 '{index_name}'가 이미 존재합니다.")

    def index_document(self, index_name, document):
        result = self.es.index(index=index_name, body=document)
        return result