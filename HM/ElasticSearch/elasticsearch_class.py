import os
from elasticsearch import Elasticsearch

class elasticsearch:
    def __init__(self, username='elastic', password='i0U-yIVKQ-vAtoP9WbJ_', 
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