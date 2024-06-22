from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
import pandas as pd


from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain.schema import Document
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
UPSTAGE_API_KEY = os.environ.get('UPSTAGE_API_KEY')
LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = 'EXP03' # 프로젝트명 수정
LANGCHAIN_PROJECT = os.environ.get('LANGCHAIN_PROJECT')

print(f'LangSmith Project: {LANGCHAIN_PROJECT}')


# 데이터 구성
file_path = '../data/documents.jsonl'
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

loader = JSONLoader(
    file_path=file_path,
    jq_schema='.',
    text_content=False,
    json_lines=True,
)
temp = loader.load()

seq_num = 1
documents = []
for tmp in temp:
    data = json.loads(tmp.page_content)
    doc = Document(page_content=data['content'], metadata={
        'docid': data['docid'],
        'src': data['src'],
        'source': '/root/upstage-ai-final-ir2/HJM/data/documents.jsonl',
        'seq_num': seq_num,
    })
    documents.append(doc)
    seq_num += 1


# # content 길이 확인
# temp = []
# for idx in range(0, len(documents)):
#     value = len(documents[idx].page_content)
#     temp.append(value)
# data = temp
# temp = pd.DataFrame(data)
# temp.describe()
# # 문장 최소 길이 = 44, 평균 길이 = 315, 최대 길이 = 1230


# # Split 생략
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=20,
# )
# # pdf나 기사 텍스트처럼 긴 텍스트가 아니라고 생각돼서 생략함
# # Splitter를 수행해도 문서 당 2개 정도로 분리됨


# Embedding
embeddings = UpstageEmbeddings(
    api_key=UPSTAGE_API_KEY, 
    model="solar-embedding-1-large"
)


# 벡터 저장소 생성
# pip install faiss-cpu
folder_path = f'./faiss_{LANGCHAIN_PROJECT}'
if not os.path.exists(folder_path):
    print('Vector Store 생성 중')
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )
    vectorstore.save_local(folder_path=folder_path)
    print('Vector Store 생성 및 로컬 저장 완료')
else:
    vectorstore = FAISS.load_local(
        folder_path=folder_path, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    print('Vector Store 로컬에서 불러옴')

# # vectorstore에서 유사도 검색
# query = '금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?'
# db_similarity = vectorstore.similarity_search(
#     query=query,
#     k=5
# )
# i = 1
# for doc in db_similarity:
#     print(f'\n{i}.')
#     print(doc.page_content)
#     i += 1

# # vectorstore에서 점수에 기반한 유사도 검색
# query = '금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?'
# db_score = vectorstore.similarity_search_with_score(
#     query=query,
#     k=5
# )
# i = 1
# for doc in db_score:
#     content, score = doc
#     print(f'\n{i}.')
#     print(content.page_content)
#     print(score)
#     i += 1


# RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
prompt = hub.pull("rlm/rag-prompt")


# LLM과 검색엔진을 활용한 RAG 구현
retriever = vectorstore.as_retriever(k=4)
chat = ChatUpstage(model='solar-1-mini-chat', temperature=0)


def format_docs(docs):
    global references
    references = docs
    return '\n\n'.join(doc.page_content for doc in docs)

def answer_question(messages):
    global references
    response = {"topk": "", "answer": "", "references": ""}

    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | chat
        | StrOutputParser()
    )

    history = '\n'.join([f"{message['role']}: {message['content']}" for message in messages]) + '\n'
    answer = rag_chain.invoke(history)

    ref_content = [reference.page_content for reference in references]
    topk = [reference.metadata['docid'] for reference in references]

    response["topk"] = topk
    response["answer"] = answer
    response["references"] = ref_content

    return response


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as eval_lines, open(output_filename, 'w') as output_lines:
        idx = 0
        for eval_line in eval_lines:
            j = json.loads(eval_line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            output_lines.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

# 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
eval_rag('../data/eval.jsonl', '../submit/EXP03.csv')

# LangSmith 저장 시간 확보
time.sleep(60)