from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
import pandas as pd

from openai import OpenAI

from langchain import hub
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
UPSTAGE_API_KEY = os.environ.get('UPSTAGE_API_KEY')
LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = 'EXP06' # 프로젝트명 수정
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
    doc = Document(
        page_content=data['content'], 
        metadata={
            'docid': data['docid'],
            'src': data['src'],
            'source': '/root/upstage-ai-final-ir2/HJM/data/documents.jsonl',
            'seq_num': seq_num,
        }
    )
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


# Embedding
embeddings = UpstageEmbeddings(
    api_key=UPSTAGE_API_KEY,
    model='solar-embedding-1-large'
)


# 벡터 저장소 생성
# pip install faiss-cpu
folder_path = f'./vectorstore/{LANGCHAIN_PROJECT}'
if not os.path.exists(folder_path):
    print('Create Vector Store ...')

    # Split
    # splitter = SemanticChunker(
    #     embeddings=embeddings,
    #     buffer_size=1, # 최대 한 문장
    #     add_start_index=False, # 분할된 청크에 시작 인덱스 추가할지 결정. 시작 인덱스는 원래 텍스트 내에서 분할된 부분의 시작 위치를 나타냄 
    #     breakpoint_threshold_type='percentile', # 분할 지점을 결정하는 기준의 유형. 'percentile', 'standard_deviation', 'interquartile', 'gradient'
    #     sentence_split_regex=r"(?<=[.?!])\s+" # 문장을 분할하는 데 사용되는 정규 표현식
    # )
    # split_documents = splitter.split_documents(documents)
    
    splitter = CharacterTextSplitter(
        separator='. ',
        chunk_size=130,
        chunk_overlap=20,
        length_function=len,
    )
    split_documents = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(
        documents=split_documents,
        embedding=embeddings,
    )
    vectorstore.save_local(folder_path=folder_path)

    print(f'Text Splitter 적용 전 문서 개수: {len(documents)}\nText Splitter 적용 후 문서 개수: {len(split_documents)}')
    print(split_documents[0])

else:
    vectorstore = FAISS.load_local(
        folder_path=folder_path, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    print('Load Vector Store')


# vectorstore retriever
retriever = vectorstore.as_retriever(
    search_type='similarity', 
    search_kwargs={
        'k': 5,
    }
)


# RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
prompt = hub.pull("rlm/rag-prompt")


client = OpenAI(api_key=OPENAI_API_KEY)


chat = ChatUpstage(
    api_key=UPSTAGE_API_KEY,
    model='solar-1-mini-chat', 
    temperature=0,
    max_tokens=250,
)


# LLM과 검색엔진을 활용한 RAG 구현
def self_prompt(text):
    prompt_ = (
        "주어진 Context가 과학 상식과 관련된 질문인지 판별할 수 있도록 'yes' 또는 'no'로 답변해주세요."
        " 제공되는 예시는 일상 대화 중 나오는 질문입니다. 이러한 일상 대화가 아닌 질의는 'yes'로 답변해주세요.\n\n"
        "예시:\n"
        "요새 너무 힘들다. -> 'no'\n"
        "니가 대답을 잘해줘서 너무 신나! -> 'no'\n"
        "이제 그만 얘기해! -> 'no'\n"
        "오늘 너무 즐거웠다! -> 'no'\n"
        "우울한데 신나는 얘기 좀 해줘! -> 'no'\n"
        "너 모르는 것도 있니? -> 'no'\n"
        "너 잘하는게 뭐야? -> 'no'\n"
        "너 정말 똑똑하다! -> 'no'\n"
        "안녕 반가워 -> 'no'\n\n"
        "프롬프트의 내용이 답변에 들어가지 않도록 주의해주세요.\n\n"
        "Context: {text}\n"
        "Answer:"
    )
    return prompt_.format(text=text)


def science_response(text, max_retries=3):
    is_prompt = self_prompt(text=text)
    retries = 0
    while retries < max_retries:
        try:
            response_ = client.chat.completions.create(
                model='gpt-3.5-turbo-0125',
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": is_prompt}
                ],
                temperature=0,
                max_tokens=3,
                timeout=60,
            )
            content = response_.choices[0].message.content

            if len(content) > 10:
                print(f"Response too long, retrying... ({retries + 1}/{max_retries})")
                retries += 1
            else:
                break
        except Exception as e:
            print(f"Error: {e}")
            retries += 1
        time.sleep(1)

    print('is science:', content)

    return content


def format_docs(docs):
    global references
    references = docs
    return '\n\n'.join(doc.page_content for doc in docs)

def answer_question(messages):
    global references

    history = '\n'.join([f"{message['role']}: {message['content']}" for message in messages]) + '\n'

    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | chat
        | StrOutputParser()
    )

    is_science = science_response(text=history).lower()

    response = {
        "topk": "", 
        "is_science": "", 
        "answer": "", 
        "references": ""
    }

    # 과학 상식과 관련한 질의일 때
    if 'yes' in is_science:

        answer = rag_chain.invoke(history)

        ref_content = [reference.page_content for reference in references]
        topk = [reference.metadata['docid'] for reference in references]


    # 과학 상식과 관련한 질의가 아닐 때
    elif 'no' in is_science:
        response = {"topk": "", "is_science": "", "answer": "", "references": ""}

        answer = chat.invoke(history).content

        ref_content = []
        topk = []

    response["topk"] = topk
    response["is_science"] = is_science
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
            output = {"eval_id": j["eval_id"], "topk": response["topk"], "is_science": response["is_science"], "answer": response["answer"], "references": response["references"]}
            output_lines.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

# 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
eval_rag('../data/eval.jsonl', '../submit/EXP06.csv')