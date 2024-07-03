from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
import cohere

from openai import OpenAI

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings, ChatUpstage, UpstageGroundednessCheck


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
UPSTAGE_API_KEY = os.environ.get('UPSTAGE_API_KEY')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')
LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = 'EXP08' # 프로젝트명 수정
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


# Embedding
embeddings = UpstageEmbeddings(
    api_key=UPSTAGE_API_KEY,
    model='solar-embedding-1-large'
)


# 벡터 저장소 생성
# pip install faiss-cpu
folder_path = f'./vectorstore/EXP07_3'
if not os.path.exists(folder_path):
    print(f'"{folder_path}" create ...')
    
    splitter = CharacterTextSplitter(
        separator='. ',
        chunk_size=300,
        chunk_overlap=30,
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


# vectorstore as retriever
retriever = vectorstore.as_retriever(
    search_type='similarity', 
    search_kwargs={'k': 8}
)


# langchain chat
chat = ChatUpstage(
    api_key=UPSTAGE_API_KEY,
    model='solar-1-mini-chat', 
    temperature=0,
    max_tokens=250,
)


# Cohere client
co = cohere.Client(api_key=COHERE_API_KEY)

# OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Upstage client
upstage_client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)

# Upstage Groundedness Check
groundedness_check = UpstageGroundednessCheck(upstage_api_key=UPSTAGE_API_KEY)


# 사회/과학 지식 질문 여부를 파악하는 프롬프트
def is_prompt(question):
    prompt = (
        "당신은 사회/과학 지식 전문가 입니다."
        " 주어진 Context가 사회/과학 상식과 관련된 질문인지 판별할 수 있도록 'yes' 또는 'no'로 답변해주세요."
        " 제공되는 예시는 일상 대화 중 나오는 질문입니다. 이러한 일상 대화가 아닌 질의는 'yes'로 답변해주세요.\n\n"
        "예시:\n"
        "요새 너무 힘들어. -> 'no'\n"
        "대답을 잘해줘서 너무 좋아! -> 'no'\n"
        "그만 얘기하자. -> 'no'\n"
        "신나는 얘기해줘! -> 'no'\n"
        "모르는게 뭐야? -> 'no'\n"
        "잘하는게 뭐야? -> 'no'\n"
        "똑똑하구나. -> 'no'\n"
        "안녕. -> 'no'\n\n"
        "프롬프트의 내용이 답변에 들어가지 않도록 주의해주세요.\n\n"
        "Context: {question}\n"
        "Answer:"
    )
    return prompt.format(question=question)


# RAG 구현에 필요한 Question Answering을 위한 프롬프트
def rag_prompt(context, question):
    context = '\n'.join(context)
    prompt = (
        "You are an assistant for answering questions related to social and scientific knowledge." 
        " Use the following pieces of retrieved context to answer the question."
        " If you don't know the answer, just say that you don't know."
        " Use three sentences maximum and keep the answer concise.\n"
        "Question: {question}\n"
        "Context: {context}\n" 
        "Answer:"
    )
    return prompt.format(context=context, question=question)


# 답변을 생성하는 LLM
def create_response(client, self_prompt, model, max_tokens, retry_len, max_retries=3):
    client = client

    prompt = self_prompt
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=max_tokens,
                timeout=60,
            )
            content = response.choices[0].message.content

            if len(content) > retry_len:
                print(f"Response too long, retrying... ({retries + 1}/{max_retries})")
                retries += 1
            else:
                break
        except Exception as e:
            print(f"Error: {e}")
            retries += 1
        time.sleep(1)

    return content


def answer_question(messages):
    save_datas = {
        "topk": "", 
        "is_science": "", 
        "answer": "", 
        "references": ""
    }

    history = '\n'.join([f"{message['role']}: {message['content']}" for message in messages]) + '\n'

    is_science = create_response(
        client=openai_client, 
        self_prompt=is_prompt(question=history), 
        model='gpt-3.5-turbo-0125', 
        max_tokens=3, 
        retry_len=10
    ).lower()

    if 'yes' in is_science:
        # Document 검색
        org_docs = retriever.invoke(history)

        # Groundedness Check
        gc_docs = []
        gc_values = []
        for doc in org_docs:
            request_input = {
                "context": history,
                "answer": doc.page_content,
            }
            grounded_response = groundedness_check.invoke(request_input)
            gc_values.append(grounded_response)

            if grounded_response == 'grounded':
                gc_docs.append(doc)

        gc_docs_length = len(gc_docs)
        if gc_docs_length <= 0:
            gc_docs = org_docs
        print('Groundedness Check:', ', '.join(gc_values))
        print('Groundedness Check 문서 개수: ', gc_docs_length)

        # Reranker - Cohere
        if 0 < gc_docs_length < 5:
            top_k = gc_docs_length
        else:
            top_k = 5

        gc_docs_contents = [doc.page_content for doc in gc_docs]

        co_rerank = co.rerank(
            model="rerank-multilingual-v3.0",
            query=history,
            documents=gc_docs_contents,
            top_n=top_k,
        )

        idxs = [co_rerank.results[i].index for i in range(top_k)]
        rerank_docs = [gc_docs[idx] for idx in idxs]

        print('Rerank 문서 개수:', len(rerank_docs))

        rerank_context = '\n\n'.join(doc.page_content for doc in rerank_docs)

        answer = create_response(
            client=upstage_client, 
            self_prompt=rag_prompt(context=rerank_context, question=history), 
            model='solar-1-mini-chat', 
            max_tokens=250, 
            retry_len=500
        ).lower()

        ref_content = [doc.page_content for doc in rerank_docs]
        topk = [doc.metadata['docid'] for doc in rerank_docs]

    elif 'no' in is_science:
        answer = chat.invoke(history).content

        ref_content = []
        topk = []

    save_datas["topk"] = topk
    save_datas["is_science"] = is_science
    save_datas["answer"] = answer
    save_datas["references"] = ref_content

    return save_datas


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as eval_lines, open(output_filename, 'w') as output_lines:
        idx = 0
        for eval_line in eval_lines:
            j = json.loads(eval_line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            save_datas = answer_question(j["msg"])
            print(f'Is science: {save_datas["is_science"]}')
            print(f'Answer: {save_datas["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "topk": save_datas["topk"], "is_science": save_datas["is_science"], "answer": save_datas["answer"], "references": save_datas["references"]}
            output_lines.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

# 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
eval_rag('../data/eval.jsonl', '../submit/EXP08.csv')