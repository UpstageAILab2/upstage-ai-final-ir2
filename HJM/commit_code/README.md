# 실험 설명

## EXP01

- sentence transformers 변경 

    - `snunlp/KR-SBERT-V40K-klueNLI-augSTS` → `intfloat/multilingual-e5-base`

- LB Score

    - MAP: 0.7121
    
    - MRR: 0.7167

## EXP02

- LLM Model 변경

    - `gpt-3.5-turbo-1106` → `solar-1-mini-chat`

- LB Score (베이스라인 변경이 필요하다고 느낌)

    - MAP: 0.1273
    
    - MRR: 0.1273

## EXP03

- LangChain 적용

    - Vector Store: `FAISS`
    
    - Embedding Model: `solar-embedding-1-large`

    - LLM Model: `solar-1-mini-chat`

- LB Score

    - MAP: 0.8045
    
    - MRR: 0.8076

## EXP04

- Text Splitter 적용

    - `CharacterTextSplitter`

- LB Score

    - MAP: 0.8205
    
    - MRR: 0.8318

## EXP05

- Text Splitter 변경

    - `CharacterTextSplitter` → `RecursiveCharacterTextSplitter`

- LB Score

    - MAP: 0.7985
    
    - MRR: 0.8061

## EXP06

- Text Splitter 변경

    - `RecursiveCharacterTextSplitter` → ~~`SemanticChunker`~~ → `CharacterTextSplitter`

- Search Type 변경 X

    - `Similarity` → ~~`MMR`~~

    - 이전 실험들에선 `Similarity`만 사용했다. 다양한 정보를 통해 답변을 생성하면 좋을 것 같아서 MMR 을 사용했으나 MAP 성능이 0.6~0.7 정도로 나왔다.

- Embedding Model 실험

    - 이전 실험들에서 `solar-embedding-1-large`를 사용했다. 이 이유는 OpenAI 보다 한국어 임베딩 성능이 더 좋았기 때문이다. 그래서 이번에 OpenAI 임베딩도 써봤지만 Solar 임베딩이 훨씬 나았다. 따라서 `solar-embedding-1-large`를 쭉 사용하려고 한다.

- LLM Model 2가지 활용

    - 답변을 생성할 때 `solar-1-mini-chat`를 사용하고 있었다. 하지만 과학 지식 질의가 아닌 것을 처리해주기 위해 `gpt-3.5-turbo-0125`도 같이 사용하게 되었다. 즉, 과학 지식 질의 여부를 처리하는건 gpt, 답변 생성은 solar로 사용하였다.

- 과학 지식 질의 아닌 것 처리

    - Retriever를 사용하기 전, 질문 쿼리의 과학 지식 질의 여부를 yes or no로 판단한다. yes면 문서를 검색하고, no면 문서를 검색하지 않고 바로 답변을 생성한다.

- LB Score

    - MAP: 0.9000
    
    - MRR: 0.9030

## EXP07

- 