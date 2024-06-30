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

- 

- LB Score

    - MAP: 
    
    - MRR: 