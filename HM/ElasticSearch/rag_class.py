import os
import json
from openai import OpenAI
import traceback
from prompt_class import Prompt

class RAG:
    def __init__(self, api_key=None, model="gpt-4-turbo", retrieval_instance=None):
        load_dotenv()  # .env 파일에서 환경 변수 로드
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.client = OpenAI()        
        self.llm_model = model
        self.retrieval_instance = retrieval_instance
        self.prompt = Prompt()

    def answer_question(self, messages):
        response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

        msg = [{"role": "system", "content": self.prompt.get_persona_function_calling()}] + messages
        try:
            result = self.client.chat.completions.create(
                model=self.llm_model,
                messages=msg,
                tools=self.prompt.get_tools(),
                temperature=0,
                seed=1,
                timeout=10
            )
        except Exception as e:
            traceback.print_exc()
            return response

        if result.choices[0].message.tool_calls:
            tool_call = result.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            standalone_query = function_args.get("standalone_query")

            sparse_results = self.retrieval_instance.sparse_retrieve(standalone_query, 10)
            dense_results = self.retrieval_instance.dense_retrieve_2(standalone_query, sparse_results['hits']['hits'], 3)

            response["standalone_query"] = standalone_query
            retrieved_context = []
            for i, rst in enumerate(dense_results['hits']['hits']):
                retrieved_context.append(rst["_source"]["content"])
                response["topk"].append(rst["_source"]["docid"])
                response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})

            content = json.dumps(retrieved_context)
            messages.append({"role": "assistant", "content": content})
            msg = [{"role": "system", "content": self.prompt.get_persona_qa()}] + messages
            try:
                qaresult = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=msg,
                    temperature=0,
                    seed=1,
                    timeout=30
                )
            except Exception as e:
                traceback.print_exc()
                return response
            response["answer"] = qaresult.choices[0].message.content
        else:
            response["answer"] = result.choices[0].message.content

        return response