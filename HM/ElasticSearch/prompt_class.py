class Prompt:
    @staticmethod
    def get_persona_qa():
        return """
        ## Role: 과학 상식 전문가

        ## Instructions
        - 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
        - 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다. 
        - 한국어로 답변을 생성한다.
        """

    @staticmethod
    def get_persona_function_calling():
        return """
        ## Role: 과학 상식 전문가

        ## Instruction
        - 사용자가 대화를 통해 과학 지식에 관한 주제로 질문하면 search api를 호출할 수 있어야 한다.
        - 과학 상식과 관련되지 않은 나머지 대화 메시지에는 적절한 대답을 생성한다. 
        """

    @staticmethod
    def get_tools():
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "search relevant documents",
                    "parameters": {
                        "properties": {
                            "standalone_query": {
                                "type": "string",
                                "description": "사용자 메세지 내역으로부터 검색에 사용할 적합한 최종 질의"
                            }
                        },
                        "required": ["standalone_query"],
                        "type": "object"
                    }
                }
            },
        ]