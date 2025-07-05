import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class Model:
    def __init__(self, llm_model, temperature):
        load_dotenv()
        self.llm = ChatOpenAI(
            temperature=temperature,
            model=llm_model,
        )

    def run(self, inputs):
        prompt = PromptTemplate(template='{question} {documents}', input_variables=["question", "documents"])
        chat_chain = prompt | self.llm
        result = chat_chain.invoke({"question": '', "documents": ''})
        return result.content

