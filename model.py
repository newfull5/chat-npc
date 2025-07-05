import torch
from loguru import logger
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class Model:
    def __init__(self, llm_model: str, embedding_model: str, temperature: float = 0.1):
        load_dotenv()
        self.llm = ChatOpenAI(
            temperature=temperature,
            model=llm_model,
        )
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model
        )

    def run(self, inputs):
        prompt = PromptTemplate(template='{question} {documents}', input_variables=["question", "documents"])
        chat_chain = prompt | self.llm
        result = chat_chain.invoke({"question": '', "documents": ''})
        return result.content

    def emb_query(self, text: str):
        return self.embeddings.embed_query(text)

    async def aemb_query(self, text: str):
        return await self.embeddings.aembed_query(text)

    @staticmethod
    def get_device() -> torch.device:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("🍎 Now using Apple MPS for running LLM (Metal Performance Shaders)")

        elif torch.cuda.is_available():
            device = torch.device("cuda:1")
            logger.info("⚡️ Now using GPU (CUDA) for running LLM")

        else:
            device = torch.device("cpu")
            logger.info("🐢 Now using CPU for running LLM — things might be slower 🐌")

        return device


# model = Model(
#     llm_model="gpt-4.1-nano",
#     embedding_model="text-embedding-ada-002",
#     temperature=0.7)
#
# print(model.emb_query("What is the meaning of life?"))