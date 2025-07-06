import re

from langchain_core.prompts import PromptTemplate
from loguru import logger

from nodes.chat_node.chat_prompt import prompt_template
from model import Model


class ChatNode:
    def __init__(self):
        self.model = Model()

    def chat(
            self,
            npc_name: str,
            npc_description: str,
            context: str,
            emotion: str,
            memories: list,
            player_input: str
    ) -> str:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "npc_name", "npc_description", "context", "emotion",
                "memories", "player_input"
            ]
        )
        chat_chain = prompt | self.model.llm
        result = chat_chain.invoke(
            {
                "npc_name": npc_name,
                "npc_description": npc_description,
                "context": context,
                "emotion": emotion,
                "memories": memories,
                "player_input": player_input
            }
        )
        mono_logue = self.parse_monologue(result.content)
        answer = self.parse_answer(result.content)

        logger.info(f"monologue: {mono_logue}")
        return answer

    def parse_monologue(self, response: str):
        pattern = r"Inner Monologue:\s*(.*?)(?=Final Response:|$)"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else "No inner monologue found"


    def parse_answer(self, response: str):
        """Final Response 부분만 추출"""
        pattern = r"Final Response:\s*(.*?)$"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else response.strip()
