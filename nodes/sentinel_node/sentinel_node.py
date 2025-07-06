import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
from transformers import pipeline
from model import Model
from nodes.sentinel_node.type import GameContext, EmotionResult


class SentinelNode:
    def __init__(self):
        self.previous_embedding = None
        self.previous_emotion = None
        self.model = Model(llm_model="gpt-4.1-nano", embedding_model="text-embedding-ada-002")
        self.classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")


    def _serialize_context(self, context: GameContext):
        """
        paper에서: player location, current quest stage, and other contextual information
        """

        game_context = GameContext(**{k: v for k, v in context.items() if k in GameContext.get_important_keys()})
        serialized = game_context.serialize()

        remaining_keys = sorted([k for k in context.keys() if k not in GameContext.get_important_keys()])

        remaining_parts = []
        for key in remaining_keys:
            remaining_parts.append(f"{key}: {context[key]}")

        if remaining_parts:
            return serialized + ";\n" + ";\n".join(remaining_parts)

        return serialized


    def _calculate_similarity(self, emb1: list, emb2: list):
        emb1 = np.array(emb1).reshape(1, -1)
        emb2 = np.array(emb2).reshape(1, -1)
        return cosine_similarity(emb1, emb2)[0][0]


    def analyze_emotion(self, player_input: str) -> EmotionResult:
        result = self.classifier(player_input)

        current_emotion = result[0]["label"]
        emotion_score = result[0]["score"]

        emotion_changed = False
        if self.previous_emotion is not None and self.previous_emotion != current_emotion:
            emotion_changed = True
            logger.info(f"previous emotion: {self.previous_emotion}, current emotion: {current_emotion}")

        emotion_result = EmotionResult(
            detected_emotion=current_emotion,
            emotion_score=emotion_score,
            emotion_changed=emotion_changed,
            previous_emotion=self.previous_emotion
        )

        self.previous_emotion = current_emotion

        return emotion_result


    def get_emotion_prompt_context(self, emotion_result: EmotionResult) -> dict:
        emotion_context = {
            "detected_emotion": emotion_result.detected_emotion,
            "emotion_score": round(emotion_result.emotion_score, 3),
            "emotion_changed": emotion_result.emotion_changed
        }

        if emotion_result.previous_emotion:
            emotion_context["previous_emotion"] = emotion_result.previous_emotion

        return emotion_context


    def detect_context_change(self, current_context: GameContext, threshold: float = 0.5):
        """
        논문에서는 nlu 모델 학습해서 사용함
        """
        current_context_text = self._serialize_context(current_context)

        current_embedding = self.model.emb_query(current_context_text)
        if self.previous_embedding is None:
            self.previous_embedding = current_embedding
            return False

        similarity = self._calculate_similarity(self.previous_embedding, current_embedding)
        self.previous_embedding = current_embedding

        if similarity > threshold:
            return False

        logger.info("in game emotion chages")
        return True


    def process_player_input(self, player_input: str, game_context: dict) -> dict:
        emotion_result = self.analyze_emotion(player_input)

        context_changed = self.detect_context_change(game_context)

        # 결과 통합
        return {
            "emotion_analysis": emotion_result,
            "emotion_context": self.get_emotion_prompt_context(emotion_result),
            "context_changed": context_changed,
            "should_retrieve_memory": emotion_result.emotion_changed or context_changed
        }
