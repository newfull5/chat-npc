import asyncio
import uuid
from datetime import datetime

import numpy as np
from beanie import Document, init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import Field
from sklearn.metrics.pairwise import cosine_similarity

from model import Model


class Memory(Document):
    memory_id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:7]}")
    embedding: list[float] = Field(default_factory=list)
    text: str = Field(default="")
    memory_type: str = Field(default="general")  # quest_event, player_trait, npc_dialogue, world_event 등
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    player_id: str = Field(default_factory=lambda: f"player_{uuid.uuid4().hex[:7]}")

    class Settings:
        name = "game_memories"  # MongoDB 컬렉션 이름
        indexes = [
            "memory_id",
            "player_id",
            "memory_type",
            "timestamp"
        ]

    def __repr__(self):
        emb_preview = (
            f"{self.embedding[:2]}...({len(self.embedding)} dims)"
            if self.embedding else "None"
        )
        return (
            f"Memory(memory_id={self.memory_id!r}, text={self.text!r}, "
            f"embedding={emb_preview}, memory_type={self.memory_type!r}, "
            f"timestamp={self.timestamp!r}, player_id={self.player_id!r})"
        )


class MemoryNode:
    def __init__(self):
        self.client = None
        self.database = None
        self.model = Model()
        self.connection_string = "mongodb://localhost:1706"
        self.database_name = "User"

    async def init_db(self):
        self.client = AsyncIOMotorClient(self.connection_string)
        self.database = self.client[self.database_name]
        await init_beanie(database=self.database, document_models=[Memory])

    async def create_memory(self, location: str, quest_stage: str, player_emotion: str, text: str, player_id: str | None) -> Memory:
        query_text = f"location: {location} quest: {quest_stage} emotion: {player_emotion} text: {text}"
        query_embedding = await self.model.aemb_query(query_text)

        if not player_id:
            player_id = f"player_{uuid.uuid4().hex[:7]}"

        user = Memory(
            embedding=query_embedding,
            text=text,
            player_id=player_id,
        )
        await user.insert()
        return user

    async def get_memory(self, player_id: str) -> list[Memory]:
        return await Memory.find(Memory.player_id == player_id).to_list()

    async def search_user(self, player_id: str, location: str, quest_stage: str, player_emotion: str, text: str) -> list[Memory]:
        query_text = f"location: {location} quest: {quest_stage} emotion: {player_emotion} text: {text}"
        query_embedding = await self.model.aemb_query(query_text)

        memories = await Memory.find(Memory.player_id == player_id).to_list()

        scored = []
        for mem in memories:
            if not mem.embedding:
                continue
            score = cosine_similarity(
                np.array(query_embedding).reshape(1, -1),
                np.array(mem.embedding).reshape(1, -1)
            )[0][0]
            scored.append((mem, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_memories = [m for m, _ in scored[:10]]
        return top_memories

