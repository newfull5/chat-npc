from langgraph.graph import StateGraph, END, MessagesState

from nodes.chat_node.chat_node import ChatNode
from nodes.memory_node.memory_node import MemoryNode
from nodes.sentinel_node.sentinel_node import SentinelNode
from nodes.sentinel_node.type import GameContext


class AgentState(MessagesState):
    user_text: str
    player_id: str
    load_memories: list
    npc_name: str
    npc_description: str
    player_emotion: str | None = None
    location: str | None = None
    quest: str | None = None
    hp: int | None = None
    mp: int | None = None
    status: str | None = None
    nearby: str | None = None
    event_flags: list[str] | None = None
    recent_action: str | None = None
    context_change: bool = False
    answer: str | None = None

class Pipeline:
    def __init__(self):
        self.sentinel_module = SentinelNode()
        self.memory_module = MemoryNode()
        self.chat_node = ChatNode()
        self.player_context = GameContext( # default
            location="forest",
            quest="find_artifact",
            hp=80,
            mp=50,
            status="healthy"
        )
        self.app = self._build_graph()

    def detect_context_change(self, state: AgentState):
        result = self.sentinel_module.detect_context_change(
            GameContext(
                location=state.get("location", self.player_context.location),
                quest=state.get("quest", self.player_context.quest),
                hp=state.get("hp", self.player_context.hp),
                mp=state.get("mp", self.player_context.mp),
                status=state.get("status", self.player_context.status)
            ),
        )
        return {"messages": state["messages"], "context_change": result}


    def analyze_emotion(self, state: AgentState):
        result = self.sentinel_module.analyze_emotion(state.get("user_text", ""))
        return {"messages": state["messages"], "player_emotion": result}

    async def search_memory(self, state: AgentState):
        result = await self.memory_module.search_user(
            text=state.get("user_text", ""),
            player_id=state.get("player_id", ""),
            location=state.get("location", self.player_context.location),
            quest_stage=state.get("quest", ""),
            player_emotion=state.get("player_emotion", "")
        )
        return {"messages": state["messages"], "load_memories": result}

    async def update_memory(self, state: AgentState):
        await self.memory_module.create_memory(
            text=state.get("user_text", ""),
            location=state.get("location", self.player_context.location),
            quest_stage=state.get("quest", ""),
            player_emotion=state.get("player_emotion", ""),
            player_id=state.get("player_id", "")
        )

        return state

    def generate_chat(self, state: AgentState):
        result = self.chat_node.chat(
            npc_name="test_npc",
            npc_description="test_npc",
            context=state.get("user_text", ""),
            emotion=state.get("player_emotion", ""),
            memories=state.get("load_memories", []),
            player_input=state.get("user_text", "")
        )
        return {"messages": state["messages"], "answer": result}


    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("detect_context_change", self.detect_context_change)
        graph.add_node("analyze_emotion", self.analyze_emotion)
        graph.add_node("search_memory", self.search_memory)
        graph.add_node("update_memory", self.update_memory)
        graph.add_node("generate_chat", self.generate_chat)

        graph.set_entry_point("detect_context_change")
        graph.add_conditional_edges(
            "detect_context_change",
            lambda state: "context_change" if state.get("context_change", False) else "no_change",
            {
                "context_change": "analyze_emotion",
                "no_change": "update_memory"
            }
        )
        graph.add_edge("analyze_emotion", "search_memory")
        graph.add_edge("search_memory", "update_memory")
        graph.add_edge("update_memory", "generate_chat")
        graph.add_edge("generate_chat", END)

        return graph.compile()

    async def arun(self, state: AgentState):
        await self.memory_module.init_db()
        return await self.app.ainvoke(state)
