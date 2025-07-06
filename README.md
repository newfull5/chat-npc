## Overview

# ChatNPC: LLM-based Context-Aware Flexible NPC System
An intelligent dialogue system that creates immersive gaming experiences through context-aware NPC interactions.

Traditional NPCs rely on fixed, pre-scripted responses that break immersion. This system generates **dynamic, context-aware dialogue** that adapts to:
- Player's emotional state
- Game context changes (location, quest, HP/MP)
- Historical interactions

### Core Components

1. **Sentinel Mechanism**
   - **Context Sentinel**: Detects significant game state changes using embedding comparison
   - **Emotion Sentinel**: Classifies player emotions using GoEmotions-based classifier

2. **Memory Manager**
   - **In-context**: Recent conversation history
   - **Out-of-context**: Long-term memory storage with semantic retrieval
   - RAG-based memory activation for relevant past interactions

3. **Chat Planning Agent**
   - Internal "pre-thinking" process before response generation
   - Emotion and context-aware dialogue synthesis


<img width="502" height="800" alt="Image" src="https://github.com/user-attachments/assets/8acb2953-9bf1-4a9d-95ed-7a99fefbf211" />

## Environment Setup

### venv setup
```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### env variables
create .env file in project root
```shell
# .env
OPENAI_API_KEY=your_openai_api_key_here
```

### Mongo DB Setup

```shell
# Start MongoDB container
docker-compose up -d

# Verify MongoDB is running
docker ps
```



### Usage Example

```python
from pipeline import AgentState, Pipeline

pipeline = Pipeline()

result = await pipeline.arun(
    AgentState(
        messages=[{"role": "user", "content": "This boss is impossible!"}],
        user_text="This boss is impossible!",
        npc_name="Battle Master",
        npc_description="Experienced warrior who provides combat guidance",
        player_id="player_123",
        location="dungeon",
        quest="defeat_boss",
        hp=15,
        mp=5,
        status="injured"
    )
)

```


## References

- ChatNPC: Towards Immersive Video Game Experience through Naturalistic and Emotive Dialogue Agent (ACL 2024 submit)
- The Turing Quest: Can Transformers Make Good NPCs? (ACL 2023)
- Hello Again! LLM-powered Personalized Agent for Long-term Dialogue (NAACL 2025)
