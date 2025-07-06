## Overview

# ChatNPC: LLM-based Context-Aware Flexible NPC System
An intelligent dialogue system that creates immersive gaming experiences through context-aware NPC interactions.

Traditional NPCs rely on fixed, pre-scripted responses that break immersion. This system generates **dynamic, context-aware dialogue** that adapts to:
- Player's emotional state
- Game context changes (location, quest, HP/MP)
- Historical interactions

<img width="502" height="800" alt="Image" src="https://github.com/user-attachments/assets/8acb2953-9bf1-4a9d-95ed-7a99fefbf211" />

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



## Example Interactions
### Case 1: Beginner Player
**Input:**
``` python
AgentState(
    messages=[{"role": "user", "content": "This is amazing! I just started playing!"}],
    user_text="This is amazing! I just started playing!",
    npc_name="Elena",
    npc_description="A cheerful village guide who loves helping newcomers learn the game",
    player_id="player_0d084ad",
    location="starting_village",
    quest="tutorial_basics",
    hp=100,
    mp=20,
    status="excited"
)
```
**Output:**
``` 
"""
That's fantastic to hear! 
I'm glad you're enjoying it so far. 
If you need any tips or guidance as you explore, just let me know—I'm here to help you make the most of your adventure!
"""
```

### Case 2: Struggling Player
**Input:**
``` python
AgentState(
    messages=[{"role": "user", "content": "This boss is impossible! I keep dying!"}],
    user_text="This boss is impossible! I keep dying!",
    npc_name="Gareth",
    npc_description="A battle-scarred veteran warrior who has conquered many dungeons",
    player_id="player_0d084ad",
    location="shadow_dungeon",
    quest="defeat_dark_lord",
    hp=15,
    mp=5,
    status="injured"
)
```
**Output:**
``` 
"""
I know this boss can be tough, but don't get discouraged! 
Sometimes it takes a few tries to learn the patterns. 
Take a deep breath, maybe try adjusting your strategy, and remember—every attempt gets you closer to victory.
You've got this!
"""
```


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
