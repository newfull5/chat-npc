import asyncio

from loguru import logger

from pipeline import AgentState, Pipeline


async def main():
    pipeline = Pipeline()

    # Case 1: 초보자 NPC
    result = await pipeline.arun(
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
    )
    logger.info(f'npc chat answer: {result["answer"]}')
    """
    >>> npc chat answer: hat's fantastic to hear! I'm glad you're enjoying it so far. If you need any tips or guidance as you explore, just let me know—I'm here to help you make the most of your adventure!
    """

    # Case 2: 어려운 던전 안내 NPC
    result = await pipeline.arun(
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
    )

    logger.info(f'npc chat answer: {result["answer"]}')
    """
    >>> npc chat answer: I know this boss can be tough, but don't get discouraged! Sometimes it takes a few tries to learn the patterns. Take a deep breath, maybe try adjusting your strategy, and remember—every attempt gets you closer to victory. You've got this!
    """


if __name__ == "__main__":
    asyncio.run(main())