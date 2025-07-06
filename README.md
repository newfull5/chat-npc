
## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Poetry 설치 (macOS)
brew install poetry

# 의존성 설치
poetry install

# 가상환경 활성화
poetry shell
```

### 2. 실행

```bash
python main.py
```

## 📝 사용 예제

```python
import asyncio
from pipeline import AgentState, Pipeline

async def main():
    pipeline = Pipeline()
    
    # NPC와의 대화 실행
    result = await pipeline.arun(
        AgentState(
            messages=[{"role": "user", "content": "안녕하세요!"}],
            user_text="안녕하세요!",
            npc_name="엘레나",
            npc_description="친절한 마을 가이드",
            player_id="player_123",
            location="시작 마을",
            quest="튜토리얼 완료",
            hp=100,
            mp=50,
            status="건강함"
        )
    )
    
    print(f"NPC 응답: {result['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```


### SentinelNode
- 컨텍스트 변화 감지 (위치, 퀘스트, 상태 변경)
- 플레이어 감정 상태 분석

### MemoryNode
- 대화 이력 저장 및 검색
- 플레이어별 개인화된 메모리 관리

### ChatNode
- 컨텍스트 기반 자연스러운 대화 생성
- NPC 특성과 상황에 맞는 응답 생성
