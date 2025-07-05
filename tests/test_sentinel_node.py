# test_sentinel_node.py
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from nodes.sentinel_node.sentinel_node import SentinelNode
from nodes.sentinel_node.type import GameContext, EmotionResult


class TestSentinelNode:
    """SentinelNode 테스트 클래스"""

    @pytest.fixture
    def mock_model(self):
        """Model 의존성 모킹"""
        with patch('nodes.sentinel_node.sentinel_node.Model') as mock:
            mock_instance = Mock()
            mock_instance.emb_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_pipeline(self):
        """Transformers pipeline 모킹"""
        with patch('nodes.sentinel_node.sentinel_node.pipeline') as mock:
            mock_classifier = Mock()
            mock_classifier.return_value = [{"label": "joy", "score": 0.95}]
            mock.return_value = mock_classifier
            yield mock_classifier

    @pytest.fixture
    def sentinel_node(self, mock_model, mock_pipeline):
        """SentinelNode 인스턴스 생성"""
        return SentinelNode()

    @pytest.fixture
    def sample_game_context(self):
        """샘플 게임 컨텍스트"""
        return {
            "location": "forest",
            "quest": "find_artifact",
            "hp": 80,
            "mp": 50,
            "status": "healthy",
            "extra_key": "extra_value"
        }

    def test_init(self, sentinel_node):
        """초기화 테스트"""
        assert sentinel_node.previous_embedding is None
        assert sentinel_node.previous_emotion is None
        assert sentinel_node.model is not None
        assert sentinel_node.classifier is not None

    def test_serialize_context_with_important_keys(self, sentinel_node, sample_game_context):
        """중요한 키들이 있는 컨텍스트 직렬화 테스트"""
        result = sentinel_node._serialize_context(sample_game_context)

        # 중요한 키들이 포함되어 있는지 확인
        assert "location: forest" in result
        assert "quest: find_artifact" in result
        assert "hp: 80" in result
        assert "extra_key: extra_value" in result

    def test_serialize_context_empty(self, sentinel_node):
        """빈 컨텍스트 직렬화 테스트"""
        result = sentinel_node._serialize_context({})
        assert result == ""

    def test_calculate_similarity(self, sentinel_node):
        """유사도 계산 테스트"""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        similarity = sentinel_node._calculate_similarity(emb1, emb2)
        assert similarity == pytest.approx(1.0, rel=1e-5)

        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        similarity = sentinel_node._calculate_similarity(emb1, emb2)
        assert similarity == pytest.approx(0.0, rel=1e-5)

    def test_analyze_emotion_first_time(self, sentinel_node, mock_pipeline):
        """첫 번째 감정 분석 테스트"""
        player_input = "I love this game!"

        result = sentinel_node.analyze_emotion(player_input)

        assert isinstance(result, EmotionResult)
        assert result.detected_emotion == "joy"
        assert result.emotion_score == 0.95
        assert result.emotion_changed is False
        assert result.previous_emotion is None
        assert sentinel_node.previous_emotion == "joy"

    def test_analyze_emotion_no_change(self, sentinel_node, mock_pipeline):
        """감정 변화 없음 테스트"""
        sentinel_node.previous_emotion = "joy"

        result = sentinel_node.analyze_emotion("I still love this!")

        assert result.detected_emotion == "joy"
        assert result.emotion_changed is False
        assert result.previous_emotion == "joy"

    def test_analyze_emotion_with_change(self, sentinel_node, mock_pipeline):
        """감정 변화 있음 테스트"""
        sentinel_node.previous_emotion = "joy"
        mock_pipeline.return_value = [{"label": "anger", "score": 0.88}]

        result = sentinel_node.analyze_emotion("I hate this quest!")

        assert result.detected_emotion == "anger"
        assert result.emotion_changed is True
        assert result.previous_emotion == "joy"
        assert sentinel_node.previous_emotion == "anger"

    def test_get_emotion_prompt_context_basic(self, sentinel_node):
        """기본 감정 프롬프트 컨텍스트 생성 테스트"""
        emotion_result = EmotionResult(
            detected_emotion="joy",
            emotion_score=0.95,
            emotion_changed=False,
            previous_emotion=None
        )

        context = sentinel_node.get_emotion_prompt_context(emotion_result)

        expected = {
            "detected_emotion": "joy",
            "emotion_score": 0.95,
            "emotion_changed": False
        }
        assert context == expected

    def test_get_emotion_prompt_context_with_previous(self, sentinel_node):
        """이전 감정 포함 프롬프트 컨텍스트 테스트"""
        emotion_result = EmotionResult(
            detected_emotion="anger",
            emotion_score=0.88,
            emotion_changed=True,
            previous_emotion="joy"
        )

        context = sentinel_node.get_emotion_prompt_context(emotion_result)

        expected = {
            "detected_emotion": "anger",
            "emotion_score": 0.88,
            "emotion_changed": True,
            "previous_emotion": "joy"
        }
        assert context == expected

    def test_detect_context_change_first_time(self, sentinel_node, sample_game_context):
        """첫 번째 컨텍스트 변화 감지 테스트"""
        result = sentinel_node.detect_context_change(sample_game_context)

        assert result is False
        assert sentinel_node.previous_embedding is not None

    def test_detect_context_change_no_change(self, sentinel_node, sample_game_context, mock_model):
        """컨텍스트 변화 없음 테스트"""
        # 첫 번째 호출
        sentinel_node.detect_context_change(sample_game_context)

        # 두 번째 호출 - 같은 컨텍스트
        mock_model.emb_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]  # 같은 임베딩
        result = sentinel_node.detect_context_change(sample_game_context)

        assert result is False

    def test_detect_context_change_with_change(self, sentinel_node, sample_game_context, mock_model):
        """컨텍스트 변화 있음 테스트"""
        # 첫 번째 호출
        sentinel_node.detect_context_change(sample_game_context)

        # 두 번째 호출 - 다른 임베딩
        mock_model.emb_query.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]  # 다른 임베딩
        result = sentinel_node.detect_context_change(sample_game_context, threshold=0.8)

        assert result is True

    def test_process_player_input_complete(self, sentinel_node, mock_pipeline, sample_game_context):
        """플레이어 입력 종합 처리 테스트"""
        player_input = "I love this adventure!"

        result = sentinel_node.process_player_input(player_input, sample_game_context)

        # 결과 구조 검증
        assert "emotion_analysis" in result
        assert "emotion_context" in result
        assert "context_changed" in result
        assert "should_retrieve_memory" in result

        # 감정 분석 결과 검증
        assert isinstance(result["emotion_analysis"], EmotionResult)
        assert result["emotion_analysis"].detected_emotion == "joy"

        # 감정 컨텍스트 검증
        assert result["emotion_context"]["detected_emotion"] == "joy"
        assert result["emotion_context"]["emotion_score"] == 0.95

    def test_process_player_input_emotion_changed(self, sentinel_node, mock_pipeline, sample_game_context):
        """감정 변화 시 메모리 검색 플래그 테스트"""
        # 첫 번째 감정 설정
        sentinel_node.previous_emotion = "joy"

        # 감정 변화 시뮬레이션
        mock_pipeline.return_value = [{"label": "anger", "score": 0.85}]

        result = sentinel_node.process_player_input("I hate this!", sample_game_context)

        assert result["emotion_analysis"].emotion_changed is True
        assert result["should_retrieve_memory"] is True

    def test_process_player_input_context_changed(self, sentinel_node, mock_pipeline, sample_game_context, mock_model):
        """컨텍스트 변화 시 메모리 검색 플래그 테스트"""
        # 첫 번째 컨텍스트 설정
        sentinel_node.detect_context_change(sample_game_context)

        # 컨텍스트 변화 시뮬레이션
        mock_model.emb_query.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]  # 다른 임베딩

        result = sentinel_node.process_player_input("Hello", sample_game_context)

        assert result["context_changed"] is True
        assert result["should_retrieve_memory"] is True


class TestIntegration:
    """통합 테스트"""

    def test_emotion_workflow(self):
        """감정 분석 워크플로우 통합 테스트"""
        with patch('nodes.sentinel_node.sentinel_node.Model') as mock_model, \
                patch('nodes.sentinel_node.sentinel_node.pipeline') as mock_pipeline:

            # 모킹 설정
            mock_model_instance = Mock()
            mock_model_instance.emb_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_model.return_value = mock_model_instance

            mock_classifier = Mock()
            mock_classifier.side_effect = [
                [{"label": "joy", "score": 0.95}],    # 첫 번째 호출
                [{"label": "anger", "score": 0.88}],  # 두 번째 호출
                [{"label": "anger", "score": 0.90}],  # 세 번째 호출
            ]
            mock_pipeline.return_value = mock_classifier

            sentinel = SentinelNode()
            game_context = {"location": "forest", "quest": "find_artifact"}

            # 첫 번째 입력 - 기쁨
            result1 = sentinel.process_player_input("I love this game!", game_context)
            assert result1["emotion_analysis"].detected_emotion == "joy"
            assert result1["emotion_analysis"].emotion_changed is False

            # 두 번째 입력 - 분노 (감정 변화)
            result2 = sentinel.process_player_input("I hate this quest!", game_context)
            assert result2["emotion_analysis"].detected_emotion == "anger"
            assert result2["emotion_analysis"].emotion_changed is True
            assert result2["should_retrieve_memory"] is True

            # 세 번째 입력 - 분노 (감정 변화 없음)
            result3 = sentinel.process_player_input("This is so frustrating!", game_context)
            assert result3["emotion_analysis"].detected_emotion == "anger"
            assert result3["emotion_analysis"].emotion_changed is False


# 실행 가능한 테스트 예시
if __name__ == "__main__":
    pytest.main([__file__, "-v"])