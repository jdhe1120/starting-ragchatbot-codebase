import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_rag_system():
    rag = MagicMock()
    rag.query.return_value = ("Test answer", ["Source A"])
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course One", "Course Two"],
    }
    rag.session_manager.create_session.return_value = "test-session-id"
    return rag


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.ANTHROPIC_API_KEY = "fake-key"
    config.ANTHROPIC_MODEL = "claude-test-model"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./fake_chroma"
    return config
