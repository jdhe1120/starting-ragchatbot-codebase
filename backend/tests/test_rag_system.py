import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_config():
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


@patch("rag_system.CourseOutlineTool")
@patch("rag_system.CourseSearchTool")
@patch("rag_system.ToolManager")
@patch("rag_system.SessionManager")
@patch("rag_system.AIGenerator")
@patch("rag_system.DocumentProcessor")
@patch("rag_system.VectorStore")
class TestRAGSystemQuery(unittest.TestCase):

    def _make_rag(self, MockVectorStore, MockDocProcessor, MockAIGen,
                  MockSessionMgr, MockToolMgr, MockSearchTool, MockOutlineTool):
        from rag_system import RAGSystem

        mock_ai = MagicMock()
        mock_ai.generate_response.return_value = "AI answer"
        MockAIGen.return_value = mock_ai

        mock_tool_mgr = MagicMock()
        mock_tool_mgr.get_last_sources.return_value = ["Source A"]
        mock_tool_mgr.get_tool_definitions.return_value = [{"name": "search_course_content"}]
        MockToolMgr.return_value = mock_tool_mgr

        mock_session = MagicMock()
        mock_session.get_conversation_history.return_value = "prev history"
        MockSessionMgr.return_value = mock_session

        rag = RAGSystem(_make_config())
        return rag, mock_ai, mock_tool_mgr, mock_session

    def test_query_returns_response_and_sources(
        self, MockVectorStore, MockDocProcessor, MockAIGen,
        MockSessionMgr, MockToolMgr, MockSearchTool, MockOutlineTool
    ):
        rag, mock_ai, mock_tool_mgr, _ = self._make_rag(
            MockVectorStore, MockDocProcessor, MockAIGen,
            MockSessionMgr, MockToolMgr, MockSearchTool, MockOutlineTool
        )
        response, sources = rag.query("what is RAG?")
        self.assertEqual(response, "AI answer")
        self.assertEqual(sources, ["Source A"])

    def test_query_wraps_user_query_in_prompt(
        self, MockVectorStore, MockDocProcessor, MockAIGen,
        MockSessionMgr, MockToolMgr, MockSearchTool, MockOutlineTool
    ):
        rag, mock_ai, _, _ = self._make_rag(
            MockVectorStore, MockDocProcessor, MockAIGen,
            MockSessionMgr, MockToolMgr, MockSearchTool, MockOutlineTool
        )
        rag.query("my user question")
        call_kwargs = mock_ai.generate_response.call_args[1]
        self.assertIn("Answer this question about course materials:", call_kwargs["query"])
        self.assertIn("my user question", call_kwargs["query"])

    def test_query_passes_tools_to_generator(
        self, MockVectorStore, MockDocProcessor, MockAIGen,
        MockSessionMgr, MockToolMgr, MockSearchTool, MockOutlineTool
    ):
        rag, mock_ai, mock_tool_mgr, _ = self._make_rag(
            MockVectorStore, MockDocProcessor, MockAIGen,
            MockSessionMgr, MockToolMgr, MockSearchTool, MockOutlineTool
        )
        rag.query("question")
        call_kwargs = mock_ai.generate_response.call_args[1]
        self.assertTrue(len(call_kwargs["tools"]) > 0)
        self.assertIs(call_kwargs["tool_manager"], mock_tool_mgr)

    def test_query_resets_sources_after_retrieval(
        self, MockVectorStore, MockDocProcessor, MockAIGen,
        MockSessionMgr, MockToolMgr, MockSearchTool, MockOutlineTool
    ):
        rag, _, mock_tool_mgr, _ = self._make_rag(
            MockVectorStore, MockDocProcessor, MockAIGen,
            MockSessionMgr, MockToolMgr, MockSearchTool, MockOutlineTool
        )
        rag.query("question")
        mock_tool_mgr.reset_sources.assert_called_once()

    def test_query_with_session_updates_history(
        self, MockVectorStore, MockDocProcessor, MockAIGen,
        MockSessionMgr, MockToolMgr, MockSearchTool, MockOutlineTool
    ):
        rag, _, _, mock_session = self._make_rag(
            MockVectorStore, MockDocProcessor, MockAIGen,
            MockSessionMgr, MockToolMgr, MockSearchTool, MockOutlineTool
        )
        rag.query("question", session_id="session_1")
        mock_session.get_conversation_history.assert_called_once_with("session_1")
        mock_session.add_exchange.assert_called_once_with("session_1", "question", "AI answer")

    def test_query_without_session_no_history_lookup(
        self, MockVectorStore, MockDocProcessor, MockAIGen,
        MockSessionMgr, MockToolMgr, MockSearchTool, MockOutlineTool
    ):
        rag, _, _, mock_session = self._make_rag(
            MockVectorStore, MockDocProcessor, MockAIGen,
            MockSessionMgr, MockToolMgr, MockSearchTool, MockOutlineTool
        )
        rag.query("question")
        mock_session.get_conversation_history.assert_not_called()

    def test_query_sources_empty_when_no_tool_used(
        self, MockVectorStore, MockDocProcessor, MockAIGen,
        MockSessionMgr, MockToolMgr, MockSearchTool, MockOutlineTool
    ):
        rag, mock_ai, mock_tool_mgr, _ = self._make_rag(
            MockVectorStore, MockDocProcessor, MockAIGen,
            MockSessionMgr, MockToolMgr, MockSearchTool, MockOutlineTool
        )
        mock_tool_mgr.get_last_sources.return_value = []
        _, sources = rag.query("question")
        self.assertEqual(sources, [])


if __name__ == "__main__":
    unittest.main()
