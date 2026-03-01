import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator


def _make_text_response(text="hello", stop_reason="end_turn"):
    response = MagicMock()
    response.stop_reason = stop_reason
    block = MagicMock()
    block.type = "text"
    block.text = text
    response.content = [block]
    return response


def _make_tool_use_response(
    tool_id="tu_1", tool_name="search_course_content", tool_input=None
):
    if tool_input is None:
        tool_input = {"query": "RAG"}
    response = MagicMock()
    response.stop_reason = "tool_use"
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = tool_name
    block.input = tool_input
    response.content = [block]
    return response


@patch("anthropic.Anthropic")
class TestAIGeneratorGenerateResponse(unittest.TestCase):

    def _make_generator(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        gen = AIGenerator(api_key="fake-key", model="claude-test-model")
        return gen, mock_client

    def test_generate_response_end_turn_returns_text(self, mock_anthropic_cls):
        gen, mock_client = self._make_generator(mock_anthropic_cls)
        mock_client.messages.create.return_value = _make_text_response("hello")

        result = gen.generate_response("what is X")

        self.assertEqual(result, "hello")
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "claude-test-model")
        self.assertIn("messages", call_kwargs)
        self.assertIn("system", call_kwargs)
        self.assertNotIn("tools", call_kwargs)

    def test_generate_response_with_tools_sets_tool_choice(self, mock_anthropic_cls):
        gen, mock_client = self._make_generator(mock_anthropic_cls)
        mock_client.messages.create.return_value = _make_text_response("answer")

        tools = [
            {
                "name": "search_course_content",
                "description": "search",
                "input_schema": {},
            }
        ]
        gen.generate_response("question", tools=tools)

        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertIn("tools", call_kwargs)
        self.assertEqual(call_kwargs["tool_choice"], {"type": "auto"})

    def test_generate_response_tool_use_triggers_execution(self, mock_anthropic_cls):
        gen, mock_client = self._make_generator(mock_anthropic_cls)

        first_response = _make_tool_use_response(
            tool_id="tu_1",
            tool_name="search_course_content",
            tool_input={"query": "RAG"},
        )
        second_response = _make_text_response("Here is the answer")
        mock_client.messages.create.side_effect = [first_response, second_response]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "[Course A]\nsome content"

        tools = [
            {
                "name": "search_course_content",
                "description": "search",
                "input_schema": {},
            }
        ]
        result = gen.generate_response(
            "question", tools=tools, tool_manager=tool_manager
        )

        # execute_tool called with tool name and input kwargs
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="RAG"
        )

        # Second call (round 2 inside the loop) should still have tools
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        self.assertIn("tools", second_call_kwargs)
        self.assertEqual(second_call_kwargs["tool_choice"], {"type": "auto"})

        # Second call messages should include tool_result block
        second_messages = second_call_kwargs["messages"]
        tool_result_msg = second_messages[-1]
        self.assertEqual(tool_result_msg["role"], "user")
        tool_result_block = tool_result_msg["content"][0]
        self.assertEqual(tool_result_block["type"], "tool_result")
        self.assertEqual(tool_result_block["tool_use_id"], "tu_1")
        self.assertEqual(tool_result_block["content"], "[Course A]\nsome content")

        self.assertEqual(result, "Here is the answer")

    def test_generate_response_with_conversation_history(self, mock_anthropic_cls):
        gen, mock_client = self._make_generator(mock_anthropic_cls)
        mock_client.messages.create.return_value = _make_text_response("answer")

        gen.generate_response(
            "question", conversation_history="User: hi\nAssistant: hello"
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        system = call_kwargs["system"]
        self.assertIn(AIGenerator.SYSTEM_PROMPT, system)
        self.assertIn("User: hi\nAssistant: hello", system)

    def test_generate_response_no_history_uses_base_prompt_only(
        self, mock_anthropic_cls
    ):
        gen, mock_client = self._make_generator(mock_anthropic_cls)
        mock_client.messages.create.return_value = _make_text_response("answer")

        gen.generate_response("question")

        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["system"], AIGenerator.SYSTEM_PROMPT)

    def test_two_sequential_tool_rounds(self, mock_anthropic_cls):
        gen, mock_client = self._make_generator(mock_anthropic_cls)

        tool_use_response_1 = _make_tool_use_response(
            tool_id="tu_1",
            tool_name="get_course_outline",
            tool_input={"course_name": "Course X"},
        )
        tool_use_response_2 = _make_tool_use_response(
            tool_id="tu_2",
            tool_name="search_course_content",
            tool_input={"query": "lesson topic"},
        )
        text_response = _make_text_response("Final synthesized answer")
        mock_client.messages.create.side_effect = [
            tool_use_response_1,
            tool_use_response_2,
            text_response,
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["outline result", "search result"]

        tools = [
            {
                "name": "search_course_content",
                "description": "search",
                "input_schema": {},
            }
        ]
        result = gen.generate_response(
            "find related courses", tools=tools, tool_manager=tool_manager
        )

        self.assertEqual(mock_client.messages.create.call_count, 3)

        call1_kwargs = mock_client.messages.create.call_args_list[0][1]
        self.assertIn("tools", call1_kwargs)

        call2_kwargs = mock_client.messages.create.call_args_list[1][1]
        self.assertIn("tools", call2_kwargs)

        call3_kwargs = mock_client.messages.create.call_args_list[2][1]
        self.assertNotIn("tools", call3_kwargs)

        self.assertEqual(tool_manager.execute_tool.call_count, 2)
        self.assertEqual(result, "Final synthesized answer")

    def test_tool_execution_error_makes_synthesis_call(self, mock_anthropic_cls):
        gen, mock_client = self._make_generator(mock_anthropic_cls)

        tool_use_response = _make_tool_use_response(
            tool_id="tu_1",
            tool_name="search_course_content",
            tool_input={"query": "RAG"},
        )
        text_response = _make_text_response("Sorry, could not retrieve results")
        mock_client.messages.create.side_effect = [tool_use_response, text_response]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = Exception("DB unavailable")

        tools = [
            {
                "name": "search_course_content",
                "description": "search",
                "input_schema": {},
            }
        ]
        result = gen.generate_response(
            "question", tools=tools, tool_manager=tool_manager
        )

        self.assertEqual(mock_client.messages.create.call_count, 2)

        call2_kwargs = mock_client.messages.create.call_args_list[1][1]
        self.assertNotIn("tools", call2_kwargs)

        # Check that the tool_result content contains the error message
        messages = call2_kwargs["messages"]
        tool_result_msg = messages[-1]
        self.assertEqual(tool_result_msg["role"], "user")
        tool_result_block = tool_result_msg["content"][0]
        self.assertIn("Tool execution error", tool_result_block["content"])

        self.assertEqual(result, "Sorry, could not retrieve results")


if __name__ == "__main__":
    unittest.main()
