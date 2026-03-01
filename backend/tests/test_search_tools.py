import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch

# Allow imports from the backend directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool


class TestCourseSearchToolExecute(unittest.TestCase):

    def _make_store(self):
        store = MagicMock()
        return store

    def test_execute_returns_formatted_results(self):
        store = self._make_store()
        store.search.return_value = SearchResults(
            documents=["RAG stands for Retrieval-Augmented Generation."],
            metadata=[{"course_title": "AI Course", "lesson_number": 1}],
            distances=[0.1],
        )
        store.get_lesson_link.return_value = "https://example.com/lesson1"

        tool = CourseSearchTool(store)
        result = tool.execute(query="what is RAG")

        self.assertIn("[AI Course - Lesson 1]", result)
        self.assertIn("RAG stands for Retrieval-Augmented Generation.", result)
        self.assertEqual(len(tool.last_sources), 1)
        self.assertIn("<a href=", tool.last_sources[0])
        self.assertIn("AI Course - Lesson 1", tool.last_sources[0])

    def test_execute_returns_error_when_search_errors(self):
        store = self._make_store()
        store.search.return_value = SearchResults.empty("Search error: boom")

        tool = CourseSearchTool(store)
        result = tool.execute(query="anything")

        self.assertEqual(result, "Search error: boom")

    def test_execute_returns_no_results_message_with_course_filter(self):
        store = self._make_store()
        store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        tool = CourseSearchTool(store)
        result = tool.execute(query="what is RAG", course_name="MCP")

        self.assertEqual(result, "No relevant content found in course 'MCP'.")

    def test_execute_no_course_filter_no_results(self):
        store = self._make_store()
        store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        tool = CourseSearchTool(store)
        result = tool.execute(query="anything")

        self.assertEqual(result, "No relevant content found.")

    def test_execute_sets_last_sources_without_lesson_link(self):
        store = self._make_store()
        store.search.return_value = SearchResults(
            documents=["Some content."],
            metadata=[{"course_title": "AI Course", "lesson_number": 2}],
            distances=[0.2],
        )
        store.get_lesson_link.return_value = None

        tool = CourseSearchTool(store)
        tool.execute(query="something")

        self.assertEqual(len(tool.last_sources), 1)
        self.assertNotIn("<a ", tool.last_sources[0])
        self.assertEqual(tool.last_sources[0], "AI Course - Lesson 2")

    def test_execute_passes_all_params_to_store(self):
        store = self._make_store()
        store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        tool = CourseSearchTool(store)
        tool.execute(query="agents", course_name="MCP", lesson_number=3)

        store.search.assert_called_once_with(
            query="agents", course_name="MCP", lesson_number=3
        )


class TestCourseOutlineToolExecute(unittest.TestCase):

    def _make_store(self, resolved_title="Full Course Title", catalog_metadatas=None):
        store = MagicMock()
        store._resolve_course_name.return_value = resolved_title
        if catalog_metadatas is None:
            catalog_metadatas = [
                {
                    "title": "Full Course Title",
                    "course_link": "https://example.com",
                    "lessons_json": json.dumps(
                        [
                            {"lesson_number": 0, "lesson_title": "Intro"},
                            {"lesson_number": 1, "lesson_title": "Setup"},
                        ]
                    ),
                }
            ]
        store.course_catalog.get.return_value = {"metadatas": catalog_metadatas}
        return store

    def test_outline_returns_formatted_course(self):
        store = self._make_store()
        tool = CourseOutlineTool(store)
        result = tool.execute(course_name="Full Course Title")

        self.assertIn("Course: Full Course Title", result)
        self.assertIn("Link: https://example.com", result)
        self.assertIn("Lessons (2 total):", result)
        self.assertIn("  Lesson 0: Intro", result)
        self.assertIn("  Lesson 1: Setup", result)

    def test_outline_lessons_sorted_by_number(self):
        store = self._make_store(
            catalog_metadatas=[
                {
                    "title": "My Course",
                    "course_link": "",
                    "lessons_json": json.dumps(
                        [
                            {"lesson_number": 3, "lesson_title": "Advanced"},
                            {"lesson_number": 1, "lesson_title": "Basics"},
                            {"lesson_number": 2, "lesson_title": "Intermediate"},
                        ]
                    ),
                }
            ]
        )
        tool = CourseOutlineTool(store)
        result = tool.execute(course_name="My Course")

        lines = result.splitlines()
        lesson_lines = [
            l
            for l in lines
            if l.strip().startswith("Lesson ") and ":" in l and "total" not in l
        ]
        self.assertEqual(lesson_lines[0], "  Lesson 1: Basics")
        self.assertEqual(lesson_lines[1], "  Lesson 2: Intermediate")
        self.assertEqual(lesson_lines[2], "  Lesson 3: Advanced")

    def test_outline_missing_course_name(self):
        store = self._make_store()
        tool = CourseOutlineTool(store)
        result = tool.execute()

        self.assertEqual(result, "Error: course_name is required.")

    def test_outline_course_not_found(self):
        store = self._make_store(resolved_title=None)
        tool = CourseOutlineTool(store)
        result = tool.execute(course_name="Unknown Course")

        self.assertIn("Could not find", result)

    def test_outline_no_metadata_in_catalog(self):
        store = self._make_store(catalog_metadatas=[])
        tool = CourseOutlineTool(store)
        result = tool.execute(course_name="Full Course Title")

        self.assertEqual(result, "No metadata found for course 'Full Course Title'.")


if __name__ == "__main__":
    unittest.main()
