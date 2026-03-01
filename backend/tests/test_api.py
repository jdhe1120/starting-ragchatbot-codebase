import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional

# ---------------------------------------------------------------------------
# Inline test app — mirrors app.py without static file mounts or RAGSystem
# initialization, so it works without a real ChromaDB or frontend directory.
# ---------------------------------------------------------------------------

_rag_system = None  # replaced per-test via the fixture


def _get_rag():
    return _rag_system


test_app = FastAPI(title="Test RAG App")

test_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


@test_app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        rag = _get_rag()
        session_id = request.session_id
        if not session_id:
            session_id = rag.session_manager.create_session()
        answer, sources = rag.query(request.query, session_id)
        return QueryResponse(answer=answer, sources=sources, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@test_app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    try:
        rag = _get_rag()
        analytics = rag.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@test_app.get("/")
async def root():
    return {"message": "RAG Chatbot API"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def inject_rag(mock_rag_system):
    global _rag_system
    _rag_system = mock_rag_system
    yield
    _rag_system = None


@pytest.fixture
def client():
    return TestClient(test_app)


# ---------------------------------------------------------------------------
# Tests — POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:

    def test_query_returns_200_with_valid_payload(self, client):
        response = client.post("/api/query", json={"query": "What is RAG?"})
        assert response.status_code == 200

    def test_query_response_contains_answer_and_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("RAG is cool", ["Lesson 1"])
        response = client.post("/api/query", json={"query": "What is RAG?"})
        body = response.json()
        assert body["answer"] == "RAG is cool"
        assert body["sources"] == ["Lesson 1"]

    def test_query_creates_session_when_none_provided(self, client, mock_rag_system):
        mock_rag_system.session_manager.create_session.return_value = "new-session"
        response = client.post("/api/query", json={"query": "Hello"})
        body = response.json()
        assert body["session_id"] == "new-session"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_uses_provided_session_id(self, client, mock_rag_system):
        response = client.post(
            "/api/query", json={"query": "Hello", "session_id": "existing-session"}
        )
        body = response.json()
        assert body["session_id"] == "existing-session"
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_query_passes_correct_query_to_rag(self, client, mock_rag_system):
        client.post("/api/query", json={"query": "my question", "session_id": "s1"})
        mock_rag_system.query.assert_called_once_with("my question", "s1")

    def test_query_returns_500_on_rag_error(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("something broke")
        response = client.post("/api/query", json={"query": "boom"})
        assert response.status_code == 500
        assert "something broke" in response.json()["detail"]

    def test_query_missing_query_field_returns_422(self, client):
        response = client.post("/api/query", json={})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests — GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:

    def test_courses_returns_200(self, client):
        response = client.get("/api/courses")
        assert response.status_code == 200

    def test_courses_returns_total_count(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["A", "B", "C"],
        }
        response = client.get("/api/courses")
        body = response.json()
        assert body["total_courses"] == 3

    def test_courses_returns_titles_list(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Course One", "Course Two"],
        }
        response = client.get("/api/courses")
        body = response.json()
        assert body["course_titles"] == ["Course One", "Course Two"]

    def test_courses_returns_500_on_analytics_error(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("db error")
        response = client.get("/api/courses")
        assert response.status_code == 500
        assert "db error" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Tests — GET /
# ---------------------------------------------------------------------------

class TestRootEndpoint:

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
