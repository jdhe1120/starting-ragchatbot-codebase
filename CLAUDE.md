# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Quick start (from project root)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

App available at http://localhost:8000. Requires `ANTHROPIC_API_KEY` in a `.env` file at the project root.

## Dependencies

```bash
uv sync          # Install all dependencies
```

Always use `uv` to run Python commands (e.g. `uv run`, `uv sync`). Never use `pip` directly.

There are no tests or linting scripts configured in this project.

## Architecture

This is a full-stack RAG chatbot with a FastAPI backend and vanilla JS frontend.

**Request flow for a user query:**

1. `frontend/script.js` — `sendMessage()` POSTs `{query, session_id}` to `/api/query`
2. `backend/app.py` — `query_documents()` creates a session if needed, delegates to `RAGSystem`
3. `backend/rag_system.py` — `RAGSystem.query()` fetches conversation history, then calls `AIGenerator`
4. `backend/ai_generator.py` — makes **Claude API call #1** with the `search_course_content` tool available (`tool_choice: auto`)
   - If `stop_reason == "end_turn"`: returns direct answer (no search needed)
   - If `stop_reason == "tool_use"`: executes the tool, then makes **Claude API call #2** to synthesize results
5. `backend/search_tools.py` — `CourseSearchTool.execute()` calls `VectorStore.search()` and tracks sources
6. `backend/vector_store.py` — resolves fuzzy course name via semantic search on `course_catalog`, then queries `course_content` with optional `course_title`/`lesson_number` filters
7. Sources and response returned to frontend; session history updated in `SessionManager`

**Key architectural decisions:**
- **Tool-based RAG**: Claude decides whether to search (not every query triggers retrieval). Max one search per query, enforced via the system prompt.
- **Dual ChromaDB collections**: `course_catalog` stores course metadata for fuzzy name resolution; `course_content` stores chunked lesson text for semantic search.
- **Session history is injected into the system prompt** as formatted text (not as separate messages), capped at the last 2 exchanges.
- The backend server is started from the `backend/` directory, so all relative paths (e.g. `../docs`, `./chroma_db`) are relative to `backend/`.

## Document Format

Course documents in `docs/` must follow this format for correct parsing:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 1: <lesson title>
...
```

`DocumentProcessor` parses this structure; deviations from it will cause content to be treated as unstructured text.

## Configuration

All tunable parameters are in `backend/config.py`:

| Setting | Default | Description |
|---|---|---|
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Claude model used |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `MAX_RESULTS` | 5 | Max search results returned |
| `MAX_HISTORY` | 2 | Conversation turns remembered |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB persistence path (relative to `backend/`) |

## Adding New Tools

Tools follow the `Tool` abstract base class in `backend/search_tools.py`. Implement `get_tool_definition()` (returns an Anthropic tool schema dict) and `execute(**kwargs)`. Register with `tool_manager.register_tool(your_tool)` in `RAGSystem.__init__`.
