

Current behavior:
- Claude makes 1 tool call --> tools are removed from API params --> final response
- if Claude wants anotehr tool call after seeing resluts, it can't (Gets empty response)

Desired behavior:
- Each tool call should be a separate API request where Claude can reason about previous results
- Support for. complex queries requiring multiple searches for compariosns, multi-part questions, or when informatino from different coures/lessons is needed

Example flow:
1. User: search for a course that discusses the same topic as lesson 4 of coures X
2. Claude: get coures outline for coures X --> gets title of lesson 4
3. Cladue: uses the title to search for a coures that discusses the same topic --> returns course informatino
4. Claude: provide complete answer

Requirements:
- Maximum two sequential rounds per user query.
- Terminate when:
- A, two rounds completed.
- B, Claude's response has no tool use blocks.
- C, tool call fails.
- Preserve conversation context between rounds.
- Handle tool execution errors gracefully.

Notes:
- Update the system prompt in backend/ai_generator.py.
- Update the test in backend/tests/test_ai_generator.py.
- Write tests that verify the external behavior (API calls made, tools executed, results returned), rather than internal state details.Refactor @backend/ai_generator.py to support sequential tool calling where Claude can make up to 2 tool calls in separate API rounds

Use 2 parallel subagents to brainstorm possible plans. Do not implement any code.

