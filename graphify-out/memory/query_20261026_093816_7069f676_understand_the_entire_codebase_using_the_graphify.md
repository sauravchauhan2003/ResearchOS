---
type: "query"
date: "2026-10-26T09:38:16.966840+00:00"
question: "understand the entire codebase using the graphify skill"
contributor: "graphify"
outcome: "useful"
source_nodes: ["ResearchOrchestrator", "RagPipeline", "UserStore", "SandboxExecutor", "WebSearchTool", "MemoryManager"]
---

# Q: understand the entire codebase using the graphify skill

## Answer

Expanded from original query via vocab: [research, agent, orchestrator, orchestration, rag, retrieval, memory, auth, tool, webpage, experiment, chat]. ResearchOS is a local-network Streamlit research assistant. app.py and UI views authenticate accounts through UserStore, then route chat and research requests. ChatAgent supports concise responses; ResearchOrchestrator coordinates bounded parallel research, analysis, synthesis, and review. WebSearchTool and WebpageReader gather cited web evidence. RagPipeline ingests and retrieves account-private document chunks using embeddings and Chroma. MemoryManager, conversations, and research history persist within account UUID directories. SandboxExecutor runs code only after per-run user approval inside a constrained Docker container. Groq and Gemini are provider integrations. Graph: 519 nodes, 1502 edges, 21 communities; some cross-module uses are inferred rather than extracted.

## Outcome

- Signal: useful

## Source Nodes

- ResearchOrchestrator
- RagPipeline
- UserStore
- SandboxExecutor
- WebSearchTool
- MemoryManager