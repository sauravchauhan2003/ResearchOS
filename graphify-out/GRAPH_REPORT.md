# Graph Report - ResearchOS  (2026-10-27)

## Corpus Check
- 75 files · ~22,305 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 3 file(s) not represented in the graph (top: .example 1, (none) 1, .toml 1)

## Summary
- 585 nodes · 1747 edges · 28 communities (16 shown, 12 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 187 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- orchestrator.py
- GenerationResponse
- retriever.py
- WebpageReader
- UserStore
- MemoryManager
- chat_view.py
- test_routing.py
- ResearchOS
- ModelProvider
- WebSearchTool
- Q: understand the entire codebase using the graphify skill
- ToolRegistry
- webpage_reader.py
- ResearchHistoryStore
- chat_agent.py
- statistical_simulation.py

## God Nodes (most connected - your core abstractions)
1. `ResearchOrchestrator` - 33 edges
2. `_process_request()` - 33 edges
3. `EvidenceSource` - 32 edges
4. `RagPipeline` - 28 edges
5. `UserStore` - 27 edges
6. `log_event()` - 27 edges
7. `render_chat_view()` - 27 edges
8. `MemoryManager` - 26 edges
9. `WebSearchTool` - 26 edges
10. `compact_text()` - 25 edges

## Surprising Connections (you probably didn't know these)
- `BaseAgent` --uses--> `ModelProvider`  [INFERRED]
  agents/base_agent.py → providers/base.py
- `ChatAgent` --uses--> `MemoryManager`  [INFERRED]
  agents/chat_agent.py → memory/memory_manager.py
- `ChatAgent` --uses--> `RagPipeline`  [INFERRED]
  agents/chat_agent.py → rag/retriever.py
- `ChatAgent` --uses--> `WebSearchTool`  [INFERRED]
  agents/chat_agent.py → tools/web_search.py
- `ChatAgent` --uses--> `WebpageReader`  [INFERRED]
  agents/chat_agent.py → tools/webpage_reader.py

## Import Cycles
- None detected.

## Communities (28 total, 12 thin omitted)

### Community 0 - "orchestrator.py"
Cohesion: 0.09
Nodes (40): AdversarialReviewer, BaseAgent, parse_json_object(), ChatAnswer, compact_blocks(), compact_text(), EvidenceAnalyst, AgentFinding (+32 more)

### Community 1 - "GenerationResponse"
Cohesion: 0.07
Nodes (23): validate_model_response(), ModelInfo, ProviderName, ProviderResult, GeminiProvider, GroqProvider, ModelRegistry, _http_status() (+15 more)

### Community 2 - "retriever.py"
Cohesion: 0.07
Nodes (20): chunk_text(), evidence_context(), format_document_citation(), FastEmbedder, bm25_rank(), reciprocal_rank_fusion(), tokenize(), load_document() (+12 more)

### Community 3 - "WebpageReader"
Cohesion: 0.18
Nodes (9): FakeClient, FakeResponse, FakeSearch, public_resolver(), test_search_preserves_source_metadata_and_deduplicates_urls(), test_webpage_reader_blocks_private_ip_and_dns_rebinding_targets(), test_webpage_reader_extracts_text_and_date(), UnsafeURL (+1 more)

### Community 5 - "UserStore"
Cohesion: 0.10
Nodes (9): UserStore, AppSettings, _authenticate_app(), test_authenticated_app_renders_chat_first_ui(), test_repeated_new_chat_clicks_do_not_create_empty_history_entries(), test_browser_login_sessions_are_hashed_expiring_and_revocable(), test_register_authenticate_and_persisted_preferences(), test_registration_rejects_weak_duplicate_and_invalid_usernames() (+1 more)

### Community 6 - "MemoryManager"
Cohesion: 0.12
Nodes (9): MemoryManager, Claim, MemoryRecord, MemoryIntegrityError, PickleStore, manager(), test_account_memories_are_isolated(), test_corrupt_file_is_preserved_and_replaced_with_empty_schema() (+1 more)

### Community 7 - "chat_view.py"
Cohesion: 0.08
Nodes (36): _apply_cookie_action(), authentication_view(), _begin_user_session(), main(), provider_settings(), _read_login_cookie(), User, preflight_refusal() (+28 more)

### Community 8 - "test_routing.py"
Cohesion: 0.23
Nodes (10): extract_research_command(), should_create_report(), should_search_web(), should_use_sandbox(), should_use_statistical_simulation(), test_code_requests_route_to_the_sandbox_tool(), test_explicit_and_complex_requests_use_full_research(), test_statistical_modeling_routes_to_the_in_chat_simulation_tool() (+2 more)

### Community 9 - "ResearchOS"
Cohesion: 0.14
Nodes (13): Data locations, Features, HTTPS and accounts, Implementation status, macOS or Linux, Requirements, ResearchOS, Run (+5 more)

### Community 20 - "ModelProvider"
Cohesion: 0.15
Nodes (5): ChatAgent, ModelProvider, RecordingProvider, test_chat_agent_uses_compact_prior_turns_and_relevant_private_memory(), test_simple_arithmetic_skips_web_search_and_document_retrieval()

### Community 21 - "WebSearchTool"
Cohesion: 0.19
Nodes (5): test_search_failures_are_reported_without_fabricating_sources(), SearchBackend, SearchResponse, WebSearchTool, WebSource

### Community 22 - "Q: understand the entire codebase using the graphify skill"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: understand the entire codebase using the graphify skill, Source Nodes

### Community 24 - "webpage_reader.py"
Cohesion: 0.23
Nodes (4): canonical_url(), _assert_public_url(), _resolve(), WebPage

### Community 25 - "ResearchHistoryStore"
Cohesion: 0.34
Nodes (4): ResearchHistoryStore, test_history_persists_full_trace_and_can_delete_or_clear(), test_history_storage_is_scoped_by_account(), render_memory_view()

### Community 26 - "chat_agent.py"
Cohesion: 0.20
Nodes (3): _emit(), _raise_if_streamlit_control(), memory_search_query()

### Community 27 - "statistical_simulation.py"
Cohesion: 0.11
Nodes (23): GuardrailViolation, preflight_guardrail(), prepare_generation(), RequestRateLimiter, should_save_answer_to_memory(), validate_user_request(), test_generation_limits_and_shared_safety_policy(), test_preflight_blocks_secret_extraction_before_tools_but_allows_security_questions() (+15 more)

## Knowledge Gaps
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UserStore` connect `UserStore` to `user_store.py`, `chat_view.py`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `ResearchOrchestrator` (e.g. with `AdversarialReviewer` and `EvidenceAnalyst`) actually correct?**
  _`ResearchOrchestrator` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Should `orchestrator.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08634538152610442 - nodes in this community are weakly interconnected._
- **Why does `RagPipeline` connect `retriever.py` to `orchestrator.py`, `chat_view.py`, `ModelProvider`, `WebSearchTool`, `ResearchHistoryStore`, `chat_agent.py`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `_process_request()` (e.g. with `.research()` and `User`) actually correct?**
  _`_process_request()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Should `GenerationResponse` be split into smaller, more focused modules?**
  _Cohesion score 0.06682692307692308 - nodes in this community are weakly interconnected._
- **Why does `MemoryManager` connect `MemoryManager` to `orchestrator.py`, `retriever.py`, `user_store.py`, `chat_view.py`, `ModelProvider`, `WebSearchTool`, `ResearchHistoryStore`, `chat_agent.py`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._