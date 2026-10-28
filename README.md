# ResearchOS

ResearchOS is a local-network, chat-first research assistant built as a Streamlit application. It supports self-registration, private account conversations and knowledge, concise answers, deeper cited research when requested or warranted, and an agent-selected statistical simulation tool.

## Requirements

- Python 3.11 or newer
- HTTPS reverse proxy in front of Streamlit before LAN users can register
- Groq and/or Gemini API credentials for generation

The project pins ChromaDB 0.6.3 and ONNX Runtime 1.20.1 because the newer Rust-backed Chroma build and later ONNX Runtime builds crashed in this Windows environment. ChromaDB 0.6.3 is also usable on Linux with Python 3.9+, but keep the same Chroma version when moving a persistent vector store and test any upgrade against a backup.

## Setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### Windows CMD

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
copy .env.example .env
```

### macOS or Linux

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Add `GROQ_API_KEY` and/or `GEMINI_API_KEY` to `.env` for the common server key. On the main screen, signed-in users choose whether to use that key or enter a personal provider key. Personal keys are stored encrypted, scoped to the account and provider, and never shown again. The encryption key is stored beside the accounts database, so protect the server's `data/` directory and backups. This version does not enforce per-account usage quotas.

## HTTPS and accounts

ResearchOS uses local username/password accounts with Argon2 password hashes. The account database is SQLite. Usernames and passwords are not sent to a third-party identity service.

Run Streamlit on loopback when the reverse proxy is on the same computer:

```powershell
streamlit run app.py --server.address 127.0.0.1
```

Configure the existing reverse proxy to terminate HTTPS and forward to `127.0.0.1:8501`. Do not expose Streamlit’s plain HTTP port directly to the LAN. If the proxy runs on another host, bind only to the necessary private interface and restrict the Streamlit port with the host firewall so clients cannot bypass the proxy.

Anyone who can reach the registration page can create an account. Research memory, research history, uploads, and ChromaDB files are kept under separate account UUID directories. The application enforces this logical separation; the host administrator can access the server’s files.

## Run

```powershell
streamlit run app.py --server.address 127.0.0.1
```

Open the HTTPS URL configured at the reverse proxy. Select a configured provider, refresh its model list, select a model, and use **Test selected model** before starting research.

## Features

- **Chat:** conversations are saved per account. Simple questions get concise answers; ask for a report or prefix a prompt with `/research` to start the full research workflow.
- **Account memory:** recent relevant solved questions and the current conversation are compacted into context. Unfinished requests remain available in chat history and can be continued after a restart.
- **Knowledge library:** ingests PDF, TXT, and Markdown from the chat sidebar; matching chunks can inform answers without a separate memory or knowledge page.
- **Statistical simulation tool:** ask a question that needs Monte Carlo, binomial, or bootstrap modeling. The agent proposes a method and bounded parameters in the chat, then waits for approval before running one of the app's fixed NumPy algorithms. It never executes model-generated code. Results are summarized in the conversation.
- **Guardrails and logs:** chat and simulation requests have input and request-rate limits. Model prompts have size and output-token limits and carry shared safety instructions. Retrieved pages, uploaded files, and memory are treated as untrusted evidence. Structured logs are written to `data/logs/researchos.log` with rotation; prompts, API keys, and model output are not logged.
- **Provider reliability:** transient provider requests retry up to eight times after the initial attempt, with exponential delays of 1, 2, 4, 8, 16, 32, 64, and 128 seconds. Rate limits and server/network failures retry; permanent client errors such as invalid models, unauthorized keys, and 404 responses return immediately.
- Research defaults to two agents, at most three, and one review round. Research context is compacted before each model call; each task reads at most two full web pages, uses two document chunks, and receives at most three relevant memories.

The default local embedding model is `BAAI/bge-small-en-v1.5`. Its first use downloads the model if it is not already cached.

## Data locations

```text
data/
├── accounts.sqlite3
├── .memory-signing-key
├── logs/researchos.log
└── users/<account-uuid>/
    ├── conversations/conversations.json
    ├── memory/research_memory.pkl
    ├── chroma_db/
    ├── research_history/history.json
```

The application creates these directories as needed. Memory pickle files are authenticated before deserialization. Corrupt memory or history files are preserved under a quarantine name instead of silently discarded. Back up `data/` securely; it contains user research.

## Tests

```powershell
python -m pytest -q
python -m pip check
```

Unit tests use deterministic fake provider and search clients. They cover account isolation, memory persistence and recovery, RAG persistence, source metadata and URL checks, bounded agent orchestration, citation filtering, adversarial review, and statistical simulation limits and calculations.

Live Groq/Gemini model listing and generation require valid credentials and network access. Live DDGS search requires outbound network access. Statistical simulation runs locally using validated parameters and NumPy; no container service is required.

## Troubleshooting

- **No compatible models:** set at least one provider key in `.env`, restart Streamlit, and refresh the model list.
- **Model list fails:** check the credential, provider quota, and server outbound network access. A listing failure is shown separately from an empty compatible model list.
- **First document ingestion fails:** ensure the server can download the configured local embedding model, or preload that model in the FastEmbed cache.
- **ChromaDB upgrade:** stop ResearchOS, back up the account data, and validate the upgrade on a copy of the vector store before moving the live data.

## Implementation status

All application components are implemented in staged modules. Test output and live integration limitations are reported separately; passing mocked tests do not imply that external providers were reached.
