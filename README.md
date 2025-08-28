# Local RAG Chat

This is a CLI program that implements a Retrieval-Augmented Generation (RAG) chatbot.

## Features

1. **Ingest PDF document**: Read the text from the PDF. Chunk and embed the text into vectors.
2. **List documents**: List all documents that have been ingested.
3. **Purge document**: Delete a document, along with all chunks and embeddings.
4. **Chat**: Ask a question, supported by evidence.

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Features

### `ingest`

```bash
uv run local-rag-chat ingest FILE
```

### `list`

```bash
uv run local-rag-chat list
```

### `chat`

```bash
uv run local-rag-chat chat
```

### `purge`

```bash
uv run local-rag-chat purge DOC_ID
```

