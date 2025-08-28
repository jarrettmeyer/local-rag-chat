-- Uncomment and use with caution! These will drop all tables, resetting the database.
DROP TABLE IF EXISTS permissions;
DROP TABLE IF EXISTS embeddings;
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS docs;

CREATE EXTENSION IF NOT EXISTS vector;


CREATE TABLE IF NOT EXISTS docs (
	doc_id UUID PRIMARY KEY,
	file_name TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS chunks (
	chunk_id UUID PRIMARY KEY,
	doc_id UUID REFERENCES docs(doc_id) ON DELETE CASCADE,
	content TEXT NOT NULL,
	page_number INTEGER NOT NULL,
	chunk_number INTEGER NOT NULL,
	metadata JSONB DEFAULT '{}'
);


CREATE TABLE IF NOT EXISTS embeddings (
	embedding_id UUID PRIMARY KEY,
	chunk_id UUID REFERENCES chunks(chunk_id) ON DELETE CASCADE,
	vector VECTOR(768)
);


CREATE TABLE IF NOT EXISTS permissions (
	user_id TEXT NOT NULL,
	doc_id UUID NOT NULL REFERENCES docs(doc_id) ON DELETE CASCADE,
	PRIMARY KEY (user_id, doc_id)
);
