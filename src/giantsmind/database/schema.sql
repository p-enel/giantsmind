-- Authors table
CREATE TABLE authors (
    author_id INTEGER PRIMARY KEY,
    name TEXT
);

-- Journals table
CREATE TABLE journals (
    journal_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

-- Papers table
CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY,
    journal_id TEXT,
    file_path TEXT,
    publication_date DATE,
    title TEXT NOT NULL,
    url TEXT,
    FOREIGN KEY (journal_id) REFERENCES journals(journal_id)
);

-- Collections table
CREATE TABLE collections (
    collection_id INTEGER PRIMARY KEY,
    name TEXT
);

-- ChunkIDs table
CREATE TABLE chunk_ids (
    chunk_id TEXT PRIMARY KEY,
    paper_id TEXT,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

-- Author-Paper association table
CREATE TABLE author_paper (
    author_id INTEGER,
    paper_id TEXT,
    PRIMARY KEY (author_id, paper_id),
    FOREIGN KEY (author_id) REFERENCES authors(author_id),
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

-- Paper-Collection association table
CREATE TABLE paper_collection (
    paper_id TEXT,
    collection_id INTEGER,
    PRIMARY KEY (paper_id, collection_id),
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id),
    FOREIGN KEY (collection_id) REFERENCES collections(collection_id)
);
