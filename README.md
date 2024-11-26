# GiantsMind
GiantsMind is a Python package that provides tools for interacting with scientific article PDFs. It allows you to parse, extract metadata, search, and query scientific papers using natural language. In its current form, it uses [LlamaIndex](https://www.llamaindex.ai) to parse PDFs and [Claude Sonnet 3.5](https://www.anthropic.com/news/claude-3-5-sonnet) for various agents and natural language interaction. You will need an API key for these services to run the two commands (see [Installation](#installation))

This is an early version with limited functionalities in active development.

## Features

- PDF document parsing using Llamaparse
- Metadata extraction from PDF papers (DOI, arXiv ID)
- Automatic metadata fetching from CrossRef and arXiv APIs  
- Vector database storage for semantic search
- SQLite database for metadata management
- Natural language querying using Claude AI
- Interactive CLI interface

## Installation

1. Clone the repository
2. Install dependencies:

```sh
pip install -e ".[dev]"
```

3. Set up environment variables in `.env`:

```sh
LLAMA_API_KEY=<your-llama-api-key>
ANTHROPIC_API_KEY=<your-anthropic-api-key>
DEFAULT_PDF_PATH=<path-to-pdf-folder>
```

## Usage

### Parse PDF Papers

```sh
giantsmind --parse /path/to/papers
```

This will:
- Parse PDFs using Llamaparse
- Extract and fetch metadata
- Store content in vector database
- Save metadata in SQLite database

### Interactive Query Mode

```sh
giantsmind
```

This starts an interactive session where you can:
- Ask questions about papers in natural language
- Search by metadata (authors, dates, journals)
- Search paper content semantically
- Get AI-generated answers with citations

## Development

Requirements:
- Python 3.12+
- Development dependencies (`pip install -e ".[dev]"`)

Run tests:
```sh
pytest
```

## License

BSD 3-Clause License. See [LICENSE.txt](LICENSE.txt) for details.

## Author

Pierre Enel ([pierre.enel@gmail.com])
