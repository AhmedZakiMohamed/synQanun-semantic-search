# SynQanun - Legal Semantic Search API

A semantic search API over Egyptian laws and international conventions, built with FastAPI, LangChain, ChromaDB, and `intfloat/multilingual-e5-base`. It matches queries by meaning rather than keywords and returns results grouped by document, with Docker and local setup options.

----------

## Semantic Search Flow

The system operates in two main phases:

### 1. Data Ingestion & Indexing
1. **Load & Clean:** Reads `.txt` files from `data/` and cleans the text.
2. **Article-Level Split:** Detects legal article headers (e.g., `المادة 1`) to group text by article.
3. **Sub-chunking:** If an article exceeds `MAX_CHUNK_CHARS`, it is split into smaller chunks with overlap to prevent context loss.
4. **Vector Embedding:** Converts chunks into vector embeddings using the E5 model and saves them in ChromaDB.

### 2. Search & Aggregation
1. **Query Processing:** Cleans the user's search query, adds the E5 `query:` prefix, and generates its embedding.
2. **Chunk Search:** Finds the most relevant text chunks from ChromaDB.
3. **Document Aggregation** Groups matched chunks by article, then groups articles by source document. Each document is scored by its best-matching article (max, so long documents are not favoured for having more chunks), the top-K documents are returned, and each keeps its best articles with the full article text reconstructed.

----------

## Chunking Strategy & Trade-offs

We use an **Article-Aware Chunking** approach to keep legal contexts meaningful:

1. **Article-First Split:** Documents are first split by legal article headers (e.g., `المادة 1`). The header regex requires strict trailing line breaks/punctuation to avoid splitting on mid-text references (e.g., "...وفقاً للمادة 5 من القانون...").
   - *Trade-off:* Preserves exact legal structure and citation accuracy, but relies on clean header patterns.
2. **Recursive Sub-chunking:** `RecursiveCharacterTextSplitter` is only applied if an article exceeds the maximum chunk size, and sub-chunks maintain a slight text overlap.
   - *Trade-off:* Keeps short and medium articles intact and prevents context loss across boundaries at the cost of slight storage overhead, though extremely long paragraphs may split mid-sentence.

---------

## Vector Storage & Retrieval

- **Vector Store:** ChromaDB with Cosine Similarity.
- **Idempotent Ingestion:** Each chunk receives a deterministic ID. Re-ingesting a document automatically purges its old chunks by `source_doc` before inserting new ones.
- **Candidate Pool:** Fetches `top_k × CANDIDATE_MULTIPLIER` candidate chunks to guarantee enough distinct legal articles after chunk-to-article aggregation.

---------

## Scalability & Production Readiness

The API is architected for zero-friction production scaling:
1. **Vector Store:** Easy migration from ChromaDB to **Qdrant** / **pgvector** for distributed, multi-node setups.
2. **Decoupled Pipeline:** Isolation of background ingestion tasks from live search queries.
3. **Model Serving:** Offloading inference to dedicated frameworks (**Triton** / **TEI**) for isolated GPU auto-scaling.
4. **Caching:** **Redis** integration for sub-millisecond query result caching.

----------

## System Limitations

- **Cold Start & CPU:** Initial startup downloads the embedding model (~1.1 GB). CPU execution adds slight inference latency.
- **Text-Only Input:** Scanned PDFs without an embedded text layer require an external OCR pre-processing step.
- **Pure Semantic Search:** Relies strictly on dense embeddings without hybrid keyword (BM25) re-ranking.

-----------

## Running the Application

### Option 1: Running with Docker (Recommended)

1. **Start the container**

```bash
docker compose up -d
```

**Note:** The first launch downloads embedding model weights (~1.1 GB). Subsequent runs start faster.

2. **Populate the vector store (ingestion)**

```bash
docker compose exec semantic-search-api python -m src.ingestion
```

### Option 2: Local Setup (Python Virtual Environment)

1. **Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies (CPU-optimized PyTorch)**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

3. **Run ingestion**

```bash
python -m src.ingestion
```

4. **Run the API**

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

-----------

## Postman Collection & API Docs

Import `synQanun.postman_collection.json` in Postman (**Import** → select the file), or use Postman UI at `http://localhost:8000/health or search`.

Endpoints:

- `GET /health` — service status and indexed chunk count
- `POST /search` — semantic search

-------

## Search Endpoint (`POST /search`)

Request body accepts **`query` or `q`**, and **`top_k` or `topK`**.

#### Request body example

```json
{
  "q": "ما هي الأحكام المتعلقة بتزوير البيانات الإلكترونية وإنتاج بيانات غير أصلية لاستخدامها لأغراض قانونية؟",
  "topK": 3
}
```

#### Response example (200 OK)

```json
{
  "query": "ما هي الأحكام المتعلقة بتزوير البيانات الإلكترونية وإنتاج بيانات غير أصلية لاستخدامها لأغراض قانونية؟",
  "top_k": 3,
  "total": 2,
  "results": [
    {
      "source_doc": "United Nations Convention against Cybercrime.txt",
      "doc_type": "international_convention",
      "score": 0.8652,
      "matched_chunks": 6,
      "matched_articles": [
        {
          "article_number": "12",
          "article_label": "المادة 12",
          "section_index": 12,
          "score": 0.8652,
          "text": "المادة 12: التزوير المتعلق بنظام تكنولوجيا معلومات واتصالات\n1 - تعتمد كل دولة طرف ما قد يلزم من تدابير تشريعية...",
          "matched_chunks": 1,
          "chunk_seqs": [15]
        },
        {
          "article_number": "9",
          "article_label": "المادة 9",
          "section_index": 9,
          "score": 0.8414,
          "text": "المادة 9: التدخل في البيانات الإلكترونية\n1 - تعتمد كل دولة طرف ما قد يلزم من تدابير تشريعية...",
          "matched_chunks": 1,
          "chunk_seqs": [11]
        },
        {
          "article_number": "8",
          "article_label": "المادة 8",
          "section_index": 8,
          "score": 0.8326,
          "text": "المادة 8: الاعتراض غير المشروع\n1 - تعتمد كل دولة طرف ما قد يلزم من تدابير تشريعية...",
          "matched_chunks": 1,
          "chunk_seqs": [10]
        }
      ]
    },
    {
      "source_doc": "law_175_2018.txt",
      "doc_type": "egyptian_law",
      "score": 0.838,
      "matched_chunks": 3,
      "matched_articles": [
        {
          "article_number": "6",
          "article_label": "المادة 6",
          "section_index": 5,
          "score": 0.838,
          "text": "المادة 6\n\nقابلة\nلمأمورى الضبط القضائى المختصين ، لمدة لا تزيد على ثلاثين يوما...",
          "matched_chunks": 1,
          "chunk_seqs": [10]
        },
        {
          "article_number": "20",
          "article_label": "المادة 20",
          "section_index": 19,
          "score": 0.8215,
          "text": "المادة 20\nيعاقب بالحبس مدة لا تقل عن سنتين ، وبغرامة لا تقل عن خمسين ألف جنيه...",
          "matched_chunks": 1,
          "chunk_seqs": [26]
        },
        {
          "article_number": "2",
          "article_label": "المادة 2",
          "section_index": 1,
          "score": 0.8181,
          "text": "المادة 2\nأولا- مع عدم الإخلال بالأحكام الواردة بهذا القانون...",
          "matched_chunks": 1,
          "chunk_seqs": [5, 6]
        }
      ]
    }
  ]
}
```