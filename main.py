import sys
import uvicorn
from src.config import get_settings
from src.ingestion import main as run_ingestion

if __name__ == "__main__":
    settings = get_settings()

    if "--ingest" in sys.argv:
        print("Starting document ingestion...")
        run_ingestion([])
        print("Ingestion completed successfully.")
        raise SystemExit(0)

    print("Starting FastAPI server...")
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )