import os
import json
import logging
from src.ingestion.parser import PDFParser
from src.ingestion.splitter import MarkdownSplitter
from src.ingestion.indexer import HybridIndexer
from langchain_core.documents import Document

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = os.path.join(os.getcwd(), "data", "raw_pdfs")
PROCESSED_DIR = os.path.join(os.getcwd(), "data", "processed")

def ingest_pdfs():
    """
    Process all PDFs in the raw_pdfs directory.
    """
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
        
    parser = PDFParser()
    splitter = MarkdownSplitter()
    # Initialize Hybrid Indexer
    indexer = HybridIndexer()

    sources = []
    for filename in os.listdir(RAW_DIR):
        lower = filename.lower()
        if lower.endswith(".pdf"):
            pdf_path = os.path.join(RAW_DIR, filename)
            paper_id = os.path.splitext(filename)[0].replace(" ", "_")
            sources.append((pdf_path, paper_id, filename))
            continue

        if lower.endswith((".txt", ".urls")):
            list_path = os.path.join(RAW_DIR, filename)
            try:
                with open(list_path, "r", encoding="utf-8") as f:
                    for line in f:
                        url = line.strip()
                        if not url or url.startswith("#"):
                            continue
                        if not url.startswith(("http://", "https://")):
                            continue
                        base = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
                        paper_id = (os.path.splitext(base)[0] or base or "paper").replace(" ", "_")
                        sources.append((url, paper_id, url))
            except Exception as e:
                logger.error(f"Failed to read URL list {filename}: {e}")

    for source, paper_id, title in sources:
        paper_dir = os.path.join(PROCESSED_DIR, paper_id)
        if os.path.exists(paper_dir):
            logger.info(f"Skipping {title}, already processed.")
            continue

        logger.info(f"Processing {title}...")

        try:
            markdown_content = parser.parse(source)
            sections = splitter.split(markdown_content)

            os.makedirs(paper_dir, exist_ok=True)
            sections_dir = os.path.join(paper_dir, "sections")
            os.makedirs(sections_dir, exist_ok=True)

            index_data = {
                "paper_id": paper_id,
                "title": title,
                "sections": [],
            }

            documents_to_index = []

            for i, section in enumerate(sections):
                section_filename = f"section_{i:03d}.md"
                section_path = os.path.join(sections_dir, section_filename)

                with open(section_path, "w", encoding="utf-8") as f:
                    f.write(f"# {section['title']}\n\n{section['content']}")

                index_data["sections"].append(
                    {"title": section["title"], "filename": section_filename, "level": section["level"]}
                )

                # Create Document for indexing
                doc_metadata = {
                    "paper_id": paper_id,
                    "title": title,
                    "section_title": section["title"],
                    "source": source
                }
                documents_to_index.append(
                    Document(page_content=section["content"], metadata=doc_metadata)
                )

            # Add to Hybrid Index
            indexer.add_documents(documents_to_index)

            with open(os.path.join(paper_dir, "index.json"), "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully processed {title}.")

        except Exception as e:
            logger.error(f"Failed to process {title}: {e}")

if __name__ == "__main__":
    ingest_pdfs()
