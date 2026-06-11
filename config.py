import os                                                    # Built-in Python module for reading environment variables
from dotenv import load_dotenv                               # Reads the .env file and loads keys into the environment

load_dotenv()                                                # Must be called before os.getenv() — reads .env from project root

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")                # Reads your OpenAI key — never hardcode keys directly in code

DATA_DIR = "data"                                            # Folder where your 5 PDF documents are stored
INDEX_DIR = "index"                                          # Folder where the FAISS vector database is saved after ingestion

EMBEDDING_MODEL = "text-embedding-3-small"                   # OpenAI model that converts text into vectors (1536 numbers per chunk)
LLM_MODEL = "gpt-4o"                                         # GPT-4o reads the retrieved chunks and writes the cited answer

CHUNK_SIZE = 512                                             # Each page is split into pieces of max 512 tokens (~400 words)
CHUNK_OVERLAP = 50                                           # Each chunk shares 50 tokens with the next — prevents cutting sentences mid-clause

TOP_K = 5                                                    # How many chunks to retrieve from FAISS per question