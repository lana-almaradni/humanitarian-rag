import os                                                    # For reading environment variables
from dotenv import load_dotenv                               # Loads API keys from .env file
from llama_index.core import StorageContext, load_index_from_storage, Settings  
                                                             # StorageContext: manages storage connections
                                                             # load_index_from_storage: reconstructs index from disk
                                                             # Settings: global model configuration
from llama_index.core.memory import ChatMemoryBuffer         # Stores conversation history for follow-up questions
from llama_index.embeddings.openai import OpenAIEmbedding    # OpenAI embedding model
from llama_index.llms.openai import OpenAI                   # GPT-4o language model
from llama_index.vector_stores.faiss import FaissVectorStore # Loads saved FAISS index from disk

load_dotenv()                                                # Load API keys before anything else

from config import INDEX_DIR, EMBEDDING_MODEL, LLM_MODEL, TOP_K  
                                                             # Import settings from config.py

SYSTEM_PROMPT = """You are a humanitarian policy and operations assistant.
You answer questions strictly based on the provided humanitarian documents.

Rules:
- Only answer from the context provided                      # Prevents hallucination
- Always cite the source document and page number            # Forces citations in every answer
- If the answer is not in the context, say: not covered      # Honest fallback instead of guessing
- Use precise humanitarian terminology                       # Maintains professional language
- Keep answers concise but complete                          # Balances brevity with usefulness
- When relevant, connect answers to SDGs or Sendai targets"""# Links field standards to global frameworks

COMPARE_PROMPT = """You are a humanitarian policy expert.
The user wants to COMPARE how different frameworks approach a topic.

Structure your answer like this:
**Sphere Standards Position:**                               # Section 1 — what Sphere says
[what Sphere says, with citation]

**SDG/UN Framework Position:**                              # Section 2 — what SDGs/Sendai say
[what the SDGs or Sendai Framework say, with citation]

**Key Connections:**                                         # Section 3 — synthesis
[how these frameworks complement or differ]

If only one framework covers the topic, say so clearly."""   # Honest fallback for partial coverage


def _load_index():                                           # Shared helper — underscore means internal use only
    vector_store = FaissVectorStore.from_persist_dir(INDEX_DIR)  
                                                             # Load the FAISS index we saved during ingestion
    storage_context = StorageContext.from_defaults(          # Recreate storage context pointing to saved index
        vector_store=vector_store,                           # Use the loaded FAISS vector store
        persist_dir=INDEX_DIR                                # Point to the index/ folder on disk
    )
    return load_index_from_storage(storage_context=storage_context)  
                                                             # Reconstruct the full LlamaIndex index object


def _configure_models():                                     # Shared helper — avoids repeating model setup in every function
    Settings.embed_model = OpenAIEmbedding(                  # Must use SAME model as ingestion — vectors must match
        model=EMBEDDING_MODEL,                               # text-embedding-3-small
        api_key=os.getenv("OPENAI_API_KEY")                  # Read key from environment
    )
    Settings.llm = OpenAI(                                   # GPT-4o generates the final answer
        model=LLM_MODEL,                                     # gpt-4o
        api_key=os.getenv("OPENAI_API_KEY")                  # Read key from environment
    )


def load_query_engine():                                     # Standard engine — no memory, used for compare mode
    _configure_models()                                      # Set up OpenAI models
    index = _load_index()                                    # Load FAISS index from disk
    return index.as_query_engine(                            # Create query engine — each question is independent
        similarity_top_k=TOP_K,                              # Retrieve top 5 most similar chunks per question
        response_mode="compact"                              # Combines retrieved chunks before sending to GPT-4o
    )


def load_chat_engine():                                      # Chat engine WITH conversation memory
    _configure_models()                                      # Set up OpenAI models
    index = _load_index()                                    # Load FAISS index from disk
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)  
                                                             # Remember last ~3000 tokens of conversation
    return index.as_chat_engine(                             # Chat engine — remembers previous questions
        chat_mode="condense_plus_context",                   # Condenses chat history + retrieves fresh context each turn
        memory=memory,                                       # Pass the memory buffer
        similarity_top_k=TOP_K,                              # Retrieve top 5 chunks per question
        system_prompt=SYSTEM_PROMPT                          # Humanitarian compliance instructions for every answer
    )


def compare_frameworks(query_engine, question):              # Structured comparison across humanitarian frameworks
    from llama_index.core.prompts import PromptTemplate      # Import prompt template class

    compare_template = PromptTemplate(                       # Build custom prompt for comparison answers
        COMPARE_PROMPT + """

Context from documents:
{context_str}                                                # Placeholder — filled with retrieved chunks

Question: {query_str}                                        # Placeholder — filled with user's question

Comparison:"""
    )
    query_engine.update_prompts({                            # Replace default prompt with our comparison prompt
        "response_synthesizer:text_qa_template": compare_template  
                                                             # This key targets the answer generation step
    })
    return query_engine.query(question)                      # Run the query with the new prompt


def export_to_pdf(messages):                                 # Generates downloadable PDF report of the conversation
    from fpdf import FPDF                                    # PDF generation library
    import io                                                # For in-memory file buffer

    def clean_text(text):                                    # Inner helper — converts unicode to plain ASCII
        replacements = {                                     # Map of special characters to safe equivalents
            '\u2019': "'",                                   # Right single quote → apostrophe
            '\u2018': "'",                                   # Left single quote → apostrophe
            '\u201c': '"',                                   # Left double quote → straight quote
            '\u201d': '"',                                   # Right double quote → straight quote
            '\u2014': '-',                                   # Em dash → hyphen
            '\u2013': '-',                                   # En dash → hyphen
            '\u2022': '*',                                   # Bullet → asterisk
        }
        for char, replacement in replacements.items():      # Apply all replacements
            text = text.replace(char, replacement)
        return ''.join(c if ord(c) < 128 else '?' for c in text)  
                                                             # Remove any remaining non-ASCII characters

    pdf = FPDF()                                             # Create new blank PDF document
    pdf.set_auto_page_break(auto=True, margin=15)            # Auto-add new pages when content reaches bottom
    pdf.add_page()                                           # Add the first page
    pdf.set_left_margin(15)                                  # 15mm left margin
    pdf.set_right_margin(15)                                 # 15mm right margin

    pdf.set_font("Helvetica", "B", 16)                       # Bold Helvetica 16pt for title
    pdf.set_text_color(27, 100, 60)                          # Dark green color for humanitarian theme
    pdf.cell(0, 10, "Humanitarian Policy - Q&A Report", ln=True)  
                                                             # Title text — ln=True moves to next line
    pdf.set_font("Helvetica", "", 10)                        # Regular 10pt for subtitle
    pdf.set_text_color(100, 100, 100)                        # Grey color
    pdf.cell(0, 6, "Generated by Humanitarian Policy Assistant", ln=True)  
    pdf.ln(6)                                                # Add 6mm vertical space

    for msg in messages:                                     # Loop through every message in the conversation
        if msg["role"] == "user":                            # User questions — shown in green bold
            pdf.set_font("Helvetica", "B", 11)               # Bold 11pt
            pdf.set_text_color(27, 100, 60)                  # Green color
            pdf.multi_cell(180, 7, clean_text(f"Q: {msg['content']}"))  
                                                             # 180mm width — explicit to prevent margin overflow
            pdf.ln(2)                                        # Small gap after question
        elif msg["role"] == "assistant":                     # Assistant answers — shown in black
            pdf.set_font("Helvetica", "", 10)                # Regular 10pt
            pdf.set_text_color(30, 30, 30)                   # Near-black color
            clean = clean_text(msg["content"].replace("**", "").replace("*", ""))  
                                                             # Strip markdown bold/italic markers
            pdf.multi_cell(180, 6, clean)                    # Answer text
            if "sources" in msg and msg["sources"]:          # If sources exist, add them below answer
                pdf.set_font("Helvetica", "I", 9)            # Italic 9pt for sources
                pdf.set_text_color(100, 100, 100)            # Grey color
                for s in msg["sources"]:                     # Loop through each source citation
                    pdf.multi_cell(180, 5, clean_text(f"Source: {s[:80]}"))  
                                                             # Truncate to 80 chars to prevent overflow
            pdf.ln(4)                                        # Gap before divider line
            pdf.set_draw_color(200, 200, 200)                # Light grey line color
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())      # Horizontal divider line
            pdf.ln(4)                                        # Gap after divider line

    buffer = io.BytesIO()                                    # Create in-memory buffer — avoids writing to disk
    pdf.output(buffer)                                       # Write PDF content to the buffer
    return buffer.getvalue()                                 # Return raw bytes — what Streamlit download_button expects


if __name__ == "__main__":                                   # Only runs when you execute this file directly
    print("Loading index...")                                # Status message
    engine = load_query_engine()                             # Load the standard query engine
    print("Ready! Testing...\n")                             # Status message
    response = engine.query("What are the minimum water supply standards in humanitarian response?")  
                                                             # Test question to verify everything works
    print(response.response)                                 # Print the answer to terminal