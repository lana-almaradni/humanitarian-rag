import streamlit as st                                       # Streamlit — turns Python into a web app
from query_engine import load_query_engine, load_chat_engine, compare_frameworks, export_to_pdf  
                                                             # Import all functions from query_engine.py

st.set_page_config(                                          # Must be the FIRST Streamlit call in the file
    page_title="Humanitarian Policy Assistant",              # Browser tab title
    page_icon="🌍",                                          # Browser tab icon
    layout="wide"                                            # Use full width of the browser
)

st.title("🌍 Humanitarian Policy Assistant")                 # Large heading at top of page
st.caption("Ask plain English questions about humanitarian standards, SDGs, and disaster risk reduction frameworks")  
                                                             # Subtitle below the title

@st.cache_resource                                           # Decorator — runs this function ONCE, caches forever
def get_engine():                                            # Without cache, index reloads on every user interaction
    return load_chat_engine()                                # Chat engine with conversation memory

@st.cache_resource                                           # Separate cached engine for compare mode
def get_compare_engine():                                    # No memory needed for framework comparisons
    return load_query_engine()                               # Standard query engine

with st.spinner("Loading humanitarian documents..."):        # Show loading message while engines initialize
    engine = get_engine()                                    # First run: loads from disk. After: instant from cache
    compare_engine = get_compare_engine()                    # Same — loaded once, reused forever

if "messages" not in st.session_state:                       # Check if this is a fresh session
    st.session_state.messages = []                           # Create empty list to store conversation history
                                                             # MUST be before sidebar — sidebar references messages

with st.sidebar:                                             # Everything inside renders in the left panel
    st.header("📄 Loaded Documents")                         # Sidebar section header
    st.success("✅ Sphere Handbook 2018")                     # Green checkbox — shows loaded documents
    st.success("✅ OCHA Glossary of Humanitarian Terms")      # Green checkbox
    st.success("✅ OCHA Protection of Civilians Glossary")    # Green checkbox
    st.success("✅ UN 2030 Agenda for Sustainable Development")  # Green checkbox
    st.success("✅ Sendai Framework for DRR 2015-2030")       # Green checkbox
    st.divider()                                             # Horizontal line separator
    compare_mode = st.toggle("⚖️ Compare Frameworks mode", value=False)  
                                                             # Toggle switch — True when on, False when off
    if compare_mode:                                         # Show hint only when compare mode is active
        st.info("Ask: 'How do Sphere and the SDGs approach water and sanitation?'")  
    if len(st.session_state.messages) > 0:                   # Only show export button after first question
        st.divider()                                         # Separator before export button
        pdf_bytes = export_to_pdf(st.session_state.messages) # Generate PDF from current conversation
        st.download_button(                                  # Renders a download button in sidebar
            label="⬇️ Export Q&A as PDF",                    # Button label
            data=pdf_bytes,                                  # The PDF bytes to download
            file_name="humanitarian_policy_report.pdf",      # Downloaded file name
            mime="application/pdf"                           # File MIME type
        )
    st.divider()                                             # Final separator
    st.caption("Answers are grounded in loaded documents only. Always verify with official sources.")  

for msg in st.session_state.messages:                        # Loop through all previous messages
    with st.chat_message(msg["role"]):                       # Renders as user or assistant bubble
        st.markdown(msg["content"])                          # Display message text with markdown formatting
        if msg["role"] == "assistant" and "sources" in msg:  # Show sources only for assistant messages
            with st.expander("📎 Source passages"):          # Collapsible citations section
                for s in msg["sources"]:                     # Loop through each source citation string
                    try:                                     # Safely attempt to extract the score
                        score = float(s.split("relevance: ")[1].replace(")", ""))  
                                                             # Parse score from string e.g. "relevance: 0.798)"
                        confidence = "🟢 High" if score < 0.85 else "🟡 Medium" if score < 1.0 else "🔴 Low"  
                                                             # Assign confidence level based on score thresholds
                    except:                                  # If parsing fails for any reason
                        confidence = ""                      # Show no indicator rather than crash
                    st.caption(f"📄 {s}  {confidence}")      # Display source with confidence indicator
                    
if not st.session_state.messages:                            # Only show sample buttons on empty chat
    st.subheader("Try asking:")                              # Section header for sample questions
    cols = st.columns(2)                                     # Two equal columns side by side
    questions = [                                            # List of sample questions
        "What are the minimum water supply standards?",
        "What does SDG 6 require for clean water access?",
        "How should humanitarian actors handle protection of civilians?",
        "What are the Sendai Framework's 4 priorities for action?",
    ]
    for i, q in enumerate(questions):                        # Loop through questions with index
        if cols[i % 2].button(q, use_container_width=True):  # Alternate between left and right columns
            st.session_state.prefill = q                     # Store clicked question in session state
            st.rerun()                                       # Rerun app so question gets processed

if "prefill" in st.session_state:                            # Check if a button was clicked
    question = st.session_state.pop("prefill")               # Get and remove the prefilled question
else:
    question = None                                          # No button clicked

user_input = st.chat_input("Ask a humanitarian policy question...")  
                                                             # Chat input box fixed at bottom of page
if user_input:                                               # True when user types and presses Enter
    question = user_input                                    # Use typed question

if question:                                                 # Process question from either source
    st.session_state.messages.append({"role": "user", "content": question})  
                                                             # Save user message to history
    with st.chat_message("user"):                            # Open user message bubble
        st.markdown(question)                                # Display the question

    with st.chat_message("assistant"):                       # Open assistant message bubble
        with st.spinner("Searching humanitarian frameworks..."):  
                                                             # Show loading indicator while querying
            if compare_mode:                                 # Compare mode is on
                response = compare_frameworks(compare_engine, question)  
                                                             # Use structured comparison prompt
            else:                                            # Normal mode
                response = engine.chat(question)             # Full RAG pipeline with conversation memory

            answer = response.response                       # Extract the text answer from response object

            sources = []                                     # Empty list for source citations
            source_nodes = getattr(response, 'source_nodes', [])  
                                                             # Safely get source nodes — chat may not always have them
            for node in source_nodes:                        # Loop through retrieved chunks
                meta = node.metadata                         # Get the metadata stored during ingestion
                fname = meta.get("file_name", "Unknown")     # Which PDF this chunk came from
                page = meta.get("page_number", "?")          # Which page number
                score = round(node.score, 3)                 # Similarity score — lower = more similar
                sources.append(f"{fname} — page {page} (relevance: {score})")  
                                                             # Add formatted citation to list

            st.markdown(answer)                              # Display the answer with markdown formatting

            if source_nodes:                                 # Only show expander if sources exist
                with st.expander("📎 Source passages"):      # Collapsible source passages section
                    for node in source_nodes:                # Loop through each retrieved chunk
                        meta = node.metadata                 # Get metadata
                        fname = meta.get("file_name", "Unknown")  
                        page = meta.get("page_number", "?")
                        score = round(node.score, 3)
                        confidence = "🟢 High" if score < 0.85 else "🟡 Medium" if score < 1.0 else "🔴 Low"  
                                                             # Score thresholds for confidence levels
                        st.markdown(f"**📄 {fname} — page {page}** &nbsp; {confidence}")  
                                                             # Show document, page, confidence indicator
                        st.caption(node.text[:400] + "..." if len(node.text) > 400 else node.text)  
                                                             # Show first 400 chars of retrieved passage
                        st.divider()                         # Line between source passages

    st.session_state.messages.append({                       # Save assistant response to history
        "role": "assistant",                                 # Mark as assistant message
        "content": answer,                                   # The answer text
        "sources": sources                                   # The source citations list
    })
    st.rerun()                                               # Force sidebar refresh so export button appears