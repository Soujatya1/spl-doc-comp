import streamlit as st
import pandas as pd
import os
import tempfile
import pickle
import re
import unicodedata
import io
from docx import Document
from docx.document import Document as _Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from difflib import SequenceMatcher
from rapidfuzz import fuzz, process
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document as LangchainDocument
from langchain_groq import ChatGroq
import json

# Set page config
st.set_page_config(
    page_title="Document Comparison Tool",
    page_icon="📄",
    layout="wide"
)

# Create temp directory for storing temporary files
temp_dir = tempfile.TemporaryDirectory()

# Main functions from the original code
def preprocess_text(text):
    """Cleans text: removes text in < > brackets, extra spaces, normalizes, lowercases, and removes punctuation (except periods)."""
    if not text:
        return ""
    text = re.sub(r"<.*?>", "", text)  # Remove placeholders/dynamic content
    text = re.sub(r"\s+", " ", text).strip()
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^\w\s.]", "", text)
    return text

def iter_block_items(parent):
    """
    Yield each paragraph and table child in document order.
    """
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._element
    for child in parent_elm.iterchildren():
        if child.tag.endswith("p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("tbl"):
            yield Table(child, parent)

def extract_text_by_sections(docx_path):
    """
    Extracts paragraphs and tables from a DOCX file while maintaining document order.
    Returns a list of dictionaries with keys: section, text, type, and order.
    When a heading (style containing "heading") is encountered, the current section is updated.
    """
    extracted_data = []
    doc = Document(docx_path)
    current_section = "unknown_section"
    order_counter = 0

    for block in iter_block_items(doc):
        if block.__class__.__name__ == "Paragraph":
            para_text = preprocess_text(block.text)
            if block.style and "heading" in block.style.name.lower() and para_text:
                current_section = para_text
            elif para_text:
                extracted_data.append({
                    "section": current_section,
                    "text": para_text,
                    "type": "paragraph",
                    "order": order_counter
                })
                order_counter += 1
        elif block.__class__.__name__ == "Table":
            table_data = []
            for row in block.rows:
                row_cells = [preprocess_text(cell.text.strip()) for cell in row.cells]
                table_data.append(" | ".join(row_cells))
            table_text = "\n".join(table_data)
            if table_text:
                extracted_data.append({
                    "section": current_section,
                    "text": f"[TABLE] {table_text}",
                    "type": "table",
                    "order": order_counter
                })
                order_counter += 1
    return extracted_data

def split_text_into_lines(text):
    """Splits text into lines by newline or pipe symbol."""
    return re.split(r'\n|\||\. ', text)

def find_best_line_match(target, lines):
    """
    Finds the closest matching line from a list of lines for a given target string.
    Uses a dynamic threshold based on the target length and RapidFuzz for faster computation.
    Returns the best matching line if similarity is above threshold, otherwise an empty string.
    """
    target = target.lower().strip()

    # Determine dynamic threshold based on target length.
    if len(target) < 10:
        custom_threshold = 90  # RapidFuzz functions typically use percentages (0-100)
    elif len(target) < 50:
        custom_threshold = 75
    else:
        custom_threshold = 65

    # Use RapidFuzz's process.extractOne to get the best match.
    best = process.extractOne(
        query=target,
        choices=lines,
        scorer=fuzz.ratio,
        score_cutoff=custom_threshold
    )
    if best:
        # best is a tuple: (best_match, score, index)
        return best[0]
    else:
        return ""

def find_closest_match(target, text_list, threshold=0.65):
    """
    Finds the closest matching string from a list using SequenceMatcher.
    If the target is contained in a line, returns that line immediately.
    Otherwise, returns the candidate whose similarity is above the threshold.
    """
    target = target.lower()
    best_match = None
    best_score = threshold
    for text in text_list:
        lower_text = text.lower()
        if target in lower_text:
            return text
        score = SequenceMatcher(None, target, lower_text).ratio()
        if score > best_score:
            best_score = score
            best_match = text
    return best_match

def extract_content_within_markers(text, start_marker, end_marker):
    """
    Splits the text into lines and uses fuzzy matching (via SequenceMatcher) to find the best
    matching start and end markers. Then returns the substring (with both markers included).
    If either marker is not found, returns the full text.
    """
    lines = text.split("\n")
    best_start = find_closest_match(start_marker, lines)
    best_end = find_closest_match(end_marker, lines)
    if not best_start or not best_end:
        st.warning("Could not find both markers exactly; returning full text.")
        return text
    try:
        start_idx = lines.index(best_start)
        end_idx = lines.index(best_end)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx  # Swap if necessary
        return "\n".join(lines[start_idx:end_idx + 1])
    except ValueError:
        st.warning("Error finding marker indices; returning full text.")
        return text

def retrieve_section(page_name, start_marker, end_marker, faiss_index_path):
    """
    Loads the FAISS index from file and retrieves all documents whose metadata contains:
      - "section" that includes the page_name,
      - "start" that includes the start_marker, and
      - "end" that includes the end_marker.
    """
    with open(faiss_index_path, "rb") as f:
        faiss_index = pickle.load(f)

    matching_sections = []
    # Iterate through all stored documents in the FAISS index
    for doc in faiss_index.documents:
        metadata = doc.metadata
        if (page_name.lower() in metadata.get("section", "").lower() and
                start_marker.lower() in metadata.get("start", "").lower() and
                end_marker.lower() in metadata.get("end", "").lower()):
            extracted_text = extract_content_within_markers(doc.page_content, start_marker, end_marker)
            matching_sections.append({
                "text": extracted_text,
                "metadata": metadata
            })
    return matching_sections

def extract_section(extracted_data, start_marker, end_marker):
    """Sorts extracted data by order, combines text, splits into lines, then extracts substring between best matching markers."""
    if not extracted_data:
        return ""
    sorted_data = sorted(extracted_data, key=lambda d: d.get("order", 0))
    combined_text = "\n".join([item["text"] for item in sorted_data])
    lines = combined_text.split("\n")
    best_start = find_closest_match(start_marker, lines)
    best_end = find_closest_match(end_marker, lines)

    if not best_start or not best_end:
        st.warning(f"Could not find {start_marker} or {end_marker}")
        return combined_text
    try:
        start_idx = lines.index(best_start)
        end_idx = lines.index(best_end)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        return "\n".join(lines[start_idx:end_idx+1])
    except ValueError:
        return combined_text

def store_sections_in_faiss(docx_path, checklist_path, faiss_index_path):
    """
    Reads the DOCX file, extracts sections based on the checklist (PageName, StartMarker, EndMarker),
    creates Document objects with metadata, and stores them in a FAISS vector store.
    """
    extracted_data = extract_text_by_sections(docx_path)
    checklist_df = pd.read_excel(checklist_path)
    sections = []
    for _, row in checklist_df.iterrows():
        section_name = row['PageName'].strip().lower()
        start_marker = preprocess_text(row['StartMarker'])
        end_marker = preprocess_text(row['EndMarker'])
        section_text = extract_section(extracted_data, start_marker, end_marker)
        if section_text:
            metadata = {
                "section": section_name,
                "start": start_marker,
                "end": end_marker,
                "source": os.path.basename(docx_path)
            }
            doc_obj = LangchainDocument(page_content=section_text, metadata=metadata)
            sections.append(doc_obj)
    
    # Initialize embedding model
    with st.spinner("Creating embeddings... This may take a moment."):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        faiss_index = FAISS.from_documents(sections, embedding_model)
    
    # Attach documents for later retrieval
    faiss_index.documents = sections
    
    output_dir = os.path.dirname(faiss_index_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(faiss_index_path, "wb") as f:
        pickle.dump(faiss_index, f)
    
    return faiss_index

def format_batch_prompt(df_batch):
    """
    Converts a batch of DataFrame rows into a prompt string.
    Each row is formatted with its row number, company line, and customer line.
    """
    prompt_lines = []
    for idx, row in df_batch.iterrows():
        prompt_lines.append(f"Row {idx}:")
        prompt_lines.append(f"Company: {row['CompanyLine']}")
        prompt_lines.append(f"Customer: {row['CustomerLine']}")
        prompt_lines.append("---")
    prompt_text = "\n".join(prompt_lines)
    prompt_text += (
        "\n\nCompare the above rows line by line. "
        "For each row, output a JSON object with keys 'row' (the row number), 'comparison' (SAME or DIFFERENT), and "
        "'difference' (if different, a brief explanation with specific difference). Return a JSON list of these objects. I want output in JSON only. Do not mention anything else in the response."
    )
    return prompt_text

def process_batch(df_batch, api_key):
    """
    Processes a batch of rows by creating a prompt, calling the LLM, and parsing the JSON response.
    """
    chatgroq_llm = ChatGroq(
        api_key=api_key,
        model_name="Llama3-8b-8192"
    )
    
    prompt = format_batch_prompt(df_batch)
    with st.spinner("Processing batch with LLM..."):
        response = chatgroq_llm.invoke([{"role": "user", "content": prompt}]).content
    
    if not response.strip():
        st.warning("Empty response from LLM.")
        return []
    
    try:
        # Try to extract JSON from the response (in case there's additional text)
        json_start = response.find("[")
        json_end = response.rfind("]") + 1
        if json_start >= 0 and json_end > 0:
            json_str = response[json_start:json_end]
            comparisons = json.loads(json_str)
        else:
            comparisons = json.loads(response)
    except Exception as e:
        st.error(f"Error parsing LLM response: {e}")
        st.code(response)
        comparisons = []
    
    return comparisons

def compare_dataframe(df, api_key, batch_size=50):
    """
    Splits the DataFrame into batches, processes each batch using the LLM for detailed comparison,
    and maps the resulting comparison and difference back into the original DataFrame.
    """
    all_comparisons = []
    
    # Split DataFrame into batches
    with st.spinner("Comparing differences with LLM..."):
        progress_bar = st.progress(0)
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            comparisons = process_batch(batch, api_key)
            all_comparisons.extend(comparisons)
            
            # Update progress
            progress_bar.progress(min(1.0, (i + batch_size) / len(df)))
    
    # Reset progress bar
    progress_bar.empty()
    
    # Merge the comparisons back into the original DataFrame
    df['Comparison'] = "N/A"
    df['Difference'] = ""
    
    for comp in all_comparisons:
        row_idx = comp.get('row')
        if row_idx is not None and row_idx < len(df):
            df.at[row_idx, 'Comparison'] = comp.get('comparison', 'N/A')
            df.at[row_idx, 'Difference'] = comp.get('difference', '')
    
    return df

# Streamlit UI components
def main():
    st.title("Document Comparison Tool")
    
    # Create tabs for different functionality
    tab1, tab2, tab3 = st.tabs(["Document Processing", "Text Comparison", "Results"])
    
    # Session state initialization
    if 'processed_docx' not in st.session_state:
        st.session_state.processed_docx = False
    if 'comparison_ready' not in st.session_state:
        st.session_state.comparison_ready = False
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = None
    if 'faiss_index_path' not in st.session_state:
        st.session_state.faiss_index_path = os.path.join(temp_dir.name, "faiss_index.pkl")
    
    with tab1:
        st.header("Document Processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            docx_file = st.file_uploader("Upload DOCX Document", type=["docx"])
        
        with col2:
            checklist_file = st.file_uploader("Upload Checklist (Excel)", type=["xlsx", "xls"])
        
        if docx_file and checklist_file:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    # Save uploaded files to temp directory
                    docx_path = os.path.join(temp_dir.name, "document.docx")
                    checklist_path = os.path.join(temp_dir.name, "checklist.xlsx")
                    
                    with open(docx_path, "wb") as f:
                        f.write(docx_file.getvalue())
                    
                    with open(checklist_path, "wb") as f:
                        f.write(checklist_file.getvalue())
                    
                    # Process and store sections
                    store_sections_in_faiss(
                        docx_path, 
                        checklist_path, 
                        st.session_state.faiss_index_path
                    )
                    
                    st.session_state.processed_docx = True
                    st.success("Documents processed successfully!")
    
    with tab2:
        st.header("Text Comparison")
        
        if not st.session_state.processed_docx:
            st.info("Please process documents in the Document Processing tab first.")
        else:
            # API key input
            api_key = st.text_input("Enter Groq API Key", type="password")
            
            # Upload comparison file
            comparison_file = st.file_uploader("Upload Comparison File (Excel)", type=["xlsx", "xls"])
            
            if comparison_file and api_key:
                if st.button("Run Comparison"):
                    with st.spinner("Loading comparison data..."):
                        comparison_df = pd.read_excel(comparison_file)
                        
                        # Check if required columns exist
                        required_cols = ["CompanyLine", "CustomerLine"]
                        if not all(col in comparison_df.columns for col in required_cols):
                            st.error(f"Comparison file must contain columns: {', '.join(required_cols)}")
                        else:
                            # Run comparison
                            batch_size = st.slider("Batch Size", min_value=10, max_value=100, value=50)
                            result_df = compare_dataframe(comparison_df, api_key, batch_size=batch_size)
                            
                            # Store results in session state
                            st.session_state.comparison_results = result_df
                            st.session_state.comparison_ready = True
                            
                            st.success("Comparison completed! View results in the Results tab.")
    
    with tab3:
        st.header("Results")
        
        if not st.session_state.comparison_ready:
            st.info("Run a comparison in the Text Comparison tab to see results here.")
        else:
            # Display comparison results
            results_df = st.session_state.comparison_results
            
            # Filter options
            st.subheader("Filter Results")
            comparison_filter = st.selectbox(
                "Filter by Comparison",
                ["All", "SAME", "DIFFERENT", "N/A"]
            )
            
            # Apply filters
            filtered_df = results_df
            if comparison_filter != "All":
                filtered_df = filtered_df[filtered_df["Comparison"] == comparison_filter]
            
            # Display filtered results
            st.subheader("Comparison Results")
            st.dataframe(filtered_df)
            
            # Download option
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="comparison_results.csv",
                mime="text/csv"
            )
            
            # Summary statistics
            st.subheader("Summary")
            total = len(results_df)
            same = len(results_df[results_df["Comparison"] == "SAME"])
            different = len(results_df[results_df["Comparison"] == "DIFFERENT"])
            na = len(results_df[results_df["Comparison"] == "N/A"])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Comparisons", total)
            col2.metric("Same", same, f"{same/total*100:.1f}%" if total > 0 else "0%")
            col3.metric("Different", different, f"{different/total*100:.1f}%" if total > 0 else "0%")

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
