import streamlit as st
import pandas as pd
import os
import tempfile
import pickle
import re
import unicodedata
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
            progress_bar.progress((i + batch_size) / len(df))
    
    # Merge the comparisons back into the original DataFrame
    df['Comparison'] = df.index.map(
        lambda idx: next((item['comparison'] for item in all_comparisons if item['row'] == idx), "N/A"))
    df['Difference'] = df.index.map(
        lambda idx: next((item.get('difference', '') for item in all_comparisons if item['row'] == idx), ""))
    
    return df

def format_to_structured_table(comparison_results, api_key):
    """
    Creates a prompt for the LLM to structure comparison results into a standard format
    """
    chatgroq_llm = ChatGroq(
        api_key=api_key,
        model_name="Llama3-8b-8192"
    )
    
    prompt = """
    Based on the following document comparison results, create a structured table in the following format:
    
    | Product UID | Samples affected | Observation - Category | Page | Sub-category of Observation |
    
    For each difference identified, categorize it into one of these observation types:
    - Mismatch of content between Filed Copy and customer copy
    - Available in Filed Copy but missing in Customer Copy
    - Document specification differences
    
    Here are the comparison results:
    """
    
    # Add all comparison results to the prompt
    for index, result in comparison_results.iterrows():
        if result.get('Comparison') == 'DIFFERENT':
            prompt += f"\nSection: {result['Section']}\n"
            prompt += f"Company Line: {result['CompanyLine']}\n"
            prompt += f"Customer Line: {result['CustomerLine']}\n"
            prompt += f"Difference: {result.get('Difference', '')}\n"
            prompt += "---\n"
    
    # Ask the LLM to format in the desired structure
    prompt += "\nPlease organize these differences into the table format shown above."
    
    # Call LLM
    with st.spinner("Generating structured summary..."):
        response = chatgroq_llm.invoke([{"role": "user", "content": prompt}]).content
    
    return response

def process_documents(company_docx, customer_docx, checklist_path, output_excel_path, company_faiss_path, customer_faiss_path, api_key):
    """
    Main function to process and compare documents
    """
    # Step 1: Store sections from both documents in FAISS using the checklist markers
    with st.spinner("Analyzing company document..."):
        store_sections_in_faiss(company_docx, checklist_path, company_faiss_path)
    
    with st.spinner("Analyzing customer document..."):
        store_sections_in_faiss(customer_docx, checklist_path, customer_faiss_path)

    # Read checklist file
    checklist_df = pd.read_excel(checklist_path)
    final_rows = []

    # Process each checklist entry
    with st.spinner("Processing document sections..."):
        progress_bar = st.progress(0)
        total_sections = len(checklist_df)
        
        for i, row in checklist_df.iterrows():
            section_name = row['PageName'].strip().lower()
            start_marker = preprocess_text(row['StartMarker'])
            end_marker = preprocess_text(row['EndMarker'])
            
            company_section = retrieve_section(section_name, start_marker, end_marker, company_faiss_path)
            customer_section = retrieve_section(section_name, start_marker, end_marker, customer_faiss_path)

            if not company_section or not customer_section:
                st.warning(f"Could not retrieve sections for {section_name}")
                continue

            company_lines = [line.strip() for line in split_text_into_lines(company_section[0]["text"]) if
                            line.strip() and "[TABLE]" not in line]
            customer_lines = [line.strip() for line in split_text_into_lines(customer_section[0]["text"]) if
                            line.strip() and "[TABLE]" not in line]

            for comp_line in company_lines:
                best_cust_line = find_best_line_match(comp_line, customer_lines)
                final_rows.append({
                    "Section": section_name,
                    "CompanyLine": comp_line,
                    "CustomerLine": best_cust_line
                })
            
            # Update progress
            progress_bar.progress((i + 1) / total_sections)

    df = pd.DataFrame(final_rows).drop_duplicates()
    df["order"] = df.index
    same_rows = []
    different_rows = []

    # First-pass comparison
    with st.spinner("Performing initial comparison..."):
        for idx, row in df.iterrows():
            norm_company = row["CompanyLine"].lower().replace(" ", "")
            norm_customer = row["CustomerLine"].lower().replace(" ", "")
            if norm_company == norm_customer:
                same_rows.append({**row, "Comparison": "SAME"})
            elif norm_company in norm_customer:
                same_rows.append({**row, "Comparison": "SAME"})
            elif norm_customer == "":
                same_rows.append({**row, "Comparison": "DIFFERENT", "Difference": "Could not find similar line in customer document"})
            else:
                different_rows.append(row)

    df_same = pd.DataFrame(same_rows)
    df_different = pd.DataFrame(different_rows)
    
    st.info(f"Initial analysis: {df_same.shape[0]} identical lines, {df_different.shape[0]} differences requiring detailed comparison")

    # Detailed LLM-based comparison for different lines
    if not df_different.empty:
        df_diff_compared = compare_dataframe(df_different, api_key, batch_size=50)
    else:
        df_diff_compared = df_different.copy()

    # Merge the results
    df_final = pd.concat([df_same, df_diff_compared]).sort_values("order")
    df_final = df_final.drop(columns=["order"])
    
    # Save to Excel
    df_final.to_excel(output_excel_path, index=False)
    st.success(f"Comparison results saved to Excel")
    
    return df_final

# Streamlit UI
st.title("Document Comparison Tool")
st.markdown("""
This tool analyzes and compares two document versions, highlighting differences and generating a structured report.
""")

# File Upload Area
st.header("Upload Documents")
col1, col2, col3 = st.columns(3)

with col1:
    company_doc = st.file_uploader("Upload Company Document (DOCX)", type=["docx"])
    
with col2:
    customer_doc = st.file_uploader("Upload Customer Document (DOCX)", type=["docx"])
    
with col3:
    checklist_file = st.file_uploader("Upload Checklist (XLSX)", type=["xlsx"])

# API Key Input
groq_api_key = st.text_input("Enter Groq API Key", value="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", type="password")

# Process Button
if st.button("Compare Documents", disabled=not (company_doc and customer_doc and checklist_file)):
    if not groq_api_key:
        st.error("Please enter a valid Groq API key")
    else:
        # Save uploaded files to temp directory
        company_path = os.path.join(temp_dir.name, "company.docx")
        customer_path = os.path.join(temp_dir.name, "customer.docx")
        checklist_path = os.path.join(temp_dir.name, "checklist.xlsx")
        
        with open(company_path, "wb") as f:
            f.write(company_doc.getvalue())
        
        with open(customer_path, "wb") as f:
            f.write(customer_doc.getvalue())
        
        with open(checklist_path, "wb") as f:
            f.write(checklist_file.getvalue())
        
        # Set paths for output and FAISS indices
        company_faiss_path = os.path.join(temp_dir.name, "company_faiss.pkl")
        customer_faiss_path = os.path.join(temp_dir.name, "customer_faiss.pkl")
        output_excel_path = os.path.join(temp_dir.name, "comparison_results.xlsx")
        
        # Process documents
        with st.spinner("Processing documents... This may take a few minutes."):
            try:
                comparison_df = process_documents(
                    company_path, 
                    customer_path, 
                    checklist_path, 
                    output_excel_path, 
                    company_faiss_path, 
                    customer_faiss_path,
                    groq_api_key
                )
                
                # Create structured table from results
                structured_output = format_to_structured_table(comparison_df, groq_api_key)
                
                # Display results
                st.header("Comparison Results")
                
                # Display structured summary
                st.subheader("Structured Summary")
                st.markdown(structured_output)
                
                # Display raw comparison data
                st.subheader("Detailed Comparison")
                st.dataframe(
                    comparison_df,
                    column_config={
                        "Section": st.column_config.TextColumn("Section"),
                        "CompanyLine": st.column_config.TextColumn("Company Document"),
                        "CustomerLine": st.column_config.TextColumn("Customer Document"),
                        "Comparison": st.column_config.TextColumn("Comparison"),
                        "Difference": st.column_config.TextColumn("Explanation")
                    },
                    use_container_width=True
                )
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    with open(output_excel_path, "rb") as f:
                        excel_data = f.read()
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_data,
                        file_name="document_comparison.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    st.download_button(
                        label="Download Structured Summary",
                        data=structured_output,
                        file_name="structured_summary.md",
                        mime="text/markdown"
                    )
                
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")

# Cleanup on session end
def cleanup():
    temp_dir.cleanup()

# Register cleanup function
st.session_state.setdefault("cleanup_registered", False)
if not st.session_state["cleanup_registered"]:
    st.session_state["cleanup_registered"] = True
    import atexit
    atexit.register(cleanup)

# Footer
st.markdown("---")
st.markdown("Document Comparison Tool © 2025")
