import streamlit as st
import os
import pandas as pd
import re
import unicodedata
import pickle
from difflib import SequenceMatcher
from rapidfuzz import fuzz, process
from docx import Document
from docx.document import Document as _Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document as LangchainDocument
import json
import tempfile

# Set page configuration
st.set_page_config(
    page_title="Document Comparison Tool",
    page_icon="📄",
    layout="wide"
)

# Helper functions
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

def extract_text_by_sections(docx_file):
    """
    Extracts paragraphs and tables from a DOCX file while maintaining document order.
    Returns a list of dictionaries with keys: section, text, type, and order.
    When a heading (style containing "heading") is encountered, the current section is updated.
    """
    extracted_data = []
    doc = Document(docx_file)
    current_section = "unknown_section"
    order_counter = 0

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
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
        elif isinstance(block, Table):
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
    Splits the text into lines and uses fuzzy matching to find the best
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
        st.warning(f"Could not find markers: {start_marker} or {end_marker}")
        return combined_text
    try:
        start_idx = lines.index(best_start)
        end_idx = lines.index(best_end)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        return "\n".join(lines[start_idx:end_idx+1])
    except ValueError:
        return combined_text

def store_sections_in_faiss(docx_data, checklist_df, temp_dir):
    """
    Processes the DOCX data, extracts sections based on the checklist,
    creates Document objects with metadata, and stores them in a FAISS vector store.
    Returns the path to the saved FAISS index.
    """
    extracted_data = extract_text_by_sections(docx_data)
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
                "source": "uploaded_document"
            }
            doc_obj = LangchainDocument(page_content=section_text, metadata=metadata)
            sections.append(doc_obj)
    
    # Create embeddings and FAISS index
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index = FAISS.from_documents(sections, embedding_model)
    faiss_index.documents = sections
    
    # Save to temporary file
    faiss_path = os.path.join(temp_dir, "faiss_index.pkl")
    with open(faiss_path, "wb") as f:
        pickle.dump(faiss_index, f)
    
    return faiss_path

def retrieve_section(page_name, start_marker, end_marker, faiss_index_path):
    """
    Loads the FAISS index from file and retrieves documents matching the section criteria.
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

def format_batch_prompt(df_batch):
    """
    Converts a batch of DataFrame rows into a prompt string for LLM processing.
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
        "'difference' (if different, a brief explanation with specific difference). Return a JSON list of these objects. Output JSON only."
    )
    return prompt_text

def rule_based_comparison(df):
    """
    A rule-based comparison function to use when LLM is not available.
    """
    for idx, row in df.iterrows():
        comp_line = row["CompanyLine"].lower().replace(" ", "")
        cust_line = row["CustomerLine"].lower().replace(" ", "")
        
        if not cust_line:
            df.at[idx, "Comparison"] = "DIFFERENT"
            df.at[idx, "Difference"] = "Line missing in customer document"
        elif comp_line == cust_line:
            df.at[idx, "Comparison"] = "SAME"
            df.at[idx, "Difference"] = ""
        elif comp_line in cust_line or cust_line in comp_line:
            df.at[idx, "Comparison"] = "SIMILAR"
            df.at[idx, "Difference"] = "Content similar but not identical"
        else:
            df.at[idx, "Comparison"] = "DIFFERENT"
            df.at[idx, "Difference"] = "Content differs"
    
    return df

def save_df_to_excel(df, output_path):
    """Save DataFrame to Excel with formatting."""
    df.to_excel(output_path, index=False)
    return output_path

def process_documents(company_data, customer_data, checklist_data, use_llm=False):
    """
    Main processing function that:
    1. Creates temporary FAISS indexes for company and customer documents
    2. Retrieves matching sections based on checklist
    3. Compares lines between documents
    4. Returns a DataFrame with comparison results
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Load checklist data
        checklist_df = pd.read_excel(checklist_data)
        
        # Create FAISS indexes
        st.info("Indexing company document...")
        company_faiss_path = store_sections_in_faiss(company_data, checklist_df, temp_dir)
        
        st.info("Indexing customer document...")
        customer_faiss_path = store_sections_in_faiss(customer_data, checklist_df, temp_dir)
        
        final_rows = []
        
        # Process each section from checklist
        st.info("Comparing documents section by section...")
        progress_bar = st.progress(0)
        
        for i, row in enumerate(checklist_df.iterrows()):
            _, checklist_row = row
            section_name = checklist_row['PageName'].strip().lower()
            start_marker = preprocess_text(checklist_row['StartMarker'])
            end_marker = preprocess_text(checklist_row['EndMarker'])
            
            company_section = retrieve_section(section_name, start_marker, end_marker, company_faiss_path)
            customer_section = retrieve_section(section_name, start_marker, end_marker, customer_faiss_path)
            
            if not company_section or not customer_section:
                st.warning(f"Could not retrieve sections for {section_name}")
                continue
            
            company_lines = [line.strip() for line in split_text_into_lines(company_section[0]["text"]) 
                             if line.strip() and "[TABLE]" not in line]
            customer_lines = [line.strip() for line in split_text_into_lines(customer_section[0]["text"]) 
                              if line.strip() and "[TABLE]" not in line]
            
            for comp_line in company_lines:
                best_cust_line = find_best_line_match(comp_line, customer_lines)
                final_rows.append({
                    "Section": section_name,
                    "CompanyLine": comp_line,
                    "CustomerLine": best_cust_line
                })
            
            # Update progress
            progress_bar.progress((i + 1) / len(checklist_df))
        
        # Create DataFrame and perform comparisons
        df = pd.DataFrame(final_rows).drop_duplicates()
        df["order"] = df.index
        
        # Simple rule-based comparison for all rows
        same_rows = []
        different_rows = []
        
        for idx, row in df.iterrows():
            norm_company = row["CompanyLine"].lower().replace(" ", "")
            norm_customer = row["CustomerLine"].lower().replace(" ", "")
            
            if norm_company == norm_customer:
                same_rows.append({**row, "Comparison": "SAME", "Difference": ""})
            elif norm_company in norm_customer:
                same_rows.append({**row, "Comparison": "SAME", "Difference": ""})
            elif norm_customer == "":
                same_rows.append({**row, "Comparison": "DIFFERENT", "Difference": "Could not find similar line in customer document"})
            else:
                different_rows.append(row)
        
        df_same = pd.DataFrame(same_rows)
        df_different = pd.DataFrame(different_rows)
        
        # Use rule-based comparison instead of LLM
        if not df_different.empty:
            df_diff_compared = rule_based_comparison(df_different)
        else:
            df_diff_compared = df_different.copy()
        
        # Merge results and sort by original order
        df_final = pd.concat([df_same, df_diff_compared]).sort_values("order")
        df_final = df_final.drop(columns=["order"])
        
        return df_final

# Streamlit UI
def main():
    st.title("Document Comparison Tool")
    st.write("Compare two DOCX documents based on sections defined in a checklist Excel file.")
    
    # File uploads
    st.header("Upload Documents")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        company_file = st.file_uploader("Upload Company Document (DOCX)", type=["docx"])
    
    with col2:
        customer_file = st.file_uploader("Upload Customer Document (DOCX)", type=["docx"])
    
    with col3:
        checklist_file = st.file_uploader("Upload Checklist (Excel)", type=["xlsx", "xls"])
    
    # Advanced options
    with st.expander("Advanced Options"):
        use_llm = st.checkbox("Use LLM for detailed comparison (requires API key)", value=False)
        if use_llm:
            api_key = st.text_input("Enter LLM API Key (e.g., Groq, OpenAI)", type="password")
            st.warning("LLM integration is disabled in this version")
    
    # Process documents when all files are uploaded and button is clicked
    if company_file and customer_file and checklist_file:
        if st.button("Compare Documents"):
            with st.spinner("Processing documents..."):
                try:
                    # Process documents
                    result_df = process_documents(company_file, customer_file, checklist_file, use_llm=False)
                    
                    # Display results
                    st.header("Comparison Results")
                    
                    # Summary metrics
                    st.subheader("Summary")
                    total_lines = len(result_df)
                    same_lines = len(result_df[result_df['Comparison'] == 'SAME'])
                    diff_lines = total_lines - same_lines
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Lines", total_lines)
                    col2.metric("Matching Lines", same_lines)
                    col3.metric("Different Lines", diff_lines)
                    
                    # Show the data with conditional formatting
                    st.subheader("Detailed Results")
                    
                    # Apply styling to the dataframe
                    def highlight_differences(s):
                        return ['background-color: #ffcccc' if x == 'DIFFERENT' else 
                                'background-color: #ffffcc' if x == 'SIMILAR' else
                                'background-color: #ccffcc' for x in s]
                    
                    styled_df = result_df.style.apply(highlight_differences, subset=['Comparison'])
                    st.dataframe(styled_df)
                    
                    # Option to download the results
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                        result_path = tmp.name
                        save_df_to_excel(result_df, result_path)
                    
                    with open(result_path, 'rb') as f:
                        st.download_button(
                            "Download Results as Excel",
                            f,
                            file_name="document_comparison_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    # Clean up temporary file
                    os.unlink(result_path)
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    st.exception(e)
    else:
        st.info("Please upload all required files to proceed.")

    # Add some help information
    st.markdown("---")
    with st.expander("How to use this tool"):
        st.markdown("""
        1. **Upload Company Document**: The original or reference document in DOCX format
        2. **Upload Customer Document**: The document to compare against the company document
        3. **Upload Checklist**: An Excel file with columns:
           - PageName: Section name
           - StartMarker: Text that marks the beginning of a section
           - EndMarker: Text that marks the end of a section
        4. Click **Compare Documents** to start the comparison process
        5. Review the results and download the Excel report
        """)

# Run the app
if __name__ == "__main__":
    main()
