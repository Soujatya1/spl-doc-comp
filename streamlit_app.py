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

# NEW FUNCTIONS FOR STRUCTURED FINDINGS

def categorize_differences(comparison_df):
    """
    Takes the comparison dataframe and categorizes differences into structured format
    similar to the Excel template shown in the example.
    """
    # Initialize an empty dataframe for the structured findings
    findings_df = pd.DataFrame(columns=[
        'Samples affected', 
        'Observation - Category', 
        'Page', 
        'Sub-category of Observation'
    ])
    
    # Group by similar differences
    diff_groups = {}
    for idx, row in comparison_df.iterrows():
        if row['Comparison'] == 'DIFFERENT':
            # Create key based on the type of difference
            # We'll use NLP categorization in a more sophisticated version
            difference_text = row['Difference'].lower()
            
            # Basic categorization logic
            category = ""
            page = ""
            sub_category = row['Difference']
            
            # Determine category based on keywords in the difference
            if 'address' in difference_text:
                category = "Mismatch of content between Filed Copy and customer copy"
                page = "Address & Contact Details of Ombudsman Centres"
            elif 'missing' in difference_text:
                category = "Available in Filed Copy but missing in Customer Copy"
                if 'address' in difference_text or 'contact' in difference_text:
                    page = "Tnc (Ombudsman Address)"
                elif 'annexure' in difference_text:
                    page = "Tnc (Annexure AA,BB &CC)"
            elif 'policy' in difference_text or 'quotation' in difference_text:
                category = "Mismatch of content between Filed Copy and customer copy"
                page = "CIS"
            elif 'heading' in difference_text:
                category = "Mismatch of content between Filed Copy and customer copy"
                page = "Forwarding Letter"
            elif 'period' in difference_text or 'freelook' in difference_text:
                category = "Mismatch of content between Filed Copy and customer copy"
                page = "Forwarding Letter"
            else:
                category = "Other discrepancy"
                page = "Unknown"
            
            # Create group key
            group_key = f"{category}|{page}"
            
            # Add to group
            if group_key not in diff_groups:
                diff_groups[group_key] = {
                    'samples': set(),
                    'sub_categories': set()
                }
            
            # Add sample ID and sub-category
            sample_id = str(row.get('SampleID', idx))
            diff_groups[group_key]['samples'].add(sample_id)
            diff_groups[group_key]['sub_categories'].add(sub_category)
    
    # Convert groups to structured findings
    findings_rows = []
    for group_key, group_data in diff_groups.items():
        category, page = group_key.split('|')
        samples = ', '.join(sorted(group_data['samples']))
        sub_categories = '\n'.join(group_data['sub_categories'])
        
        # Use "All Samples" if many samples are affected
        if len(group_data['samples']) > 5:
            samples = "All Samples"
        
        findings_rows.append({
            'Samples affected': samples,
            'Observation - Category': category,
            'Page': page,
            'Sub-category of Observation': sub_categories
        })
    
    # Convert to DataFrame
    if findings_rows:
        findings_df = pd.DataFrame(findings_rows)
    
    return findings_df

def categorize_with_llm(comparison_df, api_key):
    """
    Uses LLM to categorize differences into structured format.
    This provides more sophisticated categorization than rule-based approach.
    """
    chatgroq_llm = ChatGroq(
        api_key=api_key,
        model_name="Llama3-8b-8192"
    )
    
    # Filter only rows with differences
    diff_rows = comparison_df[comparison_df['Comparison'] == 'DIFFERENT']
    
    if diff_rows.empty:
        return pd.DataFrame(columns=[
            'Samples affected', 
            'Observation - Category', 
            'Page', 
            'Sub-category of Observation'
        ])
    
    # Prepare the prompt
    prompt = """
    Analyze the following document comparison differences and categorize them into a structured format. 
    For each difference, determine:
    1. What category of observation it belongs to (e.g., "Mismatch of content between Filed Copy and customer copy", "Available in Filed Copy but missing in Customer Copy")
    2. Which page/section it relates to (e.g., "Forwarding Letter", "CIS", "Address & Contact Details of Ombudsman Centres")
    3. A specific sub-category description of the observation
    
    Here are the differences:
    """
    
    for idx, row in diff_rows.iterrows():
        sample_id = row.get('SampleID', idx)
        prompt += f"\nSample {sample_id}: {row['Difference']}"
    
    prompt += """
    
    Return a JSON array where each object represents a group of similar differences with the following structure:
    {
        "samples_affected": ["sample1", "sample2", ...], 
        "observation_category": "Category name",
        "page": "Page name",
        "sub_category": "Detailed description of the observation"
    }
    
    Group similar differences together. If many samples (more than 5) have the same issue, use "All Samples" for samples_affected.
    """
    
    with st.spinner("Using LLM to categorize findings..."):
        response = chatgroq_llm.invoke([{"role": "user", "content": prompt}]).content
    
    try:
        # Extract JSON from response (it might be wrapped in markdown code blocks)
        json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response
        
        # Parse the JSON
        categories = json.loads(json_str)
        
        # Convert to DataFrame
        findings_rows = []
        for cat in categories:
            samples = cat.get('samples_affected', [])
            if isinstance(samples, list) and len(samples) > 0:
                samples_str = ', '.join(samples) if samples[0] != "All Samples" else "All Samples"
            else:
                samples_str = "Unknown"
                
            findings_rows.append({
                'Samples affected': samples_str,
                'Observation - Category': cat.get('observation_category', 'Other'),
                'Page': cat.get('page', 'Unknown'),
                'Sub-category of Observation': cat.get('sub_category', '')
            })
        
        findings_df = pd.DataFrame(findings_rows)
        return findings_df
        
    except Exception as e:
        st.error(f"Error parsing LLM categorization response: {e}")
        st.code(response)
        # Fall back to rule-based categorization
        return categorize_differences(comparison_df)

def generate_structured_findings(comparison_df, api_key=None):
    """
    Uses LLM to enhance categorization of differences and format them 
    into the structured Excel template format.
    """
    if api_key:
        # Use LLM for more sophisticated categorization
        llm_categorized = categorize_with_llm(comparison_df, api_key)
        return llm_categorized
    else:
        # Use rule-based categorization
        return categorize_differences(comparison_df)

# STREAMLIT UI
def main():
    st.title("Document Comparison Tool")
    
    # Initialize session state
    if 'comparison_results' not in st.session_state:
        st.session_state['comparison_results'] = None
    if 'findings_report' not in st.session_state:
        st.session_state['findings_report'] = None
    
    # Create tabs for different sections
    tabs = st.tabs(["Document Comparison", "Structured Findings", "Settings"])
    
    with tabs[0]:
        st.header("Document Comparison")
        
        # File upload section
        st.subheader("Upload Documents")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            company_file = st.file_uploader("Upload Company Document (DOCX)", type=["docx"])
        
        with col2:
            customer_file = st.file_uploader("Upload Customer Document (DOCX)", type=["docx"])
        
        with col3:
            checklist_file = st.file_uploader("Upload Checklist (Excel)", type=["xlsx", "xls"])
        
        # Process files if uploaded
        if company_file and customer_file and checklist_file:
            # Save uploaded files to temp directory
            company_path = os.path.join(temp_dir.name, "company.docx")
            customer_path = os.path.join(temp_dir.name, "customer.docx")
            checklist_path = os.path.join(temp_dir.name, "checklist.xlsx")
    
            # Make sure the temp directory exists
            os.makedirs(temp_dir.name, exist_ok=True)
    
            # Write files with proper error handling
            try:
                with open(company_path, "wb") as f:
                    f.write(company_file.getvalue())
                with open(customer_path, "wb") as f:
                    f.write(customer_file.getvalue())
                with open(checklist_path, "wb") as f:
                    f.write(checklist_file.getvalue())
            
                # Verify files were saved correctly
                if not os.path.exists(company_path) or not os.path.exists(customer_path) or not os.path.exists(checklist_path):
                    st.error("Error saving uploaded files. Please try again.")
                else:
                    st.success("Files uploaded successfully!")
            except Exception as e:
                st.error(f"Error saving files: {str(e)}")
            
            # Create output directory for indices
            index_dir = os.path.join(temp_dir.name, "indices")
            os.makedirs(index_dir, exist_ok=True)
            
            company_index_path = os.path.join(index_dir, "company_index.pkl")
            customer_index_path = os.path.join(index_dir, "customer_index.pkl")
            
            # Process button
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    # Store sections in FAISS indices
                    company_index = store_sections_in_faiss(company_path, checklist_path, company_index_path)
                    customer_index = store_sections_in_faiss(customer_path, checklist_path, customer_index_path)
                    
                    # Read checklist
                    checklist_df = pd.read_excel(checklist_path)
                    
                    # Compare sections
                    comparison_data = []
                    for _, row in checklist_df.iterrows():
                        page_name = row['PageName'].strip().lower()
                        start_marker = preprocess_text(row['StartMarker'])
                        end_marker = preprocess_text(row['EndMarker'])
                        
                        company_sections = retrieve_section(page_name, start_marker, end_marker, company_index_path)
                        customer_sections = retrieve_section(page_name, start_marker, end_marker, customer_index_path)
                        
                        if company_sections and customer_sections:
                            company_text = company_sections[0]['text']
                            customer_text = customer_sections[0]['text']
                            
                            # Get lines
                            company_lines = split_text_into_lines(company_text)
                            customer_lines = split_text_into_lines(customer_text)
                            
                            # Line-by-line comparison
                            for i, line in enumerate(company_lines):
                                if line.strip():
                                    best_match = find_best_line_match(line, customer_lines)
                                    comparison_data.append({
                                        'PageName': page_name,
                                        'SampleID': f"{page_name}_{i}",
                                        'CompanyLine': line,
                                        'CustomerLine': best_match
                                    })
                    
                    # Create comparison dataframe
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Check if Groq API key is set
                    api_key = st.session_state.get('groq_api_key', '')
                    if not api_key:
                        st.warning("Groq API key not set. Please set it in the Settings tab.")
                        st.session_state['comparison_results'] = comparison_df
                    else:
                        # Use LLM for detailed comparison
                        comparison_df = compare_dataframe(comparison_df, api_key)
                        st.session_state['comparison_results'] = comparison_df
                    
                    st.success("Document comparison completed!")
            
            # Display comparison results if available
            if st.session_state['comparison_results'] is not None:
                st.subheader("Comparison Results")
                
                # Filter options
                filter_options = st.multiselect(
                    "Filter by comparison result:",
                    ['SAME', 'DIFFERENT', 'N/A'],
                    default=['DIFFERENT']
                )
                
                filtered_df = st.session_state['comparison_results']
                if filter_options:
                    filtered_df = filtered_df[filtered_df['Comparison'].isin(filter_options)]
                
                st.dataframe(filtered_df)
                
                # Download button for CSV
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Comparison Results (CSV)",
                    data=csv,
                    file_name="document_comparison_results.csv",
                    mime="text/csv"
                )
    
    with tabs[1]:
        st.header("Structured Findings Report")
        
        if 'comparison_results' not in st.session_state or st.session_state['comparison_results'] is None:
            st.warning("Please run a document comparison first before generating findings.")
        else:
            # Get comparison results from session state
            df = st.session_state['comparison_results']
            
            # Check if we have API key for LLM categorization
            use_llm = False
            api_key = st.session_state.get('groq_api_key', '')
            
            if api_key:
                use_llm = st.checkbox("Use LLM for advanced categorization", value=True)
            
            if st.button("Generate Structured Findings Report"):
                findings_df = generate_structured_findings(df, api_key if use_llm else None)
                
                st.session_state['findings_report'] = findings_df
                
                # Display the findings
                st.subheader("Structured Findings")
                st.dataframe(findings_df)
                
                # Add download button for Excel
                if not findings_df.empty:
                    # Create Excel file with styling
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        findings_df.to_excel(writer, index=False, sheet_name='Findings')
                        
                        # Get the workbook and worksheet
                        workbook = writer.book
                        worksheet = writer.sheets['Findings']
                        
                        # Add styling
                        from openpyxl.styles import PatternFill, Border, Side, Alignment, Font
                        
                        # Header styling
                        header_fill = PatternFill(start_color='00CCFFFF', end_color='00CCFFFF', fill_type='solid')
                        for cell in worksheet[1]:
                            cell.fill = header_fill
                            cell.font = Font(bold=True)
                        
                        # Column width
                        for col in worksheet.columns:
                            max_length = 0
                            column = col[0].column_letter
                            for cell in col:
                                if cell.value:
                                    max_length = max(max_length, len(str(cell.value)))
                            adjusted_width = (max_length + 2) * 1.2
                            worksheet.column_dimensions[column].width = min(adjusted_width, 50)
                    
                    excel_data = excel_buffer.getvalue()
                    st.download_button(
                        label="Download Findings Report (Excel)",
                        data=excel_data,
                        file_name="document_comparison_findings.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            # Display existing findings if available
            if 'findings_report' in st.session_state and st.session_state['findings_report'] is not None:
                if st.session_state['findings_report'].empty:
                    st.info("No differences found that require reporting.")
                else:
                    st.subheader("Current Findings Report")
                    st.dataframe(st.session_state['findings_report'])
    
    with tabs[2]:
        st.header("Settings")
        
        st.subheader("API Configuration")
        api_key = st.text_input(
            "Groq API Key",
            value=st.session_state.get('groq_api_key', ''),
            type="password",
            help="Enter your Groq API key for LLM-powered comparison"
        )
        
        if api_key:
            st.session_state['groq_api_key'] = api_key
            st.success("API Key saved!")
        
        st.subheader("Advanced Settings")
        batch_size = st.slider(
            "LLM Processing Batch Size",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="Number of lines to process in each LLM batch"
        )
        st.session_state['batch_size'] = batch_size
        
        # Clear session data
        if st.button("Clear All Data"):
            for key in list(st.session_state.keys()):
                if key != 'groq_api_key':
                    del st.session_state[key]
            st.success("All data scleared")

if __name__ == "__main__":
    main()
