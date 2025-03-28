import streamlit as st
import pandas as pd
import re
import unicodedata
import os
import pickle
import tempfile
from docx import Document
from docx.document import Document as _Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document as LangchainDocument
from difflib import SequenceMatcher
from rapidfuzz import fuzz, process
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Set page config
st.set_page_config(page_title="Document Comparison Tool", layout="wide")

# Create temp directory for storing files
@st.cache_resource
def get_temp_dir():
    return tempfile.mkdtemp()

temp_dir = get_temp_dir()

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

def find_closest_match(target, text_list, threshold=0.65):
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

def store_sections_in_faiss(docx_path, checklist_df, progress_bar=None):
    """
    Reads the DOCX file, extracts sections based on the checklist (PageName, StartMarker, EndMarker),
    creates Document objects with metadata, and stores them in a FAISS vector store.
    """
    extracted_data = extract_text_by_sections(docx_path)
    sections = []
    
    total_rows = len(checklist_df)
    for idx, row in checklist_df.iterrows():
        if progress_bar:
            progress_bar.progress((idx + 1) / total_rows)
            
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
            
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index = FAISS.from_documents(sections, embedding_model)
    # Attach documents for later retrieval
    faiss_index.documents = sections
    return faiss_index

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

def retrieve_section(page_name, start_marker, end_marker, faiss_index):
    """
    Retrieves all documents from the FAISS index whose metadata contains the specified criteria.
    """
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

def process_batch(df_batch, groq_api_key):
    """
    Processes a batch of rows by creating a prompt, calling the LLM, and parsing the JSON response.
    """
    import json
    
    prompt = format_batch_prompt(df_batch)
    
    # Initialize LLM with the API key
    openai_llm = ChatOpenAI(
        api_key=openai_api_key,
        model_name="gpt-4o-2024-08-06"
    )
    
    with st.spinner("Processing text comparison with LLM..."):
        response = openai_llm.invoke([{"role": "user", "content": prompt}]).content
    
    if not response.strip():
        st.warning("Empty response from LLM.")
        return []
    
    try:
        # Try to clean the response in case there's any text before/after the JSON
        response_text = response.strip()
        # Find JSON start and end
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1
        if start_idx >= 0 and end_idx > 0:
            json_text = response_text[start_idx:end_idx]
            comparisons = json.loads(json_text)
        else:
            comparisons = json.loads(response_text)  # Try with the original text
    except Exception as e:
        st.error(f"Error parsing LLM response: {e}")
        st.code(response)
        comparisons = []
        
    return comparisons

def compare_dataframe(df, groq_api_key, batch_size=50):
    """
    Splits the DataFrame into batches, processes only different rows using the LLM,
    and maps the resulting comparison and difference back into the original DataFrame.
    """
    # If all rows are similar, return the original DataFrame
    if df.empty:
        return df
    
    all_comparisons = []
    
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Split DataFrame into batches
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        comparisons = process_batch(batch, groq_api_key)
        all_comparisons.extend(comparisons)
        
        # Update progress
        progress_bar.progress((i + batch_size) / len(df) if i + batch_size < len(df) else 1.0)
    
    # Clear the progress bar
    progress_bar.empty()

    # Merge the comparisons back into the original DataFrame:
    df['Comparison'] = df.index.map(
        lambda idx: next((item['comparison'] for item in all_comparisons if item['row'] == idx), "N/A"))
    df['Difference'] = df.index.map(
        lambda idx: next((item.get('difference', '') for item in all_comparisons if item['row'] == idx), ""))
    
    return df

def format_output_prompt(section_differences):
    prompt = (
        "I'm analyzing differences between company and customer documents. "
        "For each section with differences, I need you to categorize them into the following format:\n\n"
        "1. Samples affected - a list of sample IDs or 'All Samples' which are the Customer document name (without extension like .docx) if the issue affects all samples\n"
        "2. Observation Category - categorize the issue into one of these categories:\n"
        "   - 'Mismatch of content between Filed Copy and customer copy'\n"
        "   - 'Available in Filed Copy but missing in Customer Copy'\n"
        "   - Other relevant category if these don't fit\n"
        "3. Page - the section name where the issue was found\n"
        "4. Sub-category of Observation - a concise description of what specifically differs\n\n"
        "Here are the differences found per section:\n\n"
    )
    
    for section, differences in section_differences.items():
        prompt += f"Section: {section}\n"
        prompt += "Differences:\n"
        for diff in differences:
            prompt += f"- Company: {diff['CompanyLine']}\n"
            prompt += f"- Customer: {diff['CustomerLine']}\n"
            prompt += f"- Difference: {diff['Difference']}\n\n"
    
    prompt += (
        "Please format your response as a JSON array where each object has these keys:\n"
        "- 'samples_affected': String (e.g., 'All Samples' or specific IDs)\n"
        "- 'observation_category': String (the category of the issue)\n"
        "- 'page': String (the section name)\n"
        "- 'sub_category': String (detailed description of the specific issue)\n\n"
        "Return ONLY the JSON with no additional text."
    )
    
    return prompt

def generate_formatted_output(section_differences, groq_api_key):
    
    prompt = format_output_prompt(section_differences)
    
    openai_llm = ChatOpenAI(
        api_key=openai_api_key,
        model_name="gpt-4o-2024-08-06"
    )
    
    with st.spinner("Generating formatted output with LLM..."):
        response = openai_llm.invoke([{"role": "user", "content": prompt}]).content
    
    if not response.strip():
        st.warning("Empty response from format LLM.")
        return []
    
    try:
        response_text = response.strip()
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1
        if start_idx >= 0 and end_idx > 0:
            json_text = response_text[start_idx:end_idx]
            formatted_output = json.loads(json_text)
        else:
            formatted_output = json.loads(response_text)
    except Exception as e:
        st.error(f"Error parsing LLM format response: {e}")
        st.code(response)
        formatted_output = []
        
    return formatted_output

def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

# Main Streamlit app
def main():
    st.title("Document Comparison Tool")
    
    with st.sidebar:
        st.header("Upload Files")
        company_file = st.file_uploader("Upload Company Document", type=["docx"])
        customer_file = st.file_uploader("Upload Customer Document", type=["docx"])
        checklist_file = st.file_uploader("Upload Checklist File", type=["xlsx"])
        
        st.header("API Settings")
        openai_api_key = st.text_input("Enter OpenAI API Key", type="password", 
                                     value="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")
        
        compare_btn = st.button("Run Comparison", type="primary", disabled=not (company_file and customer_file and checklist_file))
    
    # Main content
    st.header("Document Comparison Results")
    
    if compare_btn:
        if not openai_api_key:
            st.error("Please enter a valid Groq API Key")
            return
            
        # Save uploaded files to temp directory
        with st.spinner("Saving uploaded files..."):
            company_path = save_uploaded_file(company_file)
            customer_path = save_uploaded_file(customer_file)
            checklist_path = save_uploaded_file(checklist_file)
            
            if not all([company_path, customer_path, checklist_path]):
                st.error("Error saving uploaded files")
                return
    
        # Read checklist
        checklist_df = pd.read_excel(checklist_path)
    
        # Create progress container
        progress_container = st.container()
    
        with progress_container:
            st.subheader("Processing Documents")
            
            # Store sections in FAISS
            st.text("Storing company sections...")
            company_progress = st.progress(0)
            company_faiss = store_sections_in_faiss(company_path, checklist_df, company_progress)
            
            st.text("Storing customer sections...")
            customer_progress = st.progress(0)
            customer_faiss = store_sections_in_faiss(customer_path, checklist_df, customer_progress)
            
            # Process each section
            st.subheader("Comparing Sections")
            final_rows = []
            
            for idx, row in checklist_df.iterrows():
                section_name = row['PageName'].strip().lower()
                start_marker = preprocess_text(row['StartMarker'])
                end_marker = preprocess_text(row['EndMarker'])
                
                st.text(f"Processing section: {section_name}")
                
                company_section = retrieve_section(section_name, start_marker, end_marker, company_faiss)
                customer_section = retrieve_section(section_name, start_marker, end_marker, customer_faiss)
                
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
            
            # Create DataFrame and perform initial filtering
            df = pd.DataFrame(final_rows).drop_duplicates()
            df["order"] = df.index
            
            st.text("Filtering similar and different rows...")
            
            same_rows = []
            different_rows = []
            
            for idx, row in df.iterrows():
                norm_company = row["CompanyLine"].lower().replace(" ", "")
                norm_customer = row["CustomerLine"].lower().replace(" ", "")
                
                if norm_company == norm_customer:
                    same_rows.append({**row, "Comparison": "SAME"})
                elif norm_company in norm_customer:
                    same_rows.append({**row, "Comparison": "SAME"})
                elif norm_customer == "":
                    same_rows.append({**row, "Comparison": "DIFFERENT", 
                                     "Difference": "Could not find similar line in customer document"})
                else:
                    different_rows.append(row)
            
            df_same = pd.DataFrame(same_rows)
            df_different = pd.DataFrame(different_rows)
            
            st.text(f"Found {df_same.shape[0]} similar rows and {df_different.shape[0]} potentially different rows")
            
            # Process different rows with LLM
            if not df_different.empty:
                st.text("Analyzing differences with LLM...")
                df_diff_compared = compare_dataframe(df_different, groq_api_key, batch_size=10)
            else:
                df_diff_compared = df_different.copy()
                
            df_final = pd.concat([df_same, df_diff_compared]).sort_values("order")
            df_final = df_final.drop(columns=["order"])
            
            section_differences = {}
            for idx, row in df_final[df_final["Comparison"] == "DIFFERENT"].iterrows():
                section = row["Section"]
                if section not in section_differences:
                    section_differences[section] = []
                section_differences[section].append({
                    "CompanyLine": row["CompanyLine"],
                    "CustomerLine": row["CustomerLine"],
                    "Difference": row["Difference"]
                })
            
            st.text("Generating formatted output...")
            formatted_output = generate_formatted_output(section_differences, groq_api_key)
            
            if formatted_output:
                output_df = pd.DataFrame(formatted_output)
                output_df.columns = [
                    "Samples affected", 
                    "Observation - Category", 
                    "Page", 
                    "Sub-category of Observation"
                ]
            else:
                output_df = pd.DataFrame(columns=[
                    "Samples affected", 
                    "Observation - Category", 
                    "Page", 
                    "Sub-category of Observation"
                ])
            
            progress_container.empty()
            
            st.subheader("Comparison Results - Raw Differences")
            
            st.sidebar.header("Raw Difference Filters")
            section_filter = st.sidebar.multiselect(
                "Filter by Section",
                options=df_final["Section"].unique(),
                default=df_final["Section"].unique()
            )
            
            comparison_filter = st.sidebar.multiselect(
                "Filter by Comparison",
                options=df_final["Comparison"].unique(),
                default=df_final["Comparison"].unique()
            )
            
            filtered_df = df_final[
                df_final["Section"].isin(section_filter) & 
                df_final["Comparison"].isin(comparison_filter)
            ]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Lines", len(df_final))
                st.metric("Similar Lines", len(df_final[df_final["Comparison"] == "SAME"]))
            with col2:
                st.metric("Different Lines", len(df_final[df_final["Comparison"] == "DIFFERENT"]))
                st.metric("Filtered Results", len(filtered_df))
            
            st.dataframe(
                filtered_df.style.apply(
                    lambda row: ['background-color: #ffcccc' if row['Comparison'] == 'DIFFERENT' else 'background-color: #ccffcc' for _ in row], 
                    axis=1
                ),
                height=400
            )
            
            st.subheader("Formatted Output")
            st.dataframe(
                output_df.style.apply(
                    lambda _: ['background-color: #e6f3ff' for _ in range(len(output_df.columns))], 
                    axis=1
                ),
                height=300
            )
            
            st.text("Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                excel_path = os.path.join(temp_dir, "comparison_results.xlsx")
                filtered_df.to_excel(excel_path, index=False)
                
                with open(excel_path, "rb") as f:
                    st.download_button(
                        label="Download Raw Results",
                        data=f,
                        file_name="document_comparison_raw.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col2:
                formatted_path = os.path.join(temp_dir, "formatted_results.xlsx")
                output_df.to_excel(formatted_path, index=False)
                
                with open(formatted_path, "rb") as f:
                    st.download_button(
                        label="Download Formatted Results",
                        data=f,
                        file_name="document_comparison_formatted.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

if __name__ == "__main__":
    main()
