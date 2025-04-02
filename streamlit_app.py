import time
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
import tiktoken
import json

st.set_page_config(page_title="Document Comparison Tool", layout="wide")

# Use session state to store file paths and results
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
    
if 'raw_results' not in st.session_state:
    st.session_state.raw_results = None
    
if 'formatted_results' not in st.session_state:
    st.session_state.formatted_results = None
    
if 'has_results' not in st.session_state:
    st.session_state.has_results = False

def preprocess_text(text):
    """Cleans text: removes text in < > brackets, extra spaces, normalizes, lowercases, and removes punctuation (except periods)."""
    if not text:
        return ""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^\w\s.]", "", text)
    return text

def count_tokens(text, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def truncate_text(text, max_tokens=100):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:max_tokens])

def chunk_dataframe_with_token_reduction(df, max_tokens=5000):
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for _, row in df.iterrows():
        row_text = f"Row {row.name}: {row['CompanyLine']} | {row['CustomerLine']}"
        row_tokens = count_tokens(row_text)
        
        if current_tokens + row_tokens > max_tokens:
            chunks.append(pd.DataFrame(current_chunk))
            current_chunk = []
            current_tokens = 0
        
        current_chunk.append(row)
        current_tokens += row_tokens
    
    if current_chunk:
        chunks.append(pd.DataFrame(current_chunk))
    
    return chunks
    
def iter_block_items(parent):
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
    faiss_index.documents = sections
    return faiss_index

def split_text_into_lines(text):
    return re.split(r'\n|\||\. ', text)

def find_best_line_match(target, lines):
    target = target.lower().strip()

    if len(target) < 10:
        custom_threshold = 90
    elif len(target) < 50:
        custom_threshold = 75
    else:
        custom_threshold = 65

    best = process.extractOne(
        query=target,
        choices=lines,
        scorer=fuzz.ratio,
        score_cutoff=custom_threshold
    )
    if best:
        return best[0]
    else:
        return ""

def extract_content_within_markers(text, start_marker, end_marker):

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
            start_idx, end_idx = end_idx, start_idx
        return "\n".join(lines[start_idx:end_idx + 1])
    except ValueError:
        st.warning("Error finding marker indices; returning full text.")
        return text

def retrieve_section(page_name, start_marker, end_marker, faiss_index):
    """
    Retrieves all documents from the FAISS index whose metadata contains the specified criteria.
    """
    matching_sections = []
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

def process_batch(df_batch, openai_api_key):
    """
    Process a batch of rows with improved prompt construction and error handling.
    Non-recursive implementation.
    """
    # Prepare the prompt with fixed truncation
    prompt_lines = []
    for idx, row in df_batch.iterrows():
        # Aggressive truncation to save tokens
        company_text = truncate_text(str(row['CompanyLine']), max_tokens=100)
        customer_text = truncate_text(str(row['CustomerLine']), max_tokens=100)
        
        prompt_lines.append(f"Row {idx}:")
        prompt_lines.append(f"Company: {company_text}")
        prompt_lines.append(f"Customer: {customer_text}")
        prompt_lines.append("---")
    
    prompt_text = "\n".join(prompt_lines)
    prompt_text += (
        "\n\nCompare the above rows line by line. "
        "For each row, output a JSON object with keys 'row' (the row number), 'comparison' (SAME or DIFFERENT), and "
        "'difference' (if different, a brief explanation with specific difference). Return a JSON list of these objects. Return ONLY the JSON."
    )
    
    total_tokens = count_tokens(prompt_text)
    st.info(f"Batch Prompt Tokens: {total_tokens}")
    
    # If the prompt is still too large, we need to further split the batch
    if total_tokens > 4000:
        st.warning(f"Prompt too large ({total_tokens} tokens). Splitting batch.")
        
        # Instead of recursion, split the batch in half and process each separately
        half_size = len(df_batch) // 2
        if half_size == 0:  # Can't split further
            st.error("Cannot split batch further. Individual rows may be too large.")
            # Try with extreme truncation for this batch
            for idx, row in df_batch.iterrows():
                df_batch.at[idx, 'CompanyLine'] = truncate_text(str(row['CompanyLine']), max_tokens=50)
                df_batch.at[idx, 'CustomerLine'] = truncate_text(str(row['CustomerLine']), max_tokens=50)
            return process_batch(df_batch, openai_api_key)
        
        # Process each half separately
        first_half_results = process_batch(df_batch.iloc[:half_size], openai_api_key)
        second_half_results = process_batch(df_batch.iloc[half_size:], openai_api_key)
        
        # Combine results
        return first_half_results + second_half_results
    
    # If we're here, the prompt is within token limits
    openai_llm = ChatOpenAI(
        api_key=openai_api_key,
        model_name="gpt-3.5-turbo",
        max_tokens=1000
    )
    
    with st.spinner("Processing text comparison with LLM..."):
        try:
            response = openai_llm.invoke([
                {"role": "user", "content": prompt_text}
            ]).content
        except Exception as e:
            st.error(f"LLM Processing Error: {e}")
            return []
    
    if not response.strip():
        st.warning("Empty response from LLM.")
        return []
    
    try:
        response_text = response.strip()
        
        try:
            comparisons = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                comparisons = json.loads(json_text)
            else:
                import ast
                try:
                    comparisons = ast.literal_eval(response_text)
                except:
                    st.error("Could not parse response as JSON or Python literal")
                    comparisons = []
    
    except Exception as e:
        st.error(f"Comprehensive JSON Parsing Error: {e}")
        st.code(response)
        comparisons = []
    
    return comparisons

def chunk_dataframe_into_batches(df, batch_size=50):
    """
    Split a dataframe into chunks of specified batch size.
    Returns a list of dataframe chunks.
    """
    return [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]

def chunk_dataframe_with_token_reduction(df, max_tokens=3000):
    """
    Split a dataframe into chunks based on token count.
    Returns a list of dataframe chunks.
    """
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for _, row in df.iterrows():
        row_text = f"Row {row.name}: {row['CompanyLine']} | {row['CustomerLine']}"
        row_tokens = count_tokens(row_text)
        
        # If this single row exceeds max tokens, truncate it
        if row_tokens > max_tokens:
            row['CompanyLine'] = truncate_text(row['CompanyLine'], max_tokens // 2)
            row['CustomerLine'] = truncate_text(row['CustomerLine'], max_tokens // 2)
            row_text = f"Row {row.name}: {row['CompanyLine']} | {row['CustomerLine']}"
            row_tokens = count_tokens(row_text)
        
        if current_tokens + row_tokens > max_tokens and current_chunk:
            chunks.append(pd.DataFrame(current_chunk))
            current_chunk = []
            current_tokens = 0
        
        current_chunk.append(row)
        current_tokens += row_tokens
    
    if current_chunk:
        chunks.append(pd.DataFrame(current_chunk))
    
    return chunks

def compare_dataframe(df, openai_api_key, batch_size=50, rate_limit_delay=20):
    if df.empty:
        return df
    
    # Limit to 100 rows for testing/demo (optional)
    df = df.head(100)
    
    progress_bar = st.progress(0)
    
    # Split the dataframe into fixed-size batches
    df_batches = chunk_dataframe_into_batches(df, batch_size=batch_size)
    
    all_comparisons = []
    max_retries = 3
    
    for batch_idx, batch in enumerate(df_batches):
        st.text(f"Processing batch {batch_idx+1} of {len(df_batches)} ({len(batch)} rows)")
        
        # Initialize a new batch_comparisons list
        batch_comparisons = []
        
        # Further chunk each batch based on token count
        token_chunks = chunk_dataframe_with_token_reduction(batch, max_tokens=3000)
        st.info(f"Split into {len(token_chunks)} token-based chunks")
        
        for token_chunk_idx, token_chunk in enumerate(token_chunks):
            for attempt in range(max_retries):
                try:
                    if token_chunk_idx > 0 or batch_idx > 0:
                        delay_time = max(5, min(20, rate_limit_delay // len(token_chunks)))
                        st.info(f"Waiting {delay_time} seconds to avoid rate limiting...")
                        time.sleep(delay_time)
                    
                    st.text(f"Processing token chunk {token_chunk_idx+1} of {len(token_chunks)} ({len(token_chunk)} rows)")
                    chunk_comparisons = process_batch(token_chunk, openai_api_key)
                    batch_comparisons.extend(chunk_comparisons)
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.warning(f"Attempt {attempt + 1} failed for chunk {token_chunk_idx}. Retrying... Error: {e}")
                        time.sleep(rate_limit_delay)
                    else:
                        st.error(f"Failed after {max_retries} attempts for chunk {token_chunk_idx}. Error: {e}")
        
        all_comparisons.extend(batch_comparisons)
        progress_bar.progress((batch_idx + 1) / len(df_batches))
    
    progress_bar.empty()

    # Map the comparison results back to the dataframe
    df['Comparison'] = df.index.map(
        lambda idx: next((item['comparison'] for item in all_comparisons if str(item.get('row', '')) == str(idx)), "N/A"))
    df['Difference'] = df.index.map(
        lambda idx: next((item.get('difference', '') for item in all_comparisons if str(item.get('row', '')) == str(idx)), ""))
    
    return df

def format_output_prompt(section_differences):
    """
    Create a prompt for the second LLM call to format the differences.
    Uses a non-recursive approach for chunking.
    """
    # Count total differences
    total_differences = sum(len(diffs) for diffs in section_differences.values())
    
    # If small enough, process all at once
    if total_differences <= 20:
        return [create_single_format_prompt(section_differences)]
    
    # Otherwise, chunk by sections
    prompts = []
    current_sections = {}
    current_diff_count = 0
    max_diff_per_prompt = 15
    
    for section, differences in section_differences.items():
        # If adding this section would exceed our limit and we already have content
        if current_diff_count + len(differences) > max_diff_per_prompt and current_diff_count > 0:
            # Create a prompt from current accumulated sections
            prompts.append(create_single_format_prompt(current_sections))
            current_sections = {}
            current_diff_count = 0
        
        # If a single section has too many differences, split it
        if len(differences) > max_diff_per_prompt:
            for i in range(0, len(differences), max_diff_per_prompt):
                chunk = differences[i:i+max_diff_per_prompt]
                prompts.append(create_single_format_prompt({section: chunk}))
        else:
            current_sections[section] = differences
            current_diff_count += len(differences)
    
    # Add any remaining sections
    if current_diff_count > 0:
        prompts.append(create_single_format_prompt(current_sections))
    
    return prompts

def create_single_format_prompt(section_differences):
    """
    Create a single format prompt for given section differences.
    """
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
            prompt += f"- Company: {truncate_text(diff['CompanyLine'], 75)}\n"
            prompt += f"- Customer: {truncate_text(diff['CustomerLine'], 75)}\n"
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

def chunk_section_differences(section_differences):
    """
    Split large section differences into multiple prompts.
    Returns a list of prompts.
    """
    prompts = []
    current_sections = {}
    current_diff_count = 0
    max_diff_per_prompt = 20  # Adjust based on testing
    
    for section, differences in section_differences.items():
        if current_diff_count + len(differences) > max_diff_per_prompt and current_diff_count > 0:
            # Create a prompt from current accumulated sections
            prompts.append(format_output_prompt(current_sections))
            current_sections = {}
            current_diff_count = 0
        
        current_sections[section] = differences
        current_diff_count += len(differences)
    
    # Add any remaining sections
    if current_diff_count > 0:
        prompts.append(format_output_prompt(current_sections))
    
    return prompts

def generate_formatted_output(section_differences, openai_api_key):
    """
    Generate formatted output from section differences.
    Non-recursive implementation.
    """
    prompts = format_output_prompt(section_differences)
    all_formatted_outputs = []
    
    for i, prompt in enumerate(prompts):
        st.text(f"Processing format chunk {i+1} of {len(prompts)}")
        
        openai_llm = ChatOpenAI(
            api_key=openai_api_key,
            model_name="gpt-3.5-turbo"
        )
        
        with st.spinner(f"Generating formatted output chunk {i+1}/{len(prompts)} with LLM..."):
            response = openai_llm.invoke([{"role": "user", "content": prompt}]).content
        
        if not response.strip():
            st.warning(f"Empty response from format LLM in chunk {i+1}.")
            continue
        
        try:
            response_text = response.strip()
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            if start_idx >= 0 and end_idx > 0:
                json_text = response_text[start_idx:end_idx]
                formatted_output = json.loads(json_text)
            else:
                formatted_output = json.loads(response_text)
                
            all_formatted_outputs.extend(formatted_output)
            
            # Add delay between chunks if needed
            if i < len(prompts) - 1:
                time.sleep(10)
                
        except Exception as e:
            st.error(f"Error parsing LLM format response in chunk {i+1}: {e}")
            st.code(response)
    
    return all_formatted_outputs

def process_format_prompt(prompt, openai_api_key):
    """
    Process a single formatting prompt.
    """
    openai_llm = ChatOpenAI(
        api_key=openai_api_key,
        model_name="gpt-3.5-turbo"
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
        file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def run_comparison():
    # Get values from session state
    company_file = st.session_state.company_file
    customer_file = st.session_state.customer_file
    checklist_file = st.session_state.checklist_file
    openai_api_key = st.session_state.openai_api_key
    
    if not openai_api_key:
        st.error("Please enter a valid OpenAI API Key")
        return
        
    with st.spinner("Saving uploaded files..."):
        company_path = save_uploaded_file(company_file)
        customer_path = save_uploaded_file(customer_file)
        checklist_path = save_uploaded_file(checklist_file)
        
        if not all([company_path, customer_path, checklist_path]):
            st.error("Error saving uploaded files")
            return

    checklist_df = pd.read_excel(checklist_path)

    progress_container = st.container()

    with progress_container:
        st.subheader("Processing Documents")
        
        st.text("Storing company sections...")
        company_progress = st.progress(0)
        company_faiss = store_sections_in_faiss(company_path, checklist_df, company_progress)
        
        st.text("Storing customer sections...")
        customer_progress = st.progress(0)
        customer_faiss = store_sections_in_faiss(customer_path, checklist_df, customer_progress)
        
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
                different_rows.append({**row, "Comparison": "DIFFERENT", 
                                     "Difference": "Could not find similar line in customer document"})
            else:
                different_rows.append(row)
        
        df_same = pd.DataFrame(same_rows)
        df_different = pd.DataFrame(different_rows)
        
        st.text(f"Found {df_same.shape[0]} similar rows and {df_different.shape[0]} potentially different rows")
        if df_different.shape[0] > 0:
            st.text("Comparing potentially different rows with LLM...")
    
    # Print the type before comparison
            st.text(f"Type of df_different before comparison: {type(df_different)}")
    
            result = compare_dataframe(df_different, openai_api_key)
    
    # Print the type after comparison
            st.text(f"Type of result after comparison: {type(result)}")
    
    # Check if it's a DataFrame or something else
            if isinstance(result, pd.DataFrame):
                df_different = result
            else:
                st.error(f"compare_dataframe returned {type(result)} instead of DataFrame")
        # Try to convert it to a DataFrame if possible
                try:
                    if isinstance(result, dict):
                        df_different = pd.DataFrame([result])
                    elif isinstance(result, list):
                        df_different = pd.DataFrame(result)
                    else:
                # Fall back to original df_different
                        st.warning("Could not convert result to DataFrame, using original df_different")
                except Exception as e:
                    st.error(f"Error converting to DataFrame: {e}")
            # Keep the original df_different
    
    # Combine the results
            df_results = pd.concat([df_same, df_different]).sort_values("order")
    
    # Store the raw results in session state
            st.session_state.raw_results = df_results
    
    # Process the formatted output
            st.text("Generating formatted output...")
            section_differences = {}
    
            for section_name, section_df in df_results[df_results["Comparison"] == "DIFFERENT"].groupby("Section"):
                section_differences[section_name] = section_df.to_dict('records')
    
            if section_differences:
                formatted_output = generate_formatted_output(section_differences, openai_api_key)
                st.session_state.formatted_results = formatted_output
                st.session_state.has_results = True
            else:
                st.info("No differences found between documents.")
        else:
            st.info("No differences found between documents.")
            st.session_state.raw_results = df_same
            st.session_state.has_results = True

# UI Layout
st.title("Document Comparison Tool")

with st.expander("Settings", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.file_uploader("Upload Company Document (DOCX)", type=["docx"], key="company_file")
        st.file_uploader("Upload Customer Document (DOCX)", type=["docx"], key="customer_file")
    
    with col2:
        st.file_uploader("Upload Checklist (Excel)", type=["xlsx", "xls"], key="checklist_file")
        st.text_input("OpenAI API Key", type="password", key="openai_api_key")

if (st.session_state.company_file is not None and 
    st.session_state.customer_file is not None and 
    st.session_state.checklist_file is not None):
    
    st.button("Compare Documents", on_click=run_comparison)

# Display results if available
if st.session_state.has_results:
    st.header("Comparison Results")
    
    if st.session_state.raw_results is not None:
        with st.expander("Raw Results", expanded=False):
            st.dataframe(st.session_state.raw_results)
        
        # Count differences
        diff_count = len(st.session_state.raw_results[st.session_state.raw_results["Comparison"] == "DIFFERENT"])
        st.info(f"Found {diff_count} differences between documents.")
        
        # Display formatted results
        if st.session_state.formatted_results:
            st.subheader("Formatted Differences")
            for item in st.session_state.formatted_results:
                st.write("---")
                st.write(f"**Samples Affected:** {item.get('samples_affected', 'Not specified')}")
                st.write(f"**Category:** {item.get('observation_category', 'Not categorized')}")
                st.write(f"**Page:** {item.get('page', item.get('Page', 'Not specified'))}")
                st.write(f"**Issue:** {item.get('sub_category', 'Not specified')}")
            
            # Download option
            formatted_df = pd.DataFrame(st.session_state.formatted_results)
            csv = formatted_df.to_csv(index=False)
            
            st.download_button(
                label="Download Formatted Results (CSV)",
                data=csv,
                file_name="document_comparison_results.csv",
                mime="text/csv"
            )
