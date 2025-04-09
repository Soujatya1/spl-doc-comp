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
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.docstore.document import Document as LangchainDocument
from difflib import SequenceMatcher
from rapidfuzz import fuzz, process
from langchain_groq import ChatGroq
import tiktoken
import json
from io import BytesIO
from openpyxl import Workbook
from openpyxl.styles import Alignment, PatternFill, Font, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

st.set_page_config(page_title="Document Comparison Tool", layout="wide")

if 'comparison_completed' not in st.session_state:
    st.session_state.comparison_completed = False
if 'company_faiss' not in st.session_state:
    st.session_state.company_faiss = None
if 'customer_faiss' not in st.session_state:
    st.session_state.customer_faiss = None
if 'final_df' not in st.session_state:
    st.session_state.final_df = None
if 'output_df' not in st.session_state:
    st.session_state.output_df = None
if 'processing_step' not in st.session_state:
    st.session_state.processing_step = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

temp_dir = st.session_state.temp_dir

def preprocess_text(text):
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

@st.cache_data
def store_sections_in_faiss(docx_path, checklist_df):
    progress_bar = st.progress(0)
    extracted_data = extract_text_by_sections(docx_path)
    sections = []
    
    total_rows = len(checklist_df)
    for idx, row in checklist_df.iterrows():
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
    
    embedding_model = FastEmbedEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    faiss_index = FAISS.from_documents(sections, embedding_model)
    faiss_index.documents = sections
    progress_bar.empty()
    return faiss_index

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

def extract_customer_number(filename):
    if '__' in filename:
        number_part = filename.split('__')[0]
        return number_part
    return os.path.splitext(filename)[0]

def create_direct_comparison_prompt(sections_data, customer_number="All Samples"):
    prompt = (
        "You are a document comparison expert. You will analyze differences between company (Filed Copy) and customer document sections "
        "and produce a consolidated report. The format of your response is critical.\n\n"
        
        "IMPORTANT: Create a table with EXACTLY these column headers:\n"
        "1. 'Samples affected' - Should always be '{customer_number}'\n"
        "2. 'Observation - Category' - Must be one of these two categories:\n"
        "   - 'Mismatch of content between Filed Copy and customer copy': Use ONLY when similar content exists in BOTH documents but with differences\n"
        "   - 'Available in Filed Copy but missing in Customer Copy': Use ONLY when content exists in Filed copy but is COMPLETELY ABSENT from customer copy\n"
        "3. 'Page' - The specific document section(s) EXACTLY affected as per the below:\n"
        "   - 'Preamble'\n"
        "   - 'Terms and conditions'\n"
        "   - 'Annexure 1'\n"
        "   - 'Annexure AA'\n"
        "   - 'Forwarding Letter'\n"
        "   - 'Ombudsman Page'\n"
        "   - 'Schedule'\n"
        "4. 'Sub-category of Observation' - Comprehensive summary of ALL differences for that page\n\n"
        
        "CLEAR DISTINCTION BETWEEN CATEGORIES:\n"
        "- 'Mismatch of content': Use ONLY when the same type of information exists in both documents but differs in details. Example: Company copy mentions '30-day period' while filed copy shows '15-day period'.\n"
        "- 'Available in Filed Copy but missing': Use ONLY when information appears in company document but is completely absent from customer document. Example: Company document has a section on 'Cancellation Policy' that doesn't appear at all in customer document.\n\n"
        
        "Here are the sections to compare:\n\n"
    )
    
    for section_name, data in sections_data.items():
        prompt += f"SECTION: {section_name}\n"
        prompt += f"COMPANY COPY:\n{truncate_text(data['company_text'], 500)}\n\n"
        prompt += f"FILED COPY:\n{truncate_text(data['customer_text'], 500)}\n\n"
        prompt += "---\n\n"
    
    prompt += (
        "INSTRUCTIONS:\n"
        "1. Compare the content of each section carefully.\n"
        "2. Identify meaningful differences - ignore minor formatting, spacing, or punctuation differences.\n"
        "3. First determine if content is missing entirely (use 'Available in Filed Copy but missing') OR if similar content exists with differences (use 'Mismatch of content').\n"
        "4. Create ONE ROW PER PAGE in the output table, grouping by category.\n"
        "5. For each page, provide a COMPREHENSIVE summary of ALL differences in the 'Sub-category of Observation' column.\n"
        "6. Group all pages with 'Mismatch of content between Filed Copy and customer copy' together first, "
        "   then group all pages with 'Available in Filed Copy but missing in Customer Copy'.\n\n"
        
        "CRITICAL CATEGORIZATION RULES:\n"
        "- If similar content/information exists in both copies but with differences -> 'Mismatch of content'\n"
        "- If content/information exists in Filed copy but is completely absent in Customer copy -> 'Available in Filed Copy but missing'\n"
        "- Example of 'Mismatch': Both documents mention grace period but one says 30 days and other says 15 days\n"
        "- Example of 'Available but missing': Filed copy has contact details section that doesn't exist at all in customer copy\n\n"
        
        "Return your analysis as a JSON array where each object has these keys:\n"
        "- 'Samples affected': Always set to '{customer_number}'\n"
        "- 'Observation - Category': One of the two categories listed above\n"
        "- 'Page': One of the page categories mentioned above\n"
        "- 'Sub-category of Observation': Comprehensive summary of ALL differences for that page\n\n"
        
        "Example structure:\n"
        "[{\n"
        "  \"Samples affected\": \"0614054616\",\n"
        "  \"Observation - Category\": \"Mismatch of content between Filed Copy and customer copy\",\n"
        "  \"Page\": \"Forwarding Letter\",\n"
        "  \"Sub-category of Observation\": \"1. Company copy mentions a 30-day response period while filed copy states 15 days. "
        "2. Company copy includes different phone number than filed copy.\"\n"
        "},\n"
        "{\n"
        "  \"Samples affected\": \"0614054616\",\n"
        "  \"Observation - Category\": \"Available in Filed Copy but missing in Customer Copy\",\n"
        "  \"Page\": \"Terms and conditions\",\n"
        "  \"Sub-category of Observation\": \"1. Section 4.2 on cancellation policy present in filed copy is completely missing from customer copy. "
        "2. Appendix B with fee schedule present in filed copy doesn't exist in customer copy.\"\n"
        "}]\n\n"
        
        "IMPORTANT: Create exactly ONE row for each affected page within each category (do not create multiple rows for the same page in the same category). "
        "Make sure each 'Page' appears only once under each 'Observation - Category'. "
        "Group rows by 'Observation - Category'."
    )
    
    return prompt

def chunk_sections_by_token_count(sections_data, max_tokens=6000):
    chunks = []
    current_chunk = {}
    current_tokens = 0
    
    for section_name, data in sections_data.items():
        section_text = f"SECTION: {section_name}\nCOMPANY COPY:\n{data['company_text']}\n\nFILED COPY:\n{data['customer_text']}\n\n"
        section_tokens = count_tokens(section_text)
        
        if section_tokens > max_tokens:
            data_copy = data.copy()
            data_copy['company_text'] = truncate_text(data['company_text'], max_tokens // 2)
            data_copy['customer_text'] = truncate_text(data['customer_text'], max_tokens // 2)
            truncated_section_text = f"SECTION: {section_name}\nCOMPANY COPY:\n{data_copy['company_text']}\n\nFILED COPY:\n{data_copy['customer_text']}\n\n"
            section_tokens = count_tokens(truncated_section_text)
            data = data_copy
        
        if current_tokens + section_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = {}
            current_tokens = 0
        
        current_chunk[section_name] = data
        current_tokens += section_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def organize_comparison_results(output_df):
    if output_df.empty:
        return output_df
        
    category_order = [
        "Mismatch of content between Filed Copy and customer copy",
        "Available in Filed Copy but missing in Customer Copy"
    ]
    
    page_order = [
        "Forwarding Letter", 
        "Schedule", 
        "Terms and conditions",
        "Ombudsman Page",
        "Annexure AA",
        "Annexure 1",
        "Preamble"
    ]
    
    output_df['category_order'] = output_df['Observation - Category'].apply(
        lambda x: category_order.index(x) if x in category_order else 999
    )
    
    output_df['page_order'] = output_df['Page'].apply(
        lambda x: page_order.index(x) if x in page_order else 999
    )
    
    sorted_df = output_df.sort_values(['category_order', 'page_order'])
    
    sorted_df = sorted_df.drop(['category_order', 'page_order'], axis=1)
    
    return sorted_df

@st.cache_data
def direct_document_comparison(sections_data, groq_api_key, customer_number="All Samples"):
    section_chunks = chunk_sections_by_token_count(sections_data, max_tokens=6000)
    st.info(f"Split comparison into {len(section_chunks)} chunks based on token limits")
    
    all_results = []
    
    for i, chunk in enumerate(section_chunks):
        st.text(f"Processing chunk {i+1} of {len(section_chunks)}")
        
        prompt = create_direct_comparison_prompt(chunk, customer_number)
        
        groq_llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama3-8b-8192",
            max_tokens=2000
        )
        
        with st.spinner(f"Analyzing document differences (chunk {i+1}/{len(section_chunks)})..."):
            try:
                response = groq_llm.invoke([{"role": "user", "content": prompt}]).content
            except Exception as e:
                st.error(f"LLM Processing Error in chunk {i+1}: {e}")
                continue
        
        if not response.strip():
            st.warning(f"Empty response from LLM for chunk {i+1}.")
            continue
        
        try:
            response_text = response.strip()
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                chunk_results = json.loads(json_text)
            else:
                try:
                    chunk_results = json.loads(response_text)
                except:
                    import ast
                    try:
                        chunk_results = ast.literal_eval(response_text)
                    except:
                        st.error(f"Could not parse response as JSON in chunk {i+1}")
                        st.code(response)
                        chunk_results = []
            
            all_results.extend(chunk_results)
            
            if i < len(section_chunks) - 1:
                time.sleep(10)
                
        except Exception as e:
            st.error(f"Error parsing LLM response in chunk {i+1}: {e}")
            st.code(response)
    
    if all_results:
        for result in all_results:
            for key in list(result.keys()):
                if key.lower() == "samples affected" or key.lower() == "samples_affected":
                    result["Samples affected"] = result.pop(key)
                elif key.lower() == "observation - category" or key.lower() == "observation_category":
                    result["Observation - Category"] = result.pop(key)
                elif key.lower() == "page":
                    result["Page"] = result.pop(key)
                elif key.lower() == "sub-category of observation" or key.lower() == "sub_category":
                    result["Sub-category of Observation"] = result.pop(key)
    
        temp_df = pd.DataFrame(all_results)
    
        required_columns = ["Samples affected", "Observation - Category", "Page", "Sub-category of Observation"]
        for col in required_columns:
            if col not in temp_df.columns:
                temp_df[col] = ""
        
        aggregated_results = []
        
        category_order = [
            "Mismatch of content between Filed Copy and customer copy",
            "Available in Filed Copy but missing in Customer Copy"
        ]
        
        for category in category_order:
            category_df = temp_df[temp_df["Observation - Category"] == category]
            
            if category_df.empty:
                continue
                
            for page in category_df["Page"].unique():
                page_rows = category_df[category_df["Page"] == page]
                
                all_observations = []
                for idx, row in page_rows.iterrows():
                    observation = row["Sub-category of Observation"]
                    if observation not in all_observations:
                        all_observations.append(observation)
                
                combined_observations = ""
                for i, obs in enumerate(all_observations, 1):
                    if not re.match(r'^\d+\.', obs.strip()):
                        combined_observations += f"{i}. {obs}\n"
                    else:
                        combined_observations += f"{obs}\n"
                
                aggregated_results.append({
                    "Samples affected": customer_number,
                    "Observation - Category": category,
                    "Page": page,
                    "Sub-category of Observation": combined_observations.strip()
                })
        
        output_df = pd.DataFrame(aggregated_results)
        
        if not output_df.empty:
            output_df = output_df[required_columns]
    
            output_df = organize_comparison_results(output_df)
    
        return output_df
    else:
        return pd.DataFrame(columns=["Samples affected", "Observation - Category", "Page", "Sub-category of Observation"])

def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def run_direct_comparison(company_path, customer_path, checklist_path, groq_api_key):
    progress_container = st.empty()
    output_df = None
    
    try:
        customer_filename = os.path.basename(customer_path)
        customer_number = extract_customer_number(customer_filename)
        
        with progress_container.container():
            st.subheader("Step 1: Loading Checklist")
            checklist_df = pd.read_excel(checklist_path)
            st.session_state.processing_step = "load_checklist"
            st.success("✅ Checklist loaded successfully")

        with progress_container.container():
            st.subheader("Step 2: Processing Company Document")
            st.text("Storing company sections...")
            st.session_state.company_faiss = store_sections_in_faiss(company_path, checklist_df)
            st.session_state.processing_step = "company_faiss"
            st.success("✅ Company document processed successfully")

        with progress_container.container():
            st.subheader("Step 3: Processing Customer Document")
            st.text("Storing customer sections...")
            st.session_state.customer_faiss = store_sections_in_faiss(customer_path, checklist_df)
            st.session_state.processing_step = "customer_faiss"
            st.success("✅ Customer document processed successfully")

        with progress_container.container():
            st.subheader("Step 4: Extracting Document Sections")
            sections_data = {}
            
            for idx, row in checklist_df.iterrows():
                section_name = row['PageName'].strip()
                start_marker = preprocess_text(row['StartMarker'])
                end_marker = preprocess_text(row['EndMarker'])
                
                st.text(f"Processing section: {section_name}")
                
                company_section = retrieve_section(section_name, start_marker, end_marker, st.session_state.company_faiss)
                customer_section = retrieve_section(section_name, start_marker, end_marker, st.session_state.customer_faiss)
                
                if not company_section or not customer_section:
                    st.warning(f"Could not retrieve sections for {section_name}")
                    continue
                
                sections_data[section_name] = {
                    "company_text": company_section[0]["text"],
                    "customer_text": customer_section[0]["text"]
                }
            
            st.session_state.processing_step = "extract_sections"
            st.success(f"✅ Extracted {len(sections_data)} document sections")

        with progress_container.container():
            st.subheader("Step 5: Analyzing Document Differences")
            output_df = direct_document_comparison(sections_data, groq_api_key, customer_number)
            
            if not output_df.empty:
                category_order = [
                    "Mismatch of content between Filed Copy and customer copy",
                    "Available in Filed Copy but missing in Customer Copy"
                ]
                
                for category in category_order:
                    if category not in output_df["Observation - Category"].values:
                        placeholder_row = {
                            "Samples affected": customer_number,
                            "Observation - Category": category,
                            "Page": "",
                            "Sub-category of Observation": ""
                        }
                        placeholder_df = pd.DataFrame([placeholder_row])
                        output_df = pd.concat([output_df, placeholder_df], ignore_index=True)
                
                output_df = output_df[output_df["Page"] != ""]
            
            output_df = organize_comparison_results(output_df)
            
            st.session_state.output_df = output_df
            
            if not output_df.empty:
                diff_count = len(output_df)
                st.session_state.final_df = pd.DataFrame({
                    "Section": output_df["Page"].tolist(),
                    "Comparison": ["DIFFERENT"] * diff_count,
                    "Difference": output_df["Sub-category of Observation"].tolist()
                })
            else:
                st.session_state.final_df = pd.DataFrame(columns=["Section", "Comparison", "Difference"])
                
            st.session_state.processing_step = "complete"
            st.session_state.comparison_completed = True
            st.success("✅ Document analysis completed")

        progress_container.empty()
        return True
        
    except Exception as e:
        if output_df is not None:
            st.session_state.output_df = output_df
        
        progress_container.error(f"Error occurred during processing: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False

def display_results():
    st.header("Document Comparison Results")
    
    if st.session_state.comparison_completed:
        if st.button("Start New Comparison"):
            st.session_state.comparison_completed = False
            st.session_state.final_df = None
            st.session_state.output_df = None
            st.rerun()
    
    if st.session_state.final_df is not None:
        df_final = st.session_state.final_df
        col1, col2 = st.columns(2)
        #with col1:
            #st.metric("Total Sections Analyzed", len(st.session_state.output_df) if not st.session_state.output_df.empty else 0)
        #with col2:
            #st.metric("Sections with Differences", len(df_final[df_final["Comparison"] == "DIFFERENT"]))
        
        st.subheader("Comparison Report")
        if st.session_state.output_df is not None and not st.session_state.output_df.empty:
            st.markdown("""
            <style>
            .comparison-table {
                width: 100%;
                text-align: left;
                border-collapse: collapse;
            }
            .comparison-table th {
                background-color: #87CEEB;
                color: black;
                font-weight: bold;
                padding: 8px;
                border: 1px solid #ddd;
            }
            .comparison-table td {
                padding: 8px;
                border: 1px solid #ddd;
            }
            .comparison-table tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            </style>
            """, unsafe_allow_html=True)
            
            html_table = st.session_state.output_df.to_html(
                classes="comparison-table", 
                index=False,
                escape=False
            )
            st.markdown(html_table, unsafe_allow_html=True)
            
            from io import BytesIO
            from openpyxl import Workbook
            from openpyxl.styles import Alignment, PatternFill, Font, Border, Side
            
            buffer = BytesIO()
            
            wb = Workbook()
            ws = wb.active
            ws.title = "Comparison Report"
            
            headers = ["Samples affected", "Observation - Category", "Page", "Sub-category of Observation"]
            for col_idx, header in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col_idx)
                cell.value = header
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="87CEEB", end_color="87CEEB", fill_type="solid")
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = Border(
                    left=Side(style="thin"),
                    right=Side(style="thin"),
                    top=Side(style="thin"),
                    bottom=Side(style="thin")
                )
            
            df = st.session_state.output_df
            current_category = None
            category_start_row = 2
            row_idx = 2
            
            for category in df['Observation - Category'].unique():
                category_df = df[df['Observation - Category'] == category]
                category_start_row = row_idx
                
                for _, row_data in category_df.iterrows():
                    cell = ws.cell(row=row_idx, column=1)
                    cell.value = row_data["Samples affected"]
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border = Border(
                        left=Side(style="thin"),
                        right=Side(style="thin"),
                        top=Side(style="thin"),
                        bottom=Side(style="thin")
                    )
                    
                    cell = ws.cell(row=row_idx, column=2)
                    cell.value = category if row_idx == category_start_row else None
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                    cell.border = Border(
                        left=Side(style="thin"),
                        right=Side(style="thin"),
                        top=Side(style="thin"),
                        bottom=Side(style="thin")
                    )
                    
                    cell = ws.cell(row=row_idx, column=3)
                    cell.value = row_data["Page"]
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border = Border(
                        left=Side(style="thin"),
                        right=Side(style="thin"),
                        top=Side(style="thin"),
                        bottom=Side(style="thin")
                    )
                    
                    cell = ws.cell(row=row_idx, column=4)
                    cell.value = row_data["Sub-category of Observation"]
                    cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
                    cell.border = Border(
                        left=Side(style="thin"),
                        right=Side(style="thin"),
                        top=Side(style="thin"),
                        bottom=Side(style="thin")
                    )
                    
                    row_idx += 1
                
                if row_idx > category_start_row + 1:
                    ws.merge_cells(start_row=category_start_row, start_column=2, end_row=row_idx-1, end_column=2)
            
            ws.column_dimensions['A'].width = 15
            ws.column_dimensions['B'].width = 40
            ws.column_dimensions['C'].width = 20
            ws.column_dimensions['D'].width = 60
            
            wb.save(buffer)
            buffer.seek(0)
            
            st.download_button(
                label="Download Comparison Report",
                data=buffer,
                file_name="document_comparison_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("No differences found between the documents.")
    else:
        st.info("No comparison results available yet. Please run a comparison first.")

def main():
    st.title("Document Comparison Tool")
    
    with st.expander("How to Use"):
        st.markdown("""
        ### Instructions
        1. Upload the company document (DOCX format)
        2. Upload the customer document (DOCX format) 
        3. Upload the checklist Excel file
        4. Enter your Groq API key
        5. Click "Run Comparison"
        
        ### About the Tool
        This tool compares two documents section by section based on the checklist provided.
        It uses AI to identify differences between the documents and categorizes them for easy review.
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        company_file = st.file_uploader("Upload Company Document (DOCX)", type=["docx"])
    with col2:
        customer_file = st.file_uploader("Upload Customer Document (DOCX)", type=["docx"])
    
    checklist_file = st.file_uploader("Upload Checklist (Excel)", type=["xlsx", "xls"])
    
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    
    if st.button("Run Comparison", disabled=not (company_file and customer_file and checklist_file and groq_api_key)):
        with st.spinner("Processing..."):
            company_path = save_uploaded_file(company_file)
            customer_path = save_uploaded_file(customer_file)
            checklist_path = save_uploaded_file(checklist_file)
            
            success = run_direct_comparison(company_path, customer_path, checklist_path, groq_api_key)
            
            if success:
                st.success("Comparison completed successfully!")
            else:
                st.error("Comparison failed. Check logs for details.")
    
    if st.session_state.comparison_completed:
        display_results()

if __name__ == "__main__":
    main()
