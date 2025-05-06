import streamlit as st
import os
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import base64
from google import genai
from google.genai import types

# Set up API keys
os.environ["GROQ_API_KEY"] = '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'
GEMINI_API_KEY = "AIzaSyCuYUYmAP3VF09d4sXZdpo6yl9mz8LcTmo"

# Initialize the Groq LLM
llm = ChatGroq(
    model='llama3-70b-8192',
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Initialize Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize session state variables
if 'pdf_ref' not in st.session_state:
    st.session_state.pdf_ref = None

if 'is_acpc_related' not in st.session_state:
    st.session_state.is_acpc_related = False
    
if 'acpc_analysis_complete' not in st.session_state:
    st.session_state.acpc_analysis_complete = False

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Define function to check if document is ACPC-related
def check_acpc_related(file_path):
    try:
        uploaded_file = gemini_client.files.upload(file=file_path)
        content = types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type=uploaded_file.mime_type,
                ),
                types.Part.from_text(text="""
Please analyze the provided document content and determine its relationship to the following topics. Use deep understanding — not surface-level keyword matching.
For each topic, classify and justify:
- **Directly Related**: Is the document explicitly about this topic? Issued by relevant authorities? Contains official steps, announcements, or structured descriptions?
- **Indirectly Related**: Related through associated frameworks, follow-up steps, or secondary context?
- **Not Related**: Mentions topic but has no substantive relevance?
---
**Topics:**
1. **ACPC (Admission Committee for Professional Courses)**  
2. **Admission Counselling Process (in Gujarat)**  
3. **MYSY (Mukhyamantri Yuva Swavalamban Yojana)**  
4. **Any overall relationship to ACPC (summary)**
---
**Output Format:**
Analysis of [Document Title or Description]:
1. **ACPC:**
Directly Related to ACPC? [Yes/No]  
Justification: [...]
Indirectly Related to ACPC? [Yes/No]  
Justification: [...]
2. **Counselling Process:**
Directly Related to Counselling Process? [Yes/No]  
Justification: [...]
Indirectly Related to Counselling Process? [Yes/No]  
Justification: [...]
3. **MYSY:**
Directly Related to MYSY? [Yes/No]  
Justification: [...]
Indirectly Related to MYSY? [Yes/No]  
Justification: [...]
4. **Summary of Any Relationship to ACPC:**  
[List and explain all relevant connections, or say "No relationship found".]
---
Finally, provide a one-word conclusion: "RELEVANT" or "NOT_RELEVANT" based on your analysis of the document's relevance to ACPC or related topics.
Only include strong, validated relationships. Avoid false positives from surface mentions. Be strict.
""")
            ]
        )
        model = "gemini-2.0-flash-thinking-exp-01-21"
        response = gemini_client.models.generate_content(
            model=model,
            contents=[content],
            config = genai.types.GenerateContentConfig(
                temperature=0,
                top_p=0.95,
                top_k=20,
                system_instruction = """
                You are an intelligent assistant designed to analyze documents and determine their actual relevance to specific topics related to the Gujarat student admission ecosystem.
                Your task is to critically assess the content of the document — not just by keyword presence but by understanding the document's main purpose, context, and issuer. Avoid being misled by mere mentions of keywords like "ACPC", "MYSY", or "Counselling". A document is only relevant if it substantively discusses or is directly issued by related authorities, or serves as an official part of the described processes.
                Always follow these rules:
                - **DO NOT** consider a document related if it only contains a passing or indirect mention of a topic.
                - **DO** validate based on the document's issuer, structure, purpose, and detailed content.
                - Focus on official processes, announcements, procedural steps, or eligibility information.
                - Justify every "Yes" or "No" with specific evidence **from the document** only.
                - At the end, clearly state "RELEVANT" or "NOT_RELEVANT" as a one-word conclusion.
                Your goal is to determine if a document should be included in a system that supports Question-Answering about ACPC, MYSY, and the counselling process in Gujarat.
                """
            )
        )
        result_text = response.text.strip()
        
        # Check if the document is relevant based on the one-word conclusion
        is_relevant = "RELEVANT" in result_text.upper().split()
        
        return result_text, is_relevant
    except Exception as e:
        return f"❌ Error: {str(e)}", False

# Define OCR functions for image and PDF files
def ocr_image(image_path, language='eng+guj'):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=language)
    return text

def ocr_pdf(pdf_path, language='eng+guj'):
    images = convert_from_path(pdf_path)
    all_text = ""
    for img in images:
        text = pytesseract.image_to_string(img, lang=language)
        all_text += text + "\n"
    return all_text

def ocr_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        text_re = ocr_pdf(file_path, language='guj+eng')
    elif file_extension in [".jpg", ".jpeg", ".png", ".bmp"]:
        text_re = ocr_image(file_path, language='guj+eng')
    else:
        raise ValueError("Unsupported file format. Supported formats are PDF, JPG, JPEG, PNG, BMP.")

    return text_re

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create or update the vector store
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    # Ensure the directory exists before saving the vector store
    os.makedirs("faiss_index", exist_ok=True)
    vector_store.save_local("faiss_index")
    
    return vector_store

# Function to process multiple files and extract vector store
def process_ocr_and_pdf_files(file_paths):
    raw_text = ""
    for file_path in file_paths:
        raw_text += ocr_file(file_path) + "\n"
    text_chunks = get_text_chunks(raw_text)
    return get_vector_store(text_chunks)

# Conversational chain for Q&A
def get_conversational_chain():
    template = """Core Identity & Responsibilities

Role: Official AI Assistant for Admission Committee for Professional Courses (ACPC), Gujarat
Mission: Process OCR-extracted text and provide clear, direct guidance on admissions and scholarships
Focus: Deliver user-friendly responses while handling OCR complexities internally

Processing Framework
1. Text & Document Processing

Process OCR-extracted text from various document types with attention to tables and structured data
Internally identify and handle OCR errors without explicitly mentioning them unless critical
Preserve tabular structures and relationships between data points
Present information in clean, readable formats regardless of source OCR quality

2. Language Handling

Support seamless communication in both Gujarati and English
Respond in the same language as the user's query
Present technical terms in both languages when relevant
Adjust language complexity to user comprehension level

3. Response Principles

Provide direct, concise answers (2-3 sentences for simple queries)
Skip unnecessary OCR quality disclaimers unless information is critically ambiguous
Present information in user-friendly formats, especially for tables and numerical data
Maintain professional yet conversational tone

Query Handling Strategies
1. Direct Information Queries

Provide straightforward answers without mentioning OCR processing
Example:
User: "What is the last date for application submission?"
Response: "The last date for application submission is June 15, 2025."
(NOT: "Based on the OCR-processed text, the last date appears to be...")

2. Table Data Extraction

Present tabular information in clean, structured format
Preserve relationships between data points
Example:
User: "What are the fees for different courses?"
Response:
"The fees for various courses are:

B.Tech: ₹1,15,000 (General), ₹58,000 (SC/ST)
B.Pharm: ₹85,000 (General), ₹42,500 (SC/ST)"
(NOT: "According to the OCR-extracted table, which may have quality issues...")

3. Ambiguous Information Handling

If OCR quality affects critical information (like dates, amounts, eligibility):

Provide the most likely correct information
Add a brief note suggesting verification only for critical information
Example: "The application deadline is June 15, 2025. For this important deadline, we recommend confirming on the official ACPC website."

4. Uncertain Information Protocol

For critically unclear OCR content:

State the most probable information
Add a simple verification suggestion without mentioning OCR
Example: "Based on the available information, the income limit appears to be ₹6,00,000. For this critical criterion, please verify on the official ACPC portal."

5. Structured Document Navigation

Present information in the same logical structure as the original document
Use headings and bullet points for clarity when appropriate
Maintain document hierarchies when explaining multi-step processes

6. Out-of-Scope Queries

Politely redirect without mentioning document or OCR limitations
Example: "This query is outside the scope of ACPC admission guidelines. For information about [topic], please contact [appropriate authority]."

7. Key Information Emphasis

Highlight critical information like deadlines, eligibility criteria, and document requirements
Make important numerical data visually distinct
Prioritize accuracy for dates, amounts, and eligibility requirements

8. Multi-Part Query Handling

Address each component of multi-part queries separately
Maintain logical flow between related pieces of information
Preserve context when explaining complex processes

9. Completeness Guidelines

Ensure responses cover all aspects of user queries
Provide step-by-step guidance for procedural questions
Include relevant related information that users might need

10. Response Quality Control

Internally verify numerical data consistency
Apply contextual understanding to identify potential OCR errors without mentioning them
Present information with confidence unless critically uncertain
Focus on delivering actionable information rather than discussing document limitations

Input:
OCR-processed text from uploaded documents: {context}
Chat History: {history}
Current Question: {question}
Output:
Give a clear, direct, and user-friendly response that focuses on the information itself rather than its OCR source. Present information confidently, mentioning verification only for critically important or potentially ambiguous details.
"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    new_vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=new_vector_store.as_retriever(), chain_type='stuff', verbose=True, chain_type_kwargs={"verbose": True,"prompt": QA_CHAIN_PROMPT,"memory": ConversationBufferMemory(memory_key="history",input_key="question"),})
    return qa_chain
     
def handle_uploaded_file(uploaded_file, show_in_sidebar=False):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show document in the main panel and optionally in the sidebar
    if show_in_sidebar:
        st.sidebar.write(f"### File: {uploaded_file.name}")

        if file_extension == ".pdf":
            # Display the PDF in the sidebar by embedding the PDF file
            with open(file_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()
            # Use the HTML iframe to display the PDF in the sidebar
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
            st.sidebar.markdown(f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="500" height="500"></iframe>', unsafe_allow_html=True)
   
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
            img = Image.open(file_path)
            st.sidebar.image(img, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            st.sidebar.text_area("File Content", content, height=300)

    return file_path

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "query": user_question}, return_only_outputs=True)
    result = response.get("result", "No result found")
    
    # Save the question and answer to session state for history tracking
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Append new question and response to the history
    st.session_state.conversation_history.append({'question': user_question, 'answer': result})
    
    return result

# Main function
def main():
    st.title("ACPC Document Checker and Chat Assistant")
    
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    
    # File uploader
    uploaded_files = st.file_uploader("Upload documents (PDF, JPG, JPEG, PNG, BMP)", type=["pdf", "jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
    
    if len(uploaded_files) > 0:
        file_paths = []
        
        # Save uploaded files
        for uploaded_file in uploaded_files[:5]:  # Limit to 5 files
            file_path = handle_uploaded_file(uploaded_file, show_in_sidebar=True)
            file_paths.append(file_path)
        
        # Check if the document is ACPC-related before processing
        if not st.session_state.acpc_analysis_complete:
            st.write("### Checking if document is ACPC-related...")
            
            with st.spinner("Analyzing document content..."):
                for file_path in file_paths:
                    analysis_result, is_relevant = check_acpc_related(file_path)
                    
                    # Store result in session state
                    st.session_state.is_acpc_related = is_relevant
                    st.session_state.acpc_analysis_complete = True
                    
                    # Display analysis result
                    with st.expander("Document Analysis Result"):
                        st.markdown(analysis_result)
                    
                    # If at least one document is relevant, break the loop
                    if is_relevant:
                        break
        
        # Process documents if they are ACPC-related
        if st.session_state.is_acpc_related:
            if st.button("Process Documents") or 'documents_processed' in st.session_state:
                st.session_state.documents_processed = True
                
                with st.spinner("Processing documents..."):
                    vector_store = process_ocr_and_pdf_files(file_paths)
                    st.success("Documents processed successfully!")
                
                # Chat interface
                st.write("### Chat with your documents")
                user_question = st.text_input("Ask a question about the uploaded documents:")
                
                if user_question:
                    with st.spinner("Generating response..."):
                        response = user_input(user_question)
                        st.write("Answer:", response)
                
                # Display conversation history
                with st.expander("Conversation History"):
                    for entry in st.session_state.conversation_history:
                        st.info(f"Q: {entry['question']}\nA: {entry['answer']}")
        else:
            st.error("⚠️ The uploaded document(s) are not related to ACPC, MYSY, or admission counseling process. Please upload relevant documents.")
            
            # Reset analysis state if user wants to try different documents
            if st.button("Upload Different Documents"):
                st.session_state.acpc_analysis_complete = False
                st.session_state.is_acpc_related = False
                st.session_state.conversation_history = []
                st.experimental_rerun()

if __name__ == "__main__":
    main()
