import streamlit as st
import os
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
# from langchain_core.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
import streamlit.components.v1 as components
from streamlit_pdf_viewer import pdf_viewer
from io import BytesIO
import base64 

if 'pdf_ref' not in st.session_state:
    st.session_state.pdf_ref = None

# Initialize the Groq API Key and the model
os.environ["GROQ_API_KEY"] = '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'
# config = {'max_new_tokens': 512, 'context_length': 8000}
llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

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

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
# new_vector_store = FAISS.load_local(
#     "faiss_index", embeddings, allow_dangerous_deserialization=True
# )

# docs = new_vector_store.similarity_search("qux")
# Conversational chain for Q&A
def get_conversational_chain():
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
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
Key Note : User can enter the question in 2 language either in english or in gujarati or in both mix sentence. so answer accordingly based on user question languge.
Input:
OCR-processed text from uploaded documents: {context}
Chat History: {chat_history}
Current Question: {question}
Output:
Give a clear, direct, and user-friendly response that focuses on the information itself rather than its OCR source. Present information confidently, mentioning verification only for critically important or potentially ambiguous details.Answer in user questioned language if question is in gujarati answer in guajarati else default in english.
"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    new_vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["chat_history", "context", "question"], template=template)
    
    # qa_chain = RetrievalQA.from_chain_type(llm, retriever=new_vector_store.as_retriever(), chain_type='stuff', verbose=True, chain_type_kwargs={"verbose": True,"prompt": QA_CHAIN_PROMPT,"memory": ConversationBufferMemory(memory_key="chat_history",input_key="question"),})
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=new_vector_store.as_retriever(),
        chain_type='stuff',
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": QA_CHAIN_PROMPT,
            "memory": st.session_state.conversation_memory,
        }
    )
    # qa_chain = RetrievalQA.from_chain_type(llm, retriever=new_vector_store.as_retriever(), chain_type='stuff', verbose=True, chain_type_kwargs={"verbose": True,"prompt": QA_CHAIN_PROMPT,"memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True)})
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

        # if file_extension == ".pdf":
        #     st.session_state.pdf_ref = uploaded_file  # Save the PDF to session state
        #     binary_data = st.session_state.pdf_ref.getvalue()  # Get the binary data of the PDF
        #     # Use the pdf_viewer to display the PDF
        #     # sidebar.pdf_viewer(input=binary_data, width=700)
        if file_extension == ".pdf":
            # Display the PDF in the sidebar by embedding the PDF file
            with open(file_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()
            # Use the HTML iframe to display the PDF in the sidebar
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
            st.sidebar.markdown(f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="500" height="500"></iframe>', unsafe_allow_html=True)
   
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
            img = Image.open(file_path)
            st.sidebar.image(img, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)  # Updated here
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            st.sidebar.text_area("File Content", content, height=300)

        
    
    # Optionally show document in the main content area
    # st.write(f"### Main Panel - {uploaded_file.name}")
    # if file_extension == '.pdf':
    #     st.write("Displaying PDF:")
    #     st.components.v1.html(f'<embed src="{file_path}" width="700" height="500" type="application/pdf">')
    # elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
    #     img = Image.open(file_path)
    #     st.image(img, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
    # else:
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         content = f.read()
    #     st.text_area("File Content", content, height=300)

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

    # with st.expander('Conversation History'):
    #     for entry in st.session_state.conversation_history:
    #         st.info(f"Q: {entry['question']}\nA: {entry['answer']}")

    # Append new question and response to the history
    st.session_state.conversation_history.append({'question': user_question, 'answer': result})
    
    return result
    
# def handle_uploaded_file(uploaded_file, show_in_sidebar=False):
#     file_extension = os.path.splitext(uploaded_file.name)[1].lower()
#     file_path = os.path.join("temp", uploaded_file.name)
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)

#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # Show document in the main panel and optionally in the sidebar
#     if show_in_sidebar:
#         st.sidebar.write(f"### File: {uploaded_file.name}")
#         if file_extension == '.pdf':
#             st.sidebar.write("Displaying PDF:")
#             st.sidebar.components.html(f'<embed src="{file_path}" width="700" height="500" type="application/pdf">')

#             # st.sidebar.components.v1.html(f'<embed src="{file_path}" width="700" height="500" type="application/pdf">')
#         elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
#             img = Image.open(file_path)
#             st.sidebar.image(img, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
#         else:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 content = f.read()
#             st.sidebar.text_area("File Content", content, height=300)
    
    # Optionally show document in the main content area
    # st.write(f"### Main Panel - {uploaded_file.name}")
    # if file_extension == '.pdf':
    #     st.write("Displaying PDF:")
    #     st.components.v1.html(f'<embed src="{file_path}" width="700" height="500" type="application/pdf">')
    # elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
    #     img = Image.open(file_path)
    #     st.image(img, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
    # else:
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         content = f.read()
    #     st.text_area("File Content", content, height=300)

# Streamlit app to upload files and interact with the Q&A system
def main():
    st.title("File Upload and OCR Processing")
    st.write("Upload up to 5 files (PDF, JPG, JPEG, PNG, BMP)")


    uploaded_files = st.file_uploader("Choose files", type=["pdf", "jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)

    if len(uploaded_files) > 0:
        file_paths = []

        # Save uploaded files and process them
        for uploaded_file in uploaded_files[:5]:  # Limit to 5 files
            file_path = os.path.join("temp", uploaded_file.name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)


        # Process the OCR and PDF files and store the vector data
        st.write("Processing files...")
        vector_store = process_ocr_and_pdf_files(file_paths)
        st.write("Processing completed! The vector store has been updated.")
        
    show_in_sidebar = st.sidebar.checkbox("Show files in Sidebar", value=True)

    if len(uploaded_files) > 0:
        # Process and display each uploaded file in its format
        for uploaded_file in uploaded_files:
            handle_uploaded_file(uploaded_file, show_in_sidebar)

        # Ask user for a question related to the documents
        user_question = st.text_input("Ask a question related to the uploaded documents:")

        if user_question:
            response = user_input(user_question)
            st.write("Answer:", response)

            # Button to display chat history

            # if st.button("Show Chat History"):
            #     history = st.session_state.get('history', [])
            #     if history:
            #         st.write("Conversation History:")
            #         for idx, (q, a) in enumerate(history):
            #             st.write(f"Q{idx+1}: {q}")
            #             st.write(f"A{idx+1}: {a}")
            #     else:
            #         st.write("No conversation history.")
            with st.expander('Conversation History'):
                for entry in st.session_state.conversation_history:
                    st.info(f"Q: {entry['question']}\nA: {entry['answer']}")
    

if __name__ == "__main__":
    main()
