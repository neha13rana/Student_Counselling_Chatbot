import os
import gradio as gr
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
from io import BytesIO

# Set up Groq API Key and LLM 
os.environ["GROQ_API_KEY"] = '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'
llm = ChatGroq(
    model='llama3-70b-8192',
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# OCR Functions
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

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    os.makedirs("faiss_index", exist_ok=True)
    vector_store.save_local("faiss_index")
    
    return vector_store

def process_ocr_and_pdf_files(file_paths):
    raw_text = ""
    for file_path in file_paths:
        raw_text += ocr_file(file_path) + "\n"
    text_chunks = get_text_chunks(raw_text)
    return get_vector_store(text_chunks)

def get_conversational_chain():
    template = """You are an intelligent educational assistant specialized in handling queries about documents. You have been provided with OCR-processed text from the uploaded files that contains important educational information.

Core Responsibilities:
1. Language Processing:
   - Identify the language of the user's query (English or Gujarati)
   - Respond in the same language as the query
   - If the query is in Gujarati, ensure the response maintains proper Gujarati grammar and terminology
   - For technical terms, provide both English and Gujarati versions when relevant

2. Document Understanding:
   - Analyze the OCR-processed text from the uploaded files
   - Account for potential OCR errors or misinterpretations
   - Focus on extracting accurate information despite possible OCR imperfections

3. Response Guidelines:
   - Provide direct, clear answers based solely on the document content
   - If information is unclear due to OCR quality, mention this limitation
   - For numerical data (dates, percentages, marks), double-check accuracy before responding
   - If information is not found in the documents, clearly state: "This information is not present in the uploaded documents"

4. Educational Context:
   - Maintain focus on educational queries related to the document content
   - For admission-related queries, emphasize important deadlines and requirements
   - For scholarship information, highlight eligibility criteria and application processes
   - For course-related queries, provide detailed, accurate information from the documents

5. Response Format:
   - Structure responses clearly with relevant subpoints when necessary
   - For complex information, break down the answer into digestible parts
   - Include relevant reference points from the documents when applicable
   - Format numerical data and dates clearly

6. Quality Control:
   - Verify that responses align with the document content
   - Don't make assumptions beyond the provided information
   - If multiple interpretations are possible due to OCR quality, mention all possibilities
   - Maintain consistency in terminology throughout the conversation

Important Rules:
- Never make up information not present in the documents
- Don't combine information from previous conversations or external knowledge
- Always indicate if certain parts of the documents are unclear due to OCR quality
- Maintain professional tone while being accessible to students and parents
- If the query is out of scope of the uploaded documents, politely redirect to relevant official sources

Context from uploaded documents:
{context}

Chat History:
{history}

Current Question: {question}
Assistant: Let me provide a clear and accurate response based on the uploaded documents...
"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    new_vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["history", "context", "question"], 
        template=template
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=new_vector_store.as_retriever(), 
        chain_type='stuff', 
        verbose=True, 
        chain_type_kwargs={
            "verbose": True,
            "prompt": QA_CHAIN_PROMPT,
            "memory": ConversationBufferMemory(memory_key="history", input_key="question"),
        }
    )
    
    return qa_chain

def process_files_and_query(files, query):
    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)
    
    # Save uploaded files
    file_paths = []
    for file in files:
        file_path = os.path.join("temp", os.path.basename(file))
        with open(file_path, "wb") as f:
            f.write(open(file, 'rb').read())
        file_paths.append(file_path)
    
    # Process files and create vector store
    process_ocr_and_pdf_files(file_paths)
    
    # Perform query
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(query)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "query": query}, return_only_outputs=True)
    result = response.get("result", "No result found")
    
    return result
def handle_uploaded_file(uploaded_files, show_in_sidebar=False):
    sidebar_content = ""
    
    # If the uploaded_files is a list, process each file
    for uploaded_file in uploaded_files:
        # Determine the file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        file_path = os.path.join("temp", uploaded_file.name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Check if the uploaded file is in 'NamedString' format (Gradio sometimes returns it this way)
        if isinstance(uploaded_file, gr.File):
            # In this case, read the file directly from the 'data' attribute
            file_data = uploaded_file.read()  # This is the file content in bytes

            # Save the file content to a local file
            with open(file_path, "wb") as f:
                f.write(file_data)

        if file_extension == ".pdf":
            # Read and encode the PDF as base64 to embed in the sidebar
            with open(file_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
            sidebar_content += f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="500" height="500"></iframe>'
        
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Display image in the sidebar
            img = Image.open(file_path)
            img_byte_array = BytesIO()
            img.save(img_byte_array, format="PNG")
            img_byte_array.seek(0)
            sidebar_content += f'<img src="data:image/png;base64,{base64.b64encode(img_byte_array.getvalue()).decode()}" width="400" height="400"/>'

        else:
            # For text files, show the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            sidebar_content += f"<pre>{content}</pre>"

    return sidebar_content

# Gradio interface setup
def upload_and_display(files):
    sidebar_content = handle_uploaded_file(files, show_in_sidebar=True)
    return sidebar_content

def launch_gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Document OCR and Q&A Assistant")
        
        with gr.Row():
            with gr.Column(scale=1):  # Main content area (adjusted scale to an integer)
                file_input = gr.File(
                    file_count="multiple", 
                    type="filepath",  # Changed from 'file' to 'filepath'
                    file_types=[".pdf", ".jpg", ".jpeg", ".png", ".bmp"],
                    label="Upload Documents (PDF/Images)"
                )
                
                query_input = gr.Textbox(
                    label="Ask a Question about the Documents", 
                    lines=3
                )
                
                submit_btn = gr.Button("Process and Query")
                
                output = gr.Textbox(label="Answer", lines=5)
                
                submit_btn.click(
                    fn=process_files_and_query, 
                    inputs=[file_input, query_input], 
                    outputs=[output]
                )
                
            with gr.Column(scale=0.5):  # Sidebar (adjusted scale to an integer)
                gr.Markdown("## Sidebar")
                file_preview = gr.HTML(label="File Preview")  # Display the preview content here
                file_input.change(fn=upload_and_display, inputs=file_input, outputs=file_preview)

    return demo

# Launch the Gradio app
if __name__ == "__main__":
    app = launch_gradio_app()
    app.launch(share=True)  # Set share=True to create a public link



# # Launch the Gradio app
# if __name__ == "__main__":
#     app = launch_gradio_app()
#     # app.launch()
#     app.launch(share=True) 
    # demo.launch()