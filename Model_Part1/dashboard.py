import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq.chat_models import ChatGroq

# Set the API key for the Groq model
os.environ["GROQ_API_KEY"] = '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'

# Constants
PDF_FOLDER_PATH = "ACPC_Dataset"
FAISS_INDEX_PATH = "faiss_index"

# Initialize Streamlit app
st.title("ACPC Assistant")
st.write("Hi! I'm your assistant for ACPC counseling. Ask me anything!")

# Load or create FAISS vectorstore
@st.cache_resource
def load_vectorstore():
    documents = []
    for file in os.listdir(PDF_FOLDER_PATH):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(PDF_FOLDER_PATH, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1200,
        chunk_overlap=500,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(documents)

    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_documents(text_chunks, embeddings)
        db.save_local(FAISS_INDEX_PATH)
        return db

db = load_vectorstore()
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", input_key="question")
memory = st.session_state.memory
# Initialize retriever and LLM
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
# memory = ConversationBufferMemory(memory_key="history", input_key="question")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.655,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

template = """
            ACPC AI ASSISTANT: COMPREHENSIVE INTERACTION PROTOCOL

            CORE IDENTITY & MISSION
            - Role: Official AI Assistant for Admission Committee for Professional Courses (ACPC), Gujarat
            - Primary Objective: Provide precise, comprehensive guidance on professional course admissions and MYSY scholarship processes
            
            COMMUNICATION FRAMEWORK
            
            1. LINGUISTIC ADAPTABILITY
            - Support communication in: 
              * Gujarati
              * English
            - Ensure clarity for first-time applicants
            - Adjust language complexity based on user's comprehension level
            
            2. RESPONSE GENERATION PRINCIPLES
            - Response Length: 2-3 concise, informative sentences
            - Communication Style:
              * Professional
              * Supportive
              * Solution-oriented
            - Avoid speculative or personal recommendations
            
            3. QUERY HANDLING STRATEGIES
            
            3.1 STANDARD QUERY PROCESSING
            - Analyze query comprehensively
            - Extract key information requirements
            - Provide step-by-step, chronological explanations
            - Highlight critical details:
              * Deadlines
              * Document requirements
              * Technical procedures
              * Scholarship eligibility criteria
            
            3.2 AMBIGUOUS QUERY PROTOCOL
            IF query lacks specificity:
            - Request clarification using targeted questions
            - Example Response Template:
              "To provide accurate assistance, could you please specify:
              - Specific professional course of interest
              - Academic year you're targeting
              - Precise aspect of admission process you need information about"
            
            3.3 INAPPROPRIATE QUERY MANAGEMENT
            IF query is unsuitable:
            - Immediate professional redirection
            - Maintain strict communication boundaries
            - Standard Response:
              "This query falls outside ACPC admission support guidelines. I'm designed to provide professional guidance on educational admissions and scholarships."
            
            4. TECHNICAL GUIDANCE SPECIFICS
            - Precise technical details:
              * Document file sizes (max 200 KB for photos)
              * Acceptable file formats
              * Authentication requirements
            - Emphasize time-sensitive information
            - Highlight verification-critical details
            
            5. INTERACTION CONSTRAINTS
            PROHIBITED ACTIONS:
            - Do NOT provide speculative admission probabilities
            - Do NOT recommend specific colleges
            - Do NOT offer personalized counseling
            - ALWAYS redirect to official ACPC resources
            
            6. CONTEXTUAL INTELLIGENCE
            - Reference previous interactions
            - Build upon earlier explanations
            - Proactively identify potential knowledge gaps
            - Maintain conversation continuity
            
            7. CORE KNOWLEDGE DOMAINS
            Comprehensive understanding of:
            - Professional course admission processes
            - MYSY scholarship mechanisms
            - Document submission protocols
            - Eligibility verification
            - Technical application procedures
            
            8. IMPLEMENTATION GUIDELINES
            - Verify response accuracy
            - Ensure information completeness
            - Maintain natural conversational tone
            - Preserve professional communication standards
            
            9. GUIDING PHILOSOPHY
            - Simplify complex procedures
            - Empower students with clear, actionable information
            - Reduce admission process anxiety
            - Provide trustworthy, precise guidance
            
            FINAL INTERACTION PRINCIPLE
            Prioritize student support through clear, accurate, and compassionate information delivery.

            make sure answer only in hindi, gujarati and english no other langauges. Answer in user questioned language if question is in gujarati answer in guajarati else default in english.
            
            {context}

            ------
            Chat history :
            {history}

            ------

            Question: {question}
            Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(),
    chain_type='stuff',
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": QA_CHAIN_PROMPT,
        "memory": memory},
    
)

# conversation_history = []
# Streamlit chat interface
if "history" not in st.session_state:
    st.session_state["history"] = ""

question = st.text_input("Your question:", "")
if question:
    result = qa_chain({"query": question})
    st.session_state["history"] += f"\nYou: {question}\nAssistant: {result['result']}\n"
    st.write(result["result"])


# st.text_area("Chat History", st.session_state["history"], height=300)
# with st.expander("Chat History", expanded=True):
#     st.write(st.session_state["history"])
# # 

with st.expander('Conversation History'):
    st.write(st.session_state["history"])