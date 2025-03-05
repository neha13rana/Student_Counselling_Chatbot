import os
import torch
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq


class ChatbotModel:
    def __init__(self):
        # Initialize the environment variable for the GROQ API Key
        os.environ["GROQ_API_KEY"] = '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'

        # Load documents from PDFs
        pdf_folder_path = "ACPC_Dataset"
        documents = []
        for file in os.listdir(pdf_folder_path):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(pdf_folder_path, file)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                                model_kwargs={'device': 'cpu'},
                                                encode_kwargs={'normalize_embeddings': True})

        # Split documents into chunks
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1200,
            chunk_overlap=500,
            length_function=len)
        self.text_chunks = self.text_splitter.split_documents(documents)

        # Create FAISS vector store
        self.db1 = FAISS.from_documents(self.text_chunks, self.embeddings)

        # Initialize memory for conversation
        self.memory = ConversationBufferMemory(memory_key="history", input_key="question")

        # Initialize the chat model
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.655,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # Create the QA chain prompt template
        self.template = """ACPC AI ASSISTANT: COMPREHENSIVE INTERACTION PROTOCOL

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
            
            {context}
            {history}
            Question: {question}
            Answer
            """

        self.QA_CHAIN_PROMPT = PromptTemplate(input_variables=["history", "context", "question"],
                                              template=self.template)
        self.qa_chain = RetrievalQA.from_chain_type(self.llm,
                                                    retriever=self.db1.as_retriever(),
                                                    chain_type='stuff',
                                                    verbose=True,
                                                    chain_type_kwargs={"verbose": True, "prompt": self.QA_CHAIN_PROMPT,
                                                                       "memory": self.memory})

    def save(self, path):
        # Save only the necessary parameters
        torch.save({
            'text_chunks': self.text_chunks,
            'embeddings_model_name': self.embeddings.model_name,  # Save the model name
            'faiss_index': self.db1.index  # Save FAISS index if needed
        }, path)

    def get_response(self, user_input):
        # Call the QA chain with the user's input
        result = self.qa_chain({"query": user_input})
        return result["result"]

    @classmethod
    def load(cls, path):
        # Load the model state
        state = torch.load(path)
        chatbot_model = cls()
        # Restore other components
        chatbot_model.text_chunks = state['text_chunks']

        # Recreate embeddings using the saved model name
        chatbot_model.embeddings = HuggingFaceEmbeddings(model_name=state['embeddings_model_name'],
                                                         model_kwargs={'device': 'cpu'},
                                                         encode_kwargs={'normalize_embeddings': True})

        # Recreate FAISS index if necessary
        chatbot_model.db1 = FAISS.from_documents(chatbot_model.text_chunks, chatbot_model.embeddings)
        return chatbot_model




# Test saving the model
if __name__ == "__main__":
    chatbot = ChatbotModel()
    chatbot.save("model2.pt")
    print("Model saved successfully.")
