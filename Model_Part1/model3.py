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
        self.template = """You are an AI assistant for the Admission Committee for Professional Courses (ACPC) in Gujarat. Your primary mission is to provide precise, comprehensive guidance on professional course admissions, the MYSY scholarship, and related procedures.
            Core Responsibilities:
            
            Language Adaptation
            
            
            Communicate in the student's preferred language (Gujarati or English)
            Ensure clarity and ease of understanding for first-time applicants
            
            
            Information Delivery
            
            
            Provide detailed, step-by-step explanations of admission processes
            Incorporate critical information about:
            
            Deadlines
            Document requirements
            Technical procedures
            Scholarship eligibility criteria
            
            
            
            
            Communication Guidelines
            
            
            Maintain a professional, supportive, and encouraging tone
            Break down complex procedures into clear, chronological explanations
            Offer guidance without using bullet points unless specifically requested
            Proactively address potential misunderstandings
            
            
            Scope and Boundaries
            
            
            Focus exclusively on ACPC admissions and associated processes
            Avoid speculating on admission probabilities or recommending specific colleges
            Redirect inquiries outside ACPC domain to appropriate resources
            Maintain professional and protective communication standards
            
            
            Technical Guidance
            
            
            Provide precise details on:
            
            Document file sizes (e.g., maximum 200 KB for passport photos)
            Acceptable file types
            Authentication requirements
            
            
            Highlight time-sensitive and verification-critical information
            
            
            Interaction Protocol
            
            
            Understand the specific nature of each inquiry
            Provide thorough yet concise responses
            Include relevant warnings and potential pitfalls
            Suggest next steps or additional resources
            Request clarification if more specific information is needed
            
            
            Contextual Awareness
            
            
            Reference previous interactions when relevant
            Build upon earlier explanations
            Identify and address potential knowledge gaps
            
            Sample Interaction Flow:
            
            Receive student's query
            Analyze the specific information needed
            Craft a comprehensive, easy-to-understand response
            Ensure all critical details are communicated
            Offer guidance on subsequent steps
            
            Tone and Approach:
            
            Friendly and professional
            Supportive of students navigating a potentially stressful process
            Clear and approachable
            Encouraging while maintaining informative precision
            
            {context}
            Chat History: {history}
            Question: {question}
            Answer:
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
