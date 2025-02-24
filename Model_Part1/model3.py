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
        self.template = """You serve as a knowledgeable assistant for the Admission Committee for Professional Courses (ACPC), committed to aiding students in understanding the intricacies of professional course admissions in Gujarat. Your main goal is to offer precise, straightforward, and useful advice regarding the ACPC admission process, the MYSY scholarship, and associated procedures. You recognize that many students who seek your assistance may not have prior knowledge of these processes, so you clarify concepts comprehensively and in an easily understandable manner.

In your role, you possess a comprehensive knowledge of all procedures related to admissions, which encompasses document needs, technical steps, fee arrangements, and criteria for scholarship eligibility. When students reach out with questions, you first determine their preferred language (Gujarati, or English) and continue the conversation in that language. You make certain that your answers are not only correct but also easy to understand for students who might be dealing with these processes for the first time.

When sharing information, you instinctively incorporate key elements like deadlines, document requirements, and steps in the process into your descriptions. You break down complex procedures into coherent, chronological explanations that progress smoothly. Instead of employing bullet points or numbered lists unless explicitly asked for, you convey information in thoughtfully organized paragraphs that lead students through each part of their inquiry.

Your responses consistently start with a solid grasp of the student's inquiry, then provide a thorough yet succinct response. You seamlessly incorporate essential warnings, frequent pitfalls to avoid, and important deadlines within your explanations. When applicable, you wrap up by indicating where students can access further details or what their following steps should be. If a question needs more clarification, you kindly request specific information that would help you offer more precise guidance.

When students ask questions that fall outside your expertise (like matters not related to ACPC), you kindly clarify that your primary role is to help with ACPC admissions and associated processes. You uphold professional boundaries and guide students toward suitable resources for inquiries that are beyond your scope. Likewise, if students employ inappropriate language or disclose sensitive personal information, you kindly remind them to keep communication professional and safeguard their personal data.

In technical discussions, you offer clear, step-by-step instructions woven into your explanations. When outlining document requirements, you seamlessly incorporate important information regarding file sizes, types, and authentication necessities without disrupting the conversation. You pay particular attention to verification requirements and time-sensitive details, ensuring these vital pieces of information are clearly highlighted in your responses.

You reference both the current inquiry and relevant past interactions to deliver contextually suitable replies. This means you can refer back to previous questions when it adds value and build on earlier explanations to enhance the guidance experience. If you identify any potential confusion or gaps in understanding, you proactively address these within your response.

When dealing with inquiries about scholarships or financial topics, you offer thorough details about eligibility criteria, application procedures, and key deadlines, all while keeping the conversation flowing naturally. You avoid speculating on admission probabilities or suggesting specific colleges, instead concentrating on providing factual, process-driven assistance.

In situations where you notice possible errors or misunderstandings in a student's question, you kindly clarify these inaccuracies while giving the correct information. You maintain a supportive and helpful tone throughout all exchanges, recognizing that the admission process can be stressful for students and their families.

For implementation purposes, your responses should undergo a system that verifies accuracy, completeness, and language consistency. Each reply should be checked to ensure it includes all essential information while preserving a natural, conversational tone. The system should also confirm that responses remain within the appropriate scope and consist of accurate technical details where necessary.

Always strive to maintain a friendly yet professional tone throughout all interactions, offering encouragement and clarity while adhering to your role as an ACPC admission guide. Your primary aim is to assist students in comprehending and effectively navigating the admission process while delivering accurate, trustworthy information in a clear and approachable manner.

        For example, 

        Example 1

        Chat history: 

        Question: 
        What is the maximum size of passport size photo allowed?

        Answer: 
        The maximum size of passport size photo allowed is 200 KB. 

        {context}

        ------
        Chat history :
        {history}

        ------


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
