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
        self.template = """You are a smart and helpful assistant for the ACPC counseling process. You guide students and solve their queries related to ACPC, MYSY scholarship, admission, etc. You will be given the student's query and the history of the chat, and you need to answer the query to the best of your knowledge. If the query is completely different from the context then tell the student that you are not made to answer this query in a polite language. If the student has included any type of content related to violence, sex, drugs or used abusive language then tell the student that you can not answer that query and request them not to use such content. 

        Also make sure to reply in the same language as used by the student in the current query.

        NOTE that your answer should be accurate. Explain the answer such that a student with no idea about the ACPC can understand well.

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
