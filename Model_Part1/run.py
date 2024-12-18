import os
os.environ["GROQ_API_KEY"] = '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'
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_groq.chat_models import ChatGroq

pdf_folder_path = "ACPC_Dataset"
documents = []
for file in os.listdir(pdf_folder_path):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
        
embeddings= HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2",model_kwargs = {'device': 'cpu'},encode_kwargs = {'normalize_embeddings': True})

text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size=1200,
        chunk_overlap=500,
        length_function=len)
text_chunks=text_splitter.split_documents(documents)

db1 = FAISS.from_documents(text_chunks, embeddings)

retriever1 = db1.as_retriever(search_type="similarity", search_kwargs={"k": 1})

memory = ConversationBufferMemory(memory_key="history", input_key="question")

llm = ChatGroq(
    # model="mixtral-8x7b-32768",
    # model ='llama3-8b-8192',
    model = "llama-3.3-70b-versatile",
    temperature=0.655,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

template="""You are a smart and helpful assistant for the ACPC counseling process. You guide students and solve their queries related to ACPC, MYSY scholarship, admission, etc. You will be given the student's query and the history of the chat, and you need to answer the query to the best of your knowledge. If the query is completely different from the context then tell the student that you are not made to answer this query in a polite language. If the student has included any type of content related to violence, sex, drugs or used abusive language then tell the student that you can not answer that query and request them not to use such content. 

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


QA_CHAIN_PROMPT = PromptTemplate(input_variables=["history", "context", "question"],template=template,)
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=db1.as_retriever(),
                                       chain_type='stuff',
                                       verbose=True,
                                       chain_type_kwargs={"verbose": True,"prompt": QA_CHAIN_PROMPT,"memory": ConversationBufferMemory(memory_key="history",input_key="question"),})

print("Hi! How can I help you today?")
while True:
    question = input("User: ")
    if question.lower() == "quit":
        print("Thank you for chatting. Goodbye!")
        break

    result1 = qa_chain({"query": question})
    print(result1["result"])
    print("-----------------------------")
