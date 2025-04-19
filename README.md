# 🎓✨ EduGuide Buddy: Your Smart Student Counselling Chatbot ✨🎓

### 📌 Overview

This project is an intelligent virtual assistant designed to streamline the admission process for educational institutions in Gujarat. The chatbot supports multiple languages and leverages advanced Large Language Models (LLMs) to provide dynamic, context-aware responses. It aims to eliminate language barriers, reduce administrative workload, and enhance accessibility for students and parents.

---

## ✨ Features

🎯 Multilingual Support – Get help in English, Gujarati, Hindi.

📄 Smart Document Processing – 1. Simple chat with existing data, 2. Upload PDFs & images and extract text using OCR (Tesseract).

🔒 Secure Authentication – Safe login/signup with JWT tokens.

📚 Chat History – Never lose a conversation; revisit past queries anytime.

🌐 Cross-Platform – Available on Web (MERN Stack) & Mobile (Android).

💡 AI-Powered Responses – Uses LangChain + Groq LLM for fast, accurate answers.

---

## 🛠️ Technologies Used

  **Backend**
  
  Node.js & Express.js: For server-side logic.
  
  MongoDB Atlas: Cloud-based database for storing user data and chat histories.
  
  JWT: For secure authentication.

  **Frontend (Web)**
  
  React + TypeScript: For building the user interface.
  
  Material UI: For a clean and responsive design.
  
  Vite: As the build tool for optimized performance.

  **Model & NLP**
  
  LangChain: Framework for integrating LLMs with external data.
  
  Hugging Face: Hosts the conversational model (Gradio interface).
  
  Groq: Provides fast and efficient AI processing.
  
  Tesseract OCR: For extracting text from uploaded documents.

 **Mobile (Android)**
 
  Android Studio: For app development.
  
  SQLite: Local database for chat history and feedback.
  
  Shared Preferences: For managing user settings.

---

## 🚀 Getting Started

  **Prerequisites**
  
  Node.js (v16 or later)
  
  MongoDB Atlas account
  
  Python (for model-related scripts)
  
  Android Studio (for mobile development)

---

## 🧠 Model Architecture


**Model 1 Architecture :**

![Minor_Diagram (1)](https://github.com/user-attachments/assets/f6c038b9-eaa4-4f75-b4a5-92e601b24bb4)

 **Model Description**
 
 1) Data Collection : Gathering all the data from the ACPC main website. At initial stage for the testing purpose we made a question answering pairs from the various documents of acpc.
 
 2) Data cleaning : Preprocesses the collected data to ensure it is in a usable format.
 
 3) Rag- Langchain : The core of our system is the Rag-Langchain pipeline, which manages the data preparation, retrieval, and processing steps to provide relevant responses to user queries. This pipeline consists of the following key components:
 
    1. Data loader :Loads the cleaned data into the system for further processing.
    
    2. Document Splitting : Splits the input documents into smaller, manageable chunks. (sets the chunk size and the chunk overlapping) Splitting the main document into several smaller chunks (for retaining meaningful relationships). There are many types of document splitter but we use a character splitter that looks at characters. Chunk size : 1200, Chunk Overlap : 500
    
    3. Embedding : Generates vector representations of the document chunks for efficient
    retrieval. Embedding vector captures meaning, text with similar content will have similar
    vectors.
    
    4. Vector Store : Stores the document embeddings for quick access during search.
    
    5. Relevant Splitting : Identifies the most relevant document chunks to the user’s query.
    
    6. Search Document: Searches the relevant document chunks to find the most pertinent information.
    
    7. User Question : Processes the user’s query to determine the appropriate response.
    
    8. Prompt : Prompt Engineering specifically to teach the model for the efficient and the accurate answer here in this project i need the chat conversation in various languages so i provide the prompt template to the model according to receive the best response. It can also ensures that model is not give any hallucinate answer or irrelevant answer of the users question.
    
    9. LLM : LLMs are advanced AI systems that can perform tasks such as language translation, text completion, summarization, and conversational interactions. They work by analyzing large amounts of language data and are pre-trained on vast amounts of data. In my project, I integrated the LLM via Groq, generates the final response based on the retrieved information.
    
    10. Output : User can find the response within a seconds if it is relevant to the ACPC or else model can not giving any answer


**Web Application Architecture :**

![WhatsApp Image 2024-11-15 at 17 21 03_5c6c6b97](https://github.com/user-attachments/assets/9871021f-7d49-420f-9238-3ad9078d4d8a)


---

## 📱 Application Screenshots

### Web Application
<div align="center" style="display: flex; flex-direction: column; gap: 20px; margin-bottom: 30px;">
  <div style="display: flex; gap: 20px; justify-content: center;">
    <div style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/59991cca-f8a2-4f4e-9a20-8716692f731a" width="300" alt="Web Main Page">
      <p><strong>Main Page</strong></p>
    </div>
    <div style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/c7f44729-9d69-4921-b243-4e24004db744" width="300" alt="Sign-up Page">
      <p><strong>Sign-up Page</strong></p>
    </div>
  </div>
  <div style="display: flex; gap: 20px; justify-content: center;">
    <div style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/8d8f69a8-5f23-4502-95fa-f6d3245ec606" width="300" alt="Login Page">
      <p><strong>Login Page</strong></p>
    </div>
    <div style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/a8bdbce8-ce35-4ef6-a234-b59acca07abd" width="300" alt="Chat Interface">
      <p><strong>Chat Interface</strong></p>
    </div>
  </div>
</div>

### Mobile Application
<div align="center" style="display: flex; flex-direction: column; gap: 20px; margin-bottom: 30px;">
  <div style="display: flex; gap: 15px; justify-content: center;">
    <div style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/d87c4dc3-7aa6-43ca-9c34-645f7b6573c0" width="160" alt="Mobile Home">
      <p><strong>Home Screen</strong></p>
    </div>
    <div style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/5a7eae14-c816-45ea-81f5-eb2cdaaa0c11" width="160" alt="Navigation">
      <p><strong>Navigation</strong></p>
    </div>
    <div style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/a7bc29f8-c5da-45a1-89cf-2d26f967671a" width="160" alt="Chat Screen">
      <p><strong>Chat Screen</strong></p>
    </div>
    <div style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/3a6e8923-d454-4842-a0a0-efab064b095e" width="160" alt="Chat History">
      <p><strong>Chat History</strong></p>
    </div>
  </div>
  <div style="display: flex; gap: 15px; justify-content: center;">
    <div style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/5bb191d2-2f1e-4d0c-a079-b95d24b63b8d" width="160" alt="FAQ">
      <p><strong>FAQ Section</strong></p>
    </div>
    <div style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/0bf26a5a-4c8c-41bd-bfcd-58bd3e36c45d" width="160" alt="Theme Settings">
      <p><strong>Theme Settings</strong></p>
    </div>
    <div style="text-align: center;">
     <img src="https://github.com/user-attachments/assets/b9a4cd62-ac10-4557-bbcb-012c6067efd9" width="160" alt="Notification">
      <p><strong>Feedback Form</strong></p>
    </div>
    <div style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/656dee5e-cdad-441e-9b52-86608852c00a" width="160" alt="Feedback">
      <p><strong>Submission Alert</strong></p>
    </div>
  </div>
</div>

### Document Processing Dashboard
<div align="center" style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">
  <div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/086a8ea1-bc23-4fda-9075-de0257c71865" width="300" alt="OCR Upload">
<!--     <p><strong>Document Upload</strong></p> -->
  </div>
  <div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/461a2869-947d-4b25-9b6e-311161ee9d60" width="300" alt="OCR Processing">
<!--     <p><strong>Text Extraction</strong></p> -->
  </div>
  <div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/c93e7d92-8848-4cbe-be0a-c2f27b69aaff" width="300" alt="OCR Results">
<!--     <p><strong>Processed Results</strong></p> -->
  </div>
</div>

---

## 🚀 Quick Setup Guide

### 1) Clone the Repository
```
git clone https://github.com/neha13rana/Student_Counselling_Chatbot.git
```

    
### 2) Backend Setup
```
 select web app backend from the branch
 cd _
 npm install
 cp .env.example .env  # Update environment variables
 npm start
```

### 3) Frontend Setup
```
 select web app frontend from the branch
 cd _
 npm install
 npm run dev
```

### 4) Model Setup (API / Dashboard)
```
 From Branch Select Model-and-API

 Model 1 :
 
 cd Model_Part1
 pip install -r requirements.txt
 python run.py  

 Model 2 :
 
 cd Model_Part2
 chmod +x setup.sh
 ./setup.sh
 streamlit run app.py
```

### 5) Mobile Setup
```
 Open the mobile folder in Android Studio.
 
 Sync Gradle and run the app on an emulator or device.
```
---

## 👩‍💻 Contributors
 - [evapatel1654](https://github.com/evapatel1654)
 - [Krish-P03](https://github.com/Krish-P03)
 - [Ashu-p07](https://github.com/Ashu-p07)
 
---





  
  





