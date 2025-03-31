**Student Counselling Chatbot**

**📌 Overview**

This project is an intelligent virtual assistant designed to streamline the admission process for educational institutions in Gujarat. The chatbot supports multiple languages and leverages advanced Large Language Models (LLMs) to provide dynamic, context-aware responses. It aims to eliminate language barriers, reduce administrative workload, and enhance accessibility for students and parents.

**✨ Features**

-> Multilingual Support: Provides assistance in multiple languages to cater to diverse users.

-> Document Processing: Allows users to upload and extract information from PDFs and images using OCR (Tesseract).

-> User Authentication: Secure login/signup using JWT tokens.

-> Chat History: Stores past interactions for future reference.

-> Web & Mobile Applications: Available on both web (MERN stack) and mobile (Android) platforms.

-> Integration with LLMs: Uses LangChain and Groq for efficient query handling.

**🛠️ Technologies Used**

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

 ** Mobile (Android)**
 
  Android Studio: For app development.
  
  SQLite: Local database for chat history and feedback.
  
  Shared Preferences: For managing user settings.

**🚀 Getting Started**

  **Prerequisites**
  
  Node.js (v16 or later)
  
  MongoDB Atlas account
  
  Python (for model-related scripts)
  
  Android Studio (for mobile development)


**📸 Screenshots**

  **1) Web Application : **

  Main Page :
  
  ![WhatsApp Image 2024-11-15 at 17 18 51_5d7781fb](https://github.com/user-attachments/assets/59991cca-f8a2-4f4e-9a20-8716692f731a)

  Sign-up Page :

  ![WhatsApp Image 2024-11-15 at 17 18 51_81982321](https://github.com/user-attachments/assets/c7f44729-9d69-4921-b243-4e24004db744)

  Login Page :

  ![WhatsApp Image 2024-11-15 at 17 18 51_d65d0101](https://github.com/user-attachments/assets/8d8f69a8-5f23-4502-95fa-f6d3245ec606)

  Chat Page :

  ![WhatsApp Image 2024-11-15 at 17 24 06_58b40ee6](https://github.com/user-attachments/assets/a8bdbce8-ce35-4ef6-a234-b59acca07abd)


  **2) Mobile Application :**

  Main Page :
  
  ![WhatsApp Image 2024-11-15 at 22 09 33_dc852104](https://github.com/user-attachments/assets/d87c4dc3-7aa6-43ca-9c34-645f7b6573c0)

  Navigation Toolbar:
  
  ![WhatsApp Image 2024-11-15 at 22 09 34_d48ef7d5](https://github.com/user-attachments/assets/5a7eae14-c816-45ea-81f5-eb2cdaaa0c11)

  
  Chat Page:
  
  ![WhatsApp Image 2024-11-15 at 22 09 34_65ae0163](https://github.com/user-attachments/assets/a7bc29f8-c5da-45a1-89cf-2d26f967671a)

  Chat History :
  
  ![WhatsApp Image 2024-11-15 at 22 09 35_09244281](https://github.com/user-attachments/assets/3a6e8923-d454-4842-a0a0-efab064b095e)

  ![WhatsApp Image 2024-11-15 at 22 09 35_a92e8a02](https://github.com/user-attachments/assets/61f0616d-f99b-4c31-8d0a-a47d362f94bb)

  FAQ :

  ![WhatsApp Image 2024-11-15 at 22 09 36_00af2651](https://github.com/user-attachments/assets/5bb191d2-2f1e-4d0c-a079-b95d24b63b8d)

  Theme Setting :

  ![WhatsApp Image 2024-11-15 at 22 09 36_3a501042](https://github.com/user-attachments/assets/0bf26a5a-4c8c-41bd-bfcd-58bd3e36c45d)

  Feedback + Submitted Notification : 

  ![WhatsApp Image 2024-11-15 at 22 09 37_eb21f94d](https://github.com/user-attachments/assets/656dee5e-cdad-441e-9b52-86608852c00a)

  ![WhatsApp Image 2024-11-15 at 22 09 37_818bf995](https://github.com/user-attachments/assets/b9a4cd62-ac10-4557-bbcb-012c6067efd9)

  **3) Dashboard/ Streamlit version of Document Upload Feature :**

  ![OCR1](https://github.com/user-attachments/assets/086a8ea1-bc23-4fda-9075-de0257c71865)
  
  ![OCR2](https://github.com/user-attachments/assets/461a2869-947d-4b25-9b6e-311161ee9d60)

  ![OCR3](https://github.com/user-attachments/assets/c93e7d92-8848-4cbe-be0a-c2f27b69aaff)


**Installation guide**

**1) Clone the Repository**
  
  git clone https://github.com/neha13rana/Student_Counselling_Chatbot.git

    
**2) Backend Setup**

  select web app backend from the branch
  cd _
  npm install
  cp .env.example .env  # Update environment variables
  npm start

**3) Frontend Setup**

  select web app frontend from the branch
  cd _
  npm install
  npm run dev

**4) Model Setup (API / Dashboard)**

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
  
**5) Mobile Setup**
    
  Open the mobile folder in Android Studio.
  
  Sync Gradle and run the app on an emulator or device.







  
  





