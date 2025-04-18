# AIzaSyCuYUYmAP3VF09d4sXZdpo6yl9mz8LcTmo    ... gemini api key
# import base64
# import os
# from google import genai
# from google.genai import types


# client = genai.Client(
#     api_key=os.environ.get("GEMINI_API_KEY"),)

# client = genai.Client(api_key="GEMINI_API_KEY")
# prompt = "Explain the concept of Occam's Razor and provide a simple, everyday example."
# response = client.models.generate_content(
#     model="gemini-2.0-flash-thinking-exp-01-21",
#     contents=prompt
# )

# print(response.text)
from google import genai
import os
import base64
import os
from google import genai
from google.genai import types
import streamlit as st
import os
GEMINI_API_KEY = "AIzaSyCuYUYmAP3VF09d4sXZdpo6yl9mz8LcTmo"  # ← replace this
# genai.configure(api_key=GEMINI_API_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

def check_acpc_related(file_path):
    try:
        uploaded_file = client.files.upload(file=file_path)

        content = types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type=uploaded_file.mime_type,
                ),
                types.Part.from_text(text="""

Please analyze the provided document content and determine its relationship to the following topics. Use deep understanding — not surface-level keyword matching.

For each topic, classify and justify:

- **Directly Related**: Is the document explicitly about this topic? Issued by relevant authorities? Contains official steps, announcements, or structured descriptions?
- **Indirectly Related**: Related through associated frameworks, follow-up steps, or secondary context?
- **Not Related**: Mentions topic but has no substantive relevance?

---

**Topics:**

1. **ACPC (Admission Committee for Professional Courses)**  
2. **Admission Counselling Process (in Gujarat)**  
3. **MYSY (Mukhyamantri Yuva Swavalamban Yojana)**  
4. **Any overall relationship to ACPC (summary)**

---

**Output Format:**

Analysis of [Document Title or Description]:

1. **ACPC:**

Directly Related to ACPC? [Yes/No]  
Justification: [...]

Indirectly Related to ACPC? [Yes/No]  
Justification: [...]

2. **Counselling Process:**

Directly Related to Counselling Process? [Yes/No]  
Justification: [...]

Indirectly Related to Counselling Process? [Yes/No]  
Justification: [...]

3. **MYSY:**

Directly Related to MYSY? [Yes/No]  
Justification: [...]

Indirectly Related to MYSY? [Yes/No]  
Justification: [...]

4. **Summary of Any Relationship to ACPC:**  
[List and explain all relevant connections, or say "No relationship found".]


---

Only include strong, validated relationships. Avoid false positives from surface mentions. Be strict.



""")
            ]
        )

        model = "gemini-2.0-flash-thinking-exp-01-21"
        response = client.models.generate_content(
            model=model,
            contents=[content],
            config = genai.types.GenerateContentConfig(
                temperature=0,
                top_p=0.95,
                top_k=20,
                # max_output_tokens=1024,
                system_instruction = """
                You are an intelligent assistant designed to analyze documents and determine their actual relevance to specific topics related to the Gujarat student admission ecosystem.

                Your task is to critically assess the content of the document — not just by keyword presence but by understanding the document's main purpose, context, and issuer. Avoid being misled by mere mentions of keywords like "ACPC", "MYSY", or "Counselling". A document is only relevant if it substantively discusses or is directly issued by related authorities, or serves as an official part of the described processes.

                Always follow these rules:
                - **DO NOT** consider a document related if it only contains a passing or indirect mention of a topic.
                - **DO** validate based on the document’s issuer, structure, purpose, and detailed content.
                - Focus on official processes, announcements, procedural steps, or eligibility information.
                - Justify every “Yes” or “No” with specific evidence **from the document** only.

                Your goal is to determine if a document should be included in a system that supports Question-Answering about ACPC, MYSY, and the counselling process in Gujarat.
                """

            
        )
        )
        return response.text.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# 🌐 Streamlit UI
st.set_page_config(page_title="ACPC Doc Checker", layout="centered")
st.title("📄 ACPC / MYSY Document Checker")

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_pdf:
    with open(uploaded_pdf.name, "wb") as f:
        f.write(uploaded_pdf.read())
    st.success(f"File uploaded: {uploaded_pdf.name}")

    with st.spinner("Analyzing with Gemini..."):
        result = check_acpc_related(uploaded_pdf.name)
        st.markdown("### ✅ Gemini Result")
        st.info(result)
