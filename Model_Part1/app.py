import gradio as gr
import torch
from model2 import ChatbotModel  # Assuming your model code is saved as chatbot_model.py

# Load the chatbot model (replace "model.pt" with your actual saved model path if needed)
chatbot = ChatbotModel.load("model.pt")

def generate_response(user_input):
    return chatbot.get_response(user_input)

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="ACPC Counseling Chatbot",
    description="Chatbot assistant for queries related to ACPC counseling, MYSY scholarships, and admissions."
)

if __name__ == "__main__":
    # Launch the Gradio interface
    iface.launch()
