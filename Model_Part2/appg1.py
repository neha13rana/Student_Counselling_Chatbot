import gradio as gr
import os
import base64
from PIL import Image
from io import BytesIO

def handle_uploaded_file(uploaded_file, show_in_sidebar=False):
    # Determine the file extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Check if the uploaded file is in 'NamedString' format (Gradio sometimes returns it this way)
    if isinstance(uploaded_file, gr.File):
        # In this case, read the file directly from the 'data' attribute
        file_data = uploaded_file.read()  # This is the file content in bytes

        # Save the file content to a local file
        with open(file_path, "wb") as f:
            f.write(file_data)

    # Prepare content to display in the sidebar
    sidebar_content = ""

    if file_extension == ".pdf":
        # Read and encode the PDF as base64 to embed in the sidebar
        with open(file_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        sidebar_content = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="500" height="500"></iframe>'
    
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Display image in the sidebar
        img = Image.open(file_path)
        img_byte_array = BytesIO()
        img.save(img_byte_array, format="PNG")
        img_byte_array.seek(0)
        sidebar_content = gr.Image(value=img_byte_array, label="Uploaded Image", type="file")
    
    else:
        # For text files, show the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        sidebar_content = content

    return sidebar_content

# Gradio interface setup
def upload_and_display(file):
    sidebar_content = handle_uploaded_file(file, show_in_sidebar=True)
    return sidebar_content

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):  # Sidebar
            gr.Markdown("### Uploaded File Content")
            file_input = gr.File(label="Upload Your Document", file_types=[".pdf", ".jpg", ".jpeg", ".png", ".txt", ".bmp"])
            file_output = gr.HTML()

        file_input.change(fn=upload_and_display, inputs=file_input, outputs=file_output)

# Launch Gradio app
demo.launch()
