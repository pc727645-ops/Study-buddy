# Study Buddy

Study Buddy is a simple RAG-based PDF question-answering app built with Gradio, LangChain, FAISS, and FLAN-T5.

## Features

- Upload a PDF file
- Split the document into chunks
- Create embeddings and store them in FAISS
- Ask questions about the uploaded PDF in a chat UI

## Tech Stack

- Python
- Gradio
- LangChain
- FAISS
- Hugging Face Transformers
- PyTorch

## Project Structure

```text
study buddy/
|-- app.py
|-- requirements.txt
|-- study_buddy.ipynb
|-- README.md
|-- .gitignore
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

## How It Works

1. Upload a PDF.
2. Click `Process Document`.
3. Ask questions in the chat panel.
4. The app retrieves relevant chunks from the PDF and generates an answer.

## Notes

- The first run may take time because the embedding model and text generation model need to download.
- GPU is used automatically if available; otherwise the app runs on CPU.

## GitHub Repo Name

Use the repository name: `study-buddy`
