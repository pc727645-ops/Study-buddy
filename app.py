import gradio as gr
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
MODEL_NAME = "google/flan-t5-small"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
VECTOR_DB = None


def process_pdf(file):
    global VECTOR_DB

    if file is None:
        return "Error: Please upload a PDF first."

    try:
        loader = PyPDFLoader(file.name)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(documents)

        VECTOR_DB = FAISS.from_documents(chunks, EMBEDDINGS)
        return f"Success: PDF split into {len(chunks)} chunks and indexed."
    except Exception as exc:
        return f"Error processing PDF: {exc}"


def respond(message, history):
    del history
    global VECTOR_DB

    if VECTOR_DB is None:
        return "Please upload and process a PDF in the left panel first."

    results = VECTOR_DB.similarity_search(message, k=3)
    context = "\n".join(result.page_content for result in results)

    prompt = (
        "Answer the question using the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{message}\n\n"
        "Answer:"
    )

    inputs = TOKENIZER(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    outputs = MODEL.generate(**inputs, max_new_tokens=150)
    return TOKENIZER.decode(outputs[0], skip_special_tokens=True)


with gr.Blocks(theme=gr.themes.Soft(), title="Study Buddy") as demo:
    gr.Markdown("# Study Buddy")
    gr.Markdown("Upload a PDF, index it, and ask questions from the document.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Setup")
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("Process Document", variant="primary")
            status = gr.Textbox(
                label="System Status",
                placeholder="Awaiting upload...",
                interactive=False,
            )

        with gr.Column(scale=2):
            gr.Markdown("### Chat")
            gr.ChatInterface(
                fn=respond,
                examples=[
                    "What is the main topic?",
                    "Summarize the document",
                ],
                cache_examples=False,
            )

    process_btn.click(fn=process_pdf, inputs=file_input, outputs=status)


if __name__ == "__main__":
    demo.launch()
