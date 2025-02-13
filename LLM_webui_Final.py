import os
import sys
import threading
import logging
from flask import Flask, render_template_string, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import requests

# Disable Flask logs
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# Suppress Flask startup logs
cli = logging.getLogger("flask.cli")
cli.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

history = []  # Stores conversation history
history_lock = threading.Lock()
mode = "chat"  # Default mode
retriever = None
qa = None

SYSTEM_PROMPT = """You are an AI assistant capable of both general conversation and document-based retrieval.
- In Chat Mode: Engage in friendly, informative conversation based on your knowledge.
- In RAG Mode: Strictly use provided documents (blogs, PDFs) to answer user queries.
- Maintain conversation history across interactions.
- Always be concise, accurate, and engaging."""

llm = OllamaLLM(model="llama3.1", system=SYSTEM_PROMPT)


# Load PDF for RAG Mode
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")
    return FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )


@app.route("/", methods=["GET"])
def index():
    with history_lock:
        formatted_history = "".join(
            [
                f'<div class="message user">{h[0]}</div><div class="message llm">{h[1]}</div>'
                for h in history
            ]
        )

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Chat</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .message {{ margin: 10px; padding: 8px; border-radius: 10px; max-width: 70%; clear: both; }}
            .user {{ background: #f0f8ff; color: #1e90ff; float: right; text-align: right; }}
            .llm {{ background: #f0fff0; color: #228b22; float: left; text-align: left; }}
            #history {{ clear: both; overflow-y: auto; max-height: 400px; border: 1px solid #ccc; padding: 10px; }}
        </style>
    </head>
    <body>
        <h2>Chat History</h2>
        <div id="history">{formatted_history}</div>
    </body>
    </html>
    """
    return render_template_string(html_template)


def main():
    print("Starting LLM Chat with WebUI (http://localhost:5000)")
    print("\n" + "=" * 50)
    print("LLM Chat with RAG Mode Support")
    print("=" * 50)
    print("Commands:")
    print("- Type 'switch to rag' to enable RAG mode")
    print("- Type 'switch to chat' to return to normal chat")
    print("- Type '[exit or quit or bye or q]' to quit\n")

    # threading.Thread(
    #     target=lambda: app.run(port=5001, threaded=True, use_reloader=False),
    #     daemon=True,
    # ).start()

    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    threading.Thread(
        target=lambda: app.run(port=5000, threaded=True, use_reloader=False),
        daemon=True,
    ).start()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    global mode, retriever, qa

    while True:
        query = input("\nYou: ").strip()

        if query.lower() in ["exit", "quit", "bye", "q"]:
            print("Goodbye! Have a great time ahead!")
            break

        if query.lower().startswith("switch to"):
            new_mode = query.split()[-1].lower()
            if new_mode == "rag":
                pdf_url = input("Enter PDF URL for RAG: ")
                pdf_path = "downloaded.pdf"
                response = requests.get(pdf_url)
                if response.status_code == 200:
                    with open(pdf_path, "wb") as f:
                        f.write(response.content)
                    retriever = load_pdf(pdf_path).as_retriever()
                    qa = RetrievalQA.from_chain_type(
                        llm=llm, chain_type="stuff", retriever=retriever
                    )
                    mode = "rag"
            else:
                mode = "chat"
            print(f"Switched to {mode.upper()} mode")
            continue
        with history_lock:
            formatted_history = "\n".join(
                [f"User: {h[0]}\nLLM: {h[1]}" for h in history[-5:]]
            )

        full_prompt = f"{formatted_history}\nUser: {query}\nLLM:"
        # response = qa.run(query) if mode == "rag" and qa else llm.invoke(full_prompt)
        if mode == "chat":
            response = llm.invoke(full_prompt)
        elif mode == "rag" and qa:
            response = qa.run(full_prompt)
        else:
            response = "RAG mode not initialized. Switch to chat or provide PDF URL."

        with history_lock:
            history.append((query, response))

        print(f"LLM: {response}")


if __name__ == "__main__":
    main()
