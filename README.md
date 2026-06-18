# Resume RAG Chatbot

A chatbot that answers questions about a PDF document using RAG (Retrieval-Augmented Generation).

## What it does
- Loads a PDF document and splits it into small chunks
- Converts each chunk into a vector embedding using a free Hugging Face model
- Stores the embeddings in a Chroma vector database
- When asked a question, retrieves the most relevant chunks and sends them to an LLM (via Groq) to generate a grounded answer
- Designed to say "I don't know" rather than hallucinate when the document doesn't contain the answer

## Tech stack
Python, LangChain, ChromaDB, Hugging Face Sentence Transformers, Groq (Llama 3.3)

## How to run it
1. Install dependencies: `pip install -r requirements.txt`
2. Add your own PDF file named `mydoc.pdf` in the project folder
3. Create a `.env` file with your own `GROQ_API_KEY`
4. Run: `python app.py`
