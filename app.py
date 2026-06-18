from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

print("Reading and preparing your document, please wait...")

loader = PyPDFLoader("mydoc.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks, embeddings)

print("Ready! Your document has been loaded.")

llm = ChatGroq(model="llama-3.3-70b-versatile")

question = input("Ask a question about your document: ")

matching_chunks = vector_store.similarity_search(question, k=3)

context_text = ""
for chunk in matching_chunks:
    context_text = context_text + chunk.page_content + "\n\n"

prompt = "Here is some context from a document:\n\n" + context_text + "\n\nBased only on this context, answer this question: " + question + "\n\nIf the context doesn't contain the answer, say you don't know."

response = llm.invoke(prompt)

print("\n---- Answer ----")
print(response.content)