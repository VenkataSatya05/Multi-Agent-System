from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ← fixed import
from langchain_ollama import OllamaEmbeddings                        # ← use Ollama
from langchain_community.vectorstores import FAISS

def build_vector_store(pdf_path: str, persist_path: str = "vectorstore"):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(persist_path)
    print(f"✅ Vector store built with {len(chunks)} chunks.")
    return store