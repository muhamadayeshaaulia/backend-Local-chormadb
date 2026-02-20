import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("1. Membaca semua file di folder 'data_dokumen'...")
folder_path = "./data_dokumen"
semua_dokumen = []

# memeriksa setiap file di dalam folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Jika file-nya PDF, pakai PyPDFLoader
    if filename.endswith(".pdf"):
        print(f" -> Membaca PDF: {filename}")
        loader = PyPDFLoader(file_path)
        semua_dokumen.extend(loader.load())
        
    # Jika file-nya TXT, pakai TextLoader
    elif filename.endswith(".txt"):
        print(f" -> Membaca TXT: {filename}")
        loader = TextLoader(file_path, encoding="utf-8")
        semua_dokumen.extend(loader.load())

print(f"Total dokumen berhasil dibaca: {len(semua_dokumen)} halaman/bagian.")

print("\n2. Memotong teks (Chunking)...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(semua_dokumen)
print(f"Teks berhasil dipotong menjadi {len(chunks)} bagian (chunks).")

print("\n3. Mengubah ke Vektor dan menyimpan ke ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# menyimpan hasil ke folder bernama 'chroma_db'
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

print("\nSelesai! Semua file PDF dan TXT berhasil disimpan ke VectorDB lokal.")