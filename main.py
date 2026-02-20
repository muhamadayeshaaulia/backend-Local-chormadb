from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import ollama
from langchain_core.prompts import PromptTemplate

app = FastAPI(title="API Asisten Akademik Kampus Global Institute")
chat_history = []

print("1. Memuat Database Vektor...")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

print("2. Memuat Model AI (Ollama - Llama 3.2)...")
llm = ollama.Ollama(model="llama3.2",num_thread=8)

class ChatRequest(BaseModel):
    question: str

template = """Kamu adalah Assistent resmi Global Institute of Technology. 
Tugasmu adalah menjawab berdasarkan DATA BERIKUT dengan lengap. JANGAN PERNAH MENGARANG DATA DI LUAR INI.

[DATA PENDAFTARAN]
1. Isi formulir online: https://global.ac.id/form-online-global-institute/
2. Dihubungi Tim Marketing via Telp/WA/Email
3. Bayar biaya pendaftaran (Cash/Cicil)
4. Upload dokumen (KTP, KK, Ijazah, Transkrip, Pas Foto)

[DATA KELAS & WAKTU]
- Reguler: Senin-Jumat (08.00-12.30)
- Non Reguler: Senin-Jumat (18.00-21.30)
- Shift: Pagi (08.00-12.30) & Malam (18.00-21.30)
- Blended Learning: Offline & Online
- Eksekutif: Hubungi Arnie (081315198308)

[DATA BEASISWA]
1. Beasiswa 10 Besar (Peringkat 10 besar di sekolah)
2. Beasiswa Putra/i Guru (Anak dari guru TK-SMA)
3. Beasiswa Prestasi (Akademik/Non-akademik, s.d 100%)
4. Beasiswa KIP (Gratis SPP + Uang Saku)
5. Beasiswa Yayasan (Berdasarkan ujian seleksi, s.d 100%)

[DATA PRODI/JURUSAN]
- FTIK: Teknik Informatika (SE, IoT, CDM) & Sistem Informasi (AIS, MIS, LPIS)
- Bisnis: Bisnis Digital (DMC, DBT, DBM) & Manajemen Retail (MSDM, MP, MRS)

DOKUMEN REFERENSI TAMBAHAN:
{context}

RIWAYAT CHAT:
{history}

INSTRUKSI:
- Jawablah dengan ramah dan poin-poin agar mudah dibaca.
- Jika user bertanya hal umum (misal: "cara daftar"), berikan semua langkahnya secara lengkap.

PERTANYAAN: {question}
JAWABAN:"""

prompt_template = PromptTemplate(
    template=template, 
    input_variables=["context", "history", "question"]
)

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    global chat_history
    print(f"\n[USER]: {request.question}")
    docs = db.similarity_search(request.question, k=5)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    history_text = "\n".join(chat_history[-4:]) 
    
    prompt_akhir = prompt_template.format(
        context=context_text, 
        history=history_text if history_text else "Tidak ada.",
        question=request.question
    )
    
    print("[SISTEM]: assistant lagi mengetik...")
    jawaban = llm.invoke(prompt_akhir)
    
    chat_history.append(f"User: {request.question}")
    chat_history.append(f"AI: {jawaban}")
    
    print(f"[AI]: {jawaban}")
    return {"answer": jawaban}

@app.post("/reset")
async def reset_chat():
    global chat_history
    chat_history = []
    return {"message": "Memory dihapus!"}