import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pdf2image import convert_from_path
from pytesseract import image_to_string # OCR tool


# Chunking
def chunk_text(text, chunk_size=150):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])


# Smart pdf loader , Pdf | Images
def extract_text(path):
    print(">>> Loading PDF:", path)
    print(">>> File exists:", os.path.exists(path))

    # ---- 1. Try normal text extraction ----
    reader = PdfReader(path)
    print(">>> Total pages:", len(reader.pages))

    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        page_len = 0 if page_text is None else len(page_text)
        print(f">>> Page {i} normal text length:", page_len)

        if page_text:
            text += page_text + "\n"

    print(">>> Total extracted text length (normal):", len(text))

    # ---- 2. If readable, return it ----
    if text.strip():
        print(">>> ✅ Normal text detected. OCR not needed.")
        return text

    # ---- 3. OCR fallback ----
    print(">>> ⚠️ No readable text found. Switching to OCR...")

    images = convert_from_path(path)
    print(">>> Total pages converted to images:", len(images))

    ocr_text = ""
    for i, img in enumerate(images):
        page_text = image_to_string(img, lang="eng+fra")
        print(f">>> OCR page {i} text length:", len(page_text))
        ocr_text += page_text + "\n"

    print(">>> Total extracted text length (OCR):", len(ocr_text))
    return ocr_text

# Find pdfs 
def find_pdfs_recursive(root_folder):
    pdf_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for f in filenames:
            if f.lower().endswith(".pdf"):
                full_path = os.path.join(dirpath, f)
                pdf_files.append(full_path)
    return pdf_files


# Vectorize and store to ChromaDB
def vectorize_and_store(documents_dir, collection_name="docs"):

    client = chromadb.PersistentClient(path="./chroma")

    collection = client.get_or_create_collection(name=collection_name)
    
    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    doc_id = 0
    pdf_paths = find_pdfs_recursive('docs/s1')
    for path in pdf_paths:  
        filename = os.path.basename(path)
        print(f"Processing {path}")
        text = extract_text(path)
        if not text.strip():
            print("No text found, skipping.")
            continue
        
        chunks = list(chunk_text(text))
        print(f"Total chunks from {filename}: {len(chunks)}")

        # Embed all chunks in batch
        embeddings = model.encode(chunks, show_progress_bar=True)

        # Store in ChromaDB
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, embeddings=embeddings.tolist(), ids=ids)
        doc_id += 1
    

    print("All documents processed and saved.")

# -------------
# Run
# -------------
if __name__ == "__main__":
    documents_folder = "docs/s1"  # Change to your folder path
    vectorize_and_store(documents_folder)
