import io
import os
import re
import hashlib
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from sentence_transformers import SentenceTransformer


# -----------------------------
# App settings
# -----------------------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# -----------------------------
# Cached models / clients
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_groq_client(api_key):
    return Groq(api_key=api_key)


# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf(file_bytes, filename):
    """Extract PDF text page by page."""
    import fitz

    pages = []
    pdf = fitz.open(stream=file_bytes, filetype="pdf")

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(
                {
                    "text": text,
                    "filename": filename,
                    "page": page_number,
                }
            )

    return pages


def extract_docx(file_bytes, filename):
    """Extract DOCX paragraphs. DOCX does not expose reliable page numbers."""
    document = Document(io.BytesIO(file_bytes))
    text = "\n".join(
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return [
        {
            "text": text,
            "filename": filename,
            "page": None,
        }
    ] if text else []


def extract_txt(file_bytes, filename):
    """Extract TXT text."""
    text = file_bytes.decode("utf-8", errors="ignore").strip()

    return [
        {
            "text": text,
            "filename": filename,
            "page": None,
        }
    ] if text else []


def extract_md(file_bytes, filename):
    """Extract Markdown text."""
    text = file_bytes.decode("utf-8", errors="ignore").strip()

    return [
        {
            "text": text,
            "filename": filename,
            "page": None,
        }
    ] if text else []


def extract_document(file_bytes, filename):
    """Choose the correct extraction function."""
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)
    if extension == ".docx":
        return extract_docx(file_bytes, filename)
    if extension == ".txt":
        return extract_txt(file_bytes, filename)
    if extension == ".md":
        return extract_md(file_bytes, filename)

    raise ValueError(
        f"Unsupported file type: {extension}. "
        "Supported types are PDF, DOCX, TXT and MD."
    )


# -----------------------------
# Chunking
# -----------------------------
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Create overlapping word-based chunks."""
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def create_chunks(extracted_pages):
    """Create chunks while preserving filename and page metadata."""
    all_chunks = []

    for item in extracted_pages:
        text_chunks = split_text(item["text"])

        for chunk_number, chunk_text in enumerate(text_chunks, start=1):
            all_chunks.append(
                {
                    "text": chunk_text,
                    "filename": item["filename"],
                    "page": item["page"],
                    "chunk_id": len(all_chunks),
                    "local_chunk_id": chunk_number,
                }
            )

    return all_chunks


# -----------------------------
# Embeddings and FAISS
# -----------------------------
def create_embeddings(chunks, model):
    """Create normalized embeddings for all chunks."""
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.astype("float32")


def build_faiss_index(embeddings):
    """Build an inner-product FAISS index."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


# -----------------------------
# Keyword search
# -----------------------------
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than",
    "is", "are", "was", "were", "be", "been", "being", "to", "of",
    "in", "on", "at", "for", "from", "with", "by", "about", "as",
    "into", "what", "which", "who", "when", "where", "why", "how",
    "do", "does", "did", "can", "could", "would", "should", "i",
    "you", "we", "they", "it", "this", "that", "these", "those",
}


def important_words(text):
    """Return simple keyword candidates from a question."""
    words = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b", text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


def keyword_score(question, chunk_text):
    """Score a chunk by the proportion of question keywords it contains."""
    keywords = important_words(question)

    if not keywords:
        return 0.0

    chunk_lower = chunk_text.lower()
    matched = sum(1 for word in keywords if word in chunk_lower)

    return matched / len(keywords)


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, chunks, embeddings, index, model, top_k=TOP_K):
    """Combine FAISS semantic similarity and keyword matching."""
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    semantic_k = min(max(top_k * 3, 10), len(chunks))
    semantic_scores, semantic_ids = index.search(
        question_embedding, semantic_k
    )

    candidates = {}

    for score, chunk_id in zip(semantic_scores[0], semantic_ids[0]):
        if chunk_id == -1:
            continue

        candidates[int(chunk_id)] = {
            "semantic_score": float(score),
            "keyword_score": keyword_score(
                question, chunks[int(chunk_id)]["text"]
            ),
        }

    # Add chunks that contain question keywords, even if FAISS did not
    # place them in the initial candidate set.
    for chunk_id, chunk in enumerate(chunks):
        kw_score = keyword_score(question, chunk["text"])

        if kw_score > 0:
            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "semantic_score": 0.0,
                    "keyword_score": kw_score,
                }
            else:
                candidates[chunk_id]["keyword_score"] = kw_score

    ranked = []

    for chunk_id, scores in candidates.items():
        semantic = max(0.0, scores["semantic_score"])
        keyword = scores["keyword_score"]

        # Weighted hybrid score.
        combined = (0.75 * semantic) + (0.25 * keyword)

        result = chunks[chunk_id].copy()
        result["semantic_score"] = semantic
        result["keyword_score"] = keyword
        result["combined_score"] = combined
        ranked.append(result)

    ranked.sort(key=lambda item: item["combined_score"], reverse=True)

    return ranked[:top_k]


# -----------------------------
# Google Drive
# -----------------------------
def drive_id_from_url(url):
    """Extract a Google Drive file/folder ID."""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"/folders/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None


def download_drive_url(url):
    """Download a public Google Drive file using gdown."""
    output_dir = Path(tempfile.mkdtemp(prefix="drive_file_"))
    output_file = output_dir / "downloaded_file"

    result = gdown.download(
        url=url,
        output=str(output_file),
        quiet=True,
    )

    if not result or not Path(result).exists():
        raise ValueError(
            "Could not download the Google Drive file. "
            "Make sure the file is shared so the app can access it."
        )

    return Path(result)


def download_drive_folder(url):
    """Download files from a public Google Drive folder."""
    output_dir = Path(tempfile.mkdtemp(prefix="drive_folder_"))

    downloaded = gdown.download_folder(
        url=url,
        output=str(output_dir),
        quiet=True,
        use_cookies=False,
    )

    if not downloaded:
        return []

    files = []

    for item in output_dir.rglob("*"):
        if item.is_file() and item.suffix.lower() in {
            ".pdf",
            ".docx",
            ".txt",
            ".md",
        }:
            files.append(item)

    return files


def load_drive_documents(url):
    """Load supported files from a Drive file or folder URL."""
    url = url.strip()

    if not url:
        raise ValueError("Please enter a Google Drive link.")

    drive_id = drive_id_from_url(url)

    if not drive_id:
        raise ValueError(
            "The Google Drive link format was not recognized."
        )

    if "/folders/" in url:
        paths = download_drive_folder(url)

        if not paths:
            raise ValueError(
                "No supported files were found in the Drive folder. "
                "Supported types: PDF, DOCX, TXT and MD."
            )

    else:
        paths = [download_drive_url(url)]

    documents = []

    for path in paths:
        extension = path.suffix.lower()

        if extension not in {
            ".pdf",
            ".docx",
            ".txt",
            ".md",
        }:
            continue

        with open(path, "rb") as file:
            file_bytes = file.read()

        filename = path.name

        if extension == "":
            filename = f"drive_file_{drive_id}"

        documents.append(
            {
                "filename": filename,
                "bytes": file_bytes,
                "source": "Google Drive",
            }
        )

    return documents

# -----------------------------
# Document processing
# -----------------------------
def file_hash(file_bytes, filename):
    """Create a stable ID for a document."""
    hasher = hashlib.sha256()
    hasher.update(filename.encode("utf-8"))
    hasher.update(file_bytes)
    return hasher.hexdigest()


def process_documents(documents, model):
    """Extract, chunk and embed documents once."""
    extracted_pages = []
    document_info = []

    for document in documents:
        filename = document["filename"]
        file_bytes = document["bytes"]

        pages = extract_document(file_bytes, filename)
        extracted_pages.extend(pages)

        full_text = "\n".join(page["text"] for page in pages)

        document_info.append(
            {
                "filename": filename,
                "source": document.get("source", "Local upload"),
                "file_type": Path(filename).suffix.lower().replace(".", "").upper(),
                "characters": len(full_text),
                "words": len(full_text.split()),
                "pages": len(pages) if Path(filename).suffix.lower() == ".pdf" else None,
                "document_id": file_hash(file_bytes, filename),
            }
        )

    chunks = create_chunks(extracted_pages)

    if not chunks:
        raise ValueError("No readable text was found in the documents.")

    embeddings = create_embeddings(chunks, model)
    index = build_faiss_index(embeddings)

    return document_info, chunks, embeddings, index


def reset_document_store():
    """Clear all processed documents."""
    for key in [
        "document_info",
        "chunks",
        "embeddings",
        "faiss_index",
        "processed_ids",
    ]:
        st.session_state.pop(key, None)


# -----------------------------
# Groq answer generation
# -----------------------------
def answer_question(question, retrieved_chunks, client):
    """Ask Groq to answer only from retrieved context."""
    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):
        page = chunk["page"]
        page_text = f", page {page}" if page is not None else ""

        context_parts.append(
            f"[Source {number}: {chunk['filename']}{page_text}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a document question-answering assistant.

Answer the user's question ONLY using the supplied document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume facts that are not in the context.
3. If the answer is not available in the context, clearly say:
   "The information is not available in the provided documents."
4. Give a concise, direct answer.
5. When useful, mention the source filename and page number.
"""

    user_prompt = f"""Document context:

{context}

User question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=800,
    )

    return response.choices[0].message.content


# -----------------------------
# Session state
# -----------------------------
if "document_info" not in st.session_state:
    st.session_state.document_info = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "processed_ids" not in st.session_state:
    st.session_state.processed_ids = set()


# -----------------------------
# UI
# -----------------------------
st.title("📄 AI Document Assistant")
st.write(
    "Upload documents or load them from Google Drive, then ask questions "
    "using semantic + keyword hybrid search."
)

with st.sidebar:
    st.header("Settings")
    st.caption(f"Embedding model: `{EMBEDDING_MODEL}`")
    st.caption(f"Chunk size: {CHUNK_SIZE} words")
    st.caption(f"Chunk overlap: {CHUNK_OVERLAP} words")
    st.caption(f"Retrieved chunks: {TOP_K}")

    if st.button("🗑️ Clear processed documents"):
        reset_document_store()
        st.rerun()

    st.divider()
    st.subheader("Groq API")
    if "GROQ_API_KEY" in st.secrets:
        st.success("Groq API key is configured.")
    else:
        st.warning(
            "GROQ_API_KEY is not configured. Add it in "
            "Streamlit Secrets before asking questions."
        )

tab_upload, tab_drive = st.tabs(["📤 Local Upload", "☁️ Google Drive"])

new_documents = []

with tab_upload:
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()

            new_documents.append(
                {
                    "filename": uploaded_file.name,
                    "bytes": file_bytes,
                    "source": "Local upload",
                }
            )

        if st.button("Process uploaded documents", type="primary"):
            model = load_embedding_model()

            with st.spinner("Extracting text, creating chunks and embeddings..."):
                try:
                    # Only process documents that have not already been processed.
                    documents_to_process = []

                    for document in new_documents:
                        doc_id = file_hash(
                            document["bytes"], document["filename"]
                        )

                        if doc_id not in st.session_state.processed_ids:
                            documents_to_process.append(document)

                    if not documents_to_process:
                        st.info("These documents are already processed.")
                    else:
                        (
                            document_info,
                            chunks,
                            embeddings,
                            index,
                        ) = process_documents(
                            documents_to_process, model
                        )

                        # This simple version rebuilds the combined index only
                        # when new documents are added, not on every question.
                        if st.session_state.chunks:
                            all_chunks = st.session_state.chunks + chunks
                            all_embeddings = np.vstack(
                                [
                                    st.session_state.embeddings,
                                    embeddings,
                                ]
                            ).astype("float32")

                            all_index = build_faiss_index(all_embeddings)

                            st.session_state.chunks = all_chunks
                            st.session_state.embeddings = all_embeddings
                            st.session_state.faiss_index = all_index
                            st.session_state.document_info.extend(
                                document_info
                            )
                        else:
                            st.session_state.chunks = chunks
                            st.session_state.embeddings = embeddings
                            st.session_state.faiss_index = index
                            st.session_state.document_info = document_info

                        for document in documents_to_process:
                            st.session_state.processed_ids.add(
                                file_hash(
                                    document["bytes"], document["filename"]
                                )
                            )

                        st.success(
                            f"Processed {len(documents_to_process)} new document(s)."
                        )

                except Exception as error:
                    st.error(f"Processing failed: {error}")


with tab_drive:
    drive_url = st.text_input(
        "Paste a public Google Drive file or folder link",
        placeholder="https://drive.google.com/file/d/... or /folders/...",
    )

    if st.button("Load from Google Drive"):
        try:
            with st.spinner("Loading Google Drive documents..."):
                drive_documents = load_drive_documents(drive_url)

            model = load_embedding_model()

            with st.spinner("Extracting text, creating chunks and embeddings..."):
                documents_to_process = []

                for document in drive_documents:
                    doc_id = file_hash(
                        document["bytes"], document["filename"]
                    )

                    if doc_id not in st.session_state.processed_ids:
                        documents_to_process.append(document)

                if not documents_to_process:
                    st.info("These Drive documents are already processed.")
                else:
                    (
                        document_info,
                        chunks,
                        embeddings,
                        index,
                    ) = process_documents(
                        documents_to_process, model
                    )

                    if st.session_state.chunks:
                        all_chunks = st.session_state.chunks + chunks
                        all_embeddings = np.vstack(
                            [
                                st.session_state.embeddings,
                                embeddings,
                            ]
                        ).astype("float32")

                        st.session_state.chunks = all_chunks
                        st.session_state.embeddings = all_embeddings
                        st.session_state.faiss_index = build_faiss_index(
                            all_embeddings
                        )
                        st.session_state.document_info.extend(
                            document_info
                        )
                    else:
                        st.session_state.chunks = chunks
                        st.session_state.embeddings = embeddings
                        st.session_state.faiss_index = index
                        st.session_state.document_info = document_info

                    for document in documents_to_process:
                        st.session_state.processed_ids.add(
                            file_hash(
                                document["bytes"], document["filename"]
                            )
                        )

                    st.success(
                        f"Loaded {len(documents_to_process)} new Drive document(s)."
                    )

        except Exception as error:
            st.error(f"Google Drive loading failed: {error}")


# -----------------------------
# Document information
# -----------------------------
if st.session_state.document_info:
    st.divider()
    st.subheader("📊 Extracted Document Information")

    for info in st.session_state.document_info:
        with st.expander(f"📄 {info['filename']}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**Source:** {info['source']}")
                st.write(f"**Type:** {info['file_type']}")

            with col2:
                st.write(f"**Words:** {info['words']:,}")
                st.write(f"**Characters:** {info['characters']:,}")

            with col3:
                pages = info["pages"]
                st.write(
                    f"**Pages:** {pages if pages is not None else 'N/A'}"
                )

    st.info(
        f"📦 **{len(st.session_state.chunks):,} chunks created** "
        f"and stored in the current session."
    )


# -----------------------------
# Question answering
# -----------------------------
st.divider()
st.subheader("💬 Ask Your Documents")

question = st.text_input(
    "Ask a question",
    placeholder="What does the document say about...?",
)

if st.button("🔎 Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not st.session_state.chunks:
        st.warning("Please upload or load a document first.")
    elif "GROQ_API_KEY" not in st.secrets:
        st.error(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets."
        )
    else:
        try:
            model = load_embedding_model()
            client = get_groq_client(st.secrets["GROQ_API_KEY"])

            with st.spinner("Searching documents..."):
                results = hybrid_search(
                    question,
                    st.session_state.chunks,
                    st.session_state.embeddings,
                    st.session_state.faiss_index,
                    model,
                    TOP_K,
                )

            with st.spinner("Generating answer..."):
                answer = answer_question(
                    question,
                    results,
                    client,
                )

            st.subheader("Answer")
            st.write(answer)

            st.subheader("📚 Retrieved Sources")

            for number, result in enumerate(results, start=1):
                page = (
                    f"Page {result['page']}"
                    if result["page"] is not None
                    else "Page not available"
                )

                with st.expander(
                    f"{number}. {result['filename']} — {page}"
                ):
                    score_col1, score_col2, score_col3 = st.columns(3)

                    with score_col1:
                        st.metric(
                            "Hybrid score",
                            f"{result['combined_score']:.3f}",
                        )

                    with score_col2:
                        st.metric(
                            "Semantic score",
                            f"{result['semantic_score']:.3f}",
                        )

                    with score_col3:
                        st.metric(
                            "Keyword score",
                            f"{result['keyword_score']:.3f}",
                        )

                    st.write(result["text"])

        except Exception as error:
            st.error(f"Question answering failed: {error}")
