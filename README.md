📄 AI Document Assistant
A simple Streamlit RAG-style document assistant that lets you upload documents or load supported files from Google Drive, search them with semantic + keyword hybrid search, and ask questions using Groq.
Features
PDF, DOCX, TXT and MD upload
Separate extraction functions for each format
PDF page-number preservation
Filename and metadata preservation
Overlapping text chunking
Sentence Transformers embeddings
FAISS vector search
Simple keyword search
Hybrid semantic + keyword ranking
Groq-powered question answering
Answers are restricted to retrieved document context
Retrieved sources shown below every answer
Google Drive file/folder loading
Local uploads and Google Drive use the same processing pipeline
Session state prevents document embeddings from being recreated for every question
Cached embedding model and Groq client
Project files
```text
document-assistant/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```
1. Get a Groq API key
Create a Groq API key from the Groq developer console.
Do not put the API key inside `app.py`.
For Streamlit Cloud, add the secret:
```toml
GROQ_API_KEY = "your-groq-api-key"
```
In Streamlit Cloud:
App → Manage app → Settings → Secrets
Paste the secret there and save.
For local development, you can also create:
```text
.streamlit/secrets.toml
```
with:
```toml
GROQ_API_KEY = "your-groq-api-key"
```
Never commit this file to GitHub.
2. Install dependencies
If you are running locally:
```bash
pip install -r requirements.txt
```
Then:
```bash
streamlit run app.py
```
The first use of Sentence Transformers may download the embedding model.
3. Deploy on Streamlit Community Cloud
Create a GitHub repository.
Add:
`app.py`
`requirements.txt`
`README.md`
`.gitignore`
Open Streamlit Community Cloud.
Create a new app.
Select your GitHub repository.
Select `app.py` as the main file.
Deploy.
Open the app's settings.
Open Secrets.
Add:
```toml
GROQ_API_KEY = "your-groq-api-key"
```
Save the secret.
Restart/redeploy the app if necessary.
4. How the app works
Step 1 — Extraction
The app chooses an extraction function based on the file extension:
```text
PDF  → extract_pdf()
DOCX → extract_docx()
TXT  → extract_txt()
MD   → extract_md()
```
PDF pages keep their page number. DOCX, TXT and MD use `page=None` because those formats do not reliably provide page numbers through this simple extraction pipeline.
Step 2 — Chunking
The extracted text is divided into overlapping chunks.
Current defaults:
```text
Chunk size: 900 words
Overlap:    150 words
```
Each chunk keeps:
```text
filename
page
chunk_id
local_chunk_id
text
```
Step 3 — Embeddings
The app uses:
```text
sentence-transformers/all-MiniLM-L6-v2
```
Each chunk is converted into a vector.
The embeddings are stored in Streamlit session state.
Step 4 — FAISS
FAISS stores the document vectors and performs semantic similarity search.
The document embeddings are not recreated when the user asks another question.
Only the question itself needs a new embedding.
Step 5 — Keyword search
The app extracts important words from the question, removes common stop words, and checks how many important words occur in each candidate chunk.
Step 6 — Hybrid search
The app combines:
```text
75% semantic similarity
25% keyword score
```
The resulting chunks are ranked by the combined score.
Step 7 — Groq
The top retrieved chunks are sent to Groq along with the question.
The system instruction tells the model:
use only the supplied context
do not use outside knowledge
do not invent information
explicitly say when the information is unavailable
Step 8 — Sources
After the answer, the app displays the retrieved chunks including:
```text
Filename
Page number when available
Hybrid score
Semantic score
Keyword score
Retrieved text
```
Google Drive
The app accepts Google Drive file and folder links.
For individual files, the app downloads into a temporary directory instead of
forcing a filename such as `downloaded_file`. This is important because the
real extension (`.pdf`, `.docx`, `.txt`, or `.md`) is needed by the document
extraction pipeline. Current gdown versions can resolve the original filename
from a Google Drive share URL.

The Drive content needs to be accessible to the app, normally through an appropriate sharing setting.
Supported formats:
```text
PDF
DOCX
TXT
MD
```
A Drive folder is searched for supported files, and each loaded document goes through the same:
```text
Drive
 ↓
Extraction
 ↓
Chunking
 ↓
Embedding
 ↓
FAISS
 ↓
Hybrid search
 ↓
Groq
```
Important limitation: document persistence
This version keeps processed chunks, embeddings and the FAISS index in Streamlit session state.
That means:
asking multiple questions does not recreate document embeddings
changing pages/tabs during the same session keeps the processed data
starting a completely new session does not permanently preserve the index
restarting the Streamlit app requires documents to be processed again
For a beginner-friendly application, this is intentional.
A future production version could persist:
```text
FAISS index
Chunk metadata
Embeddings
Document hashes
```
to disk, object storage, or a vector database.
Security
Never hardcode:
```python
GROQ_API_KEY = "..."
```
inside `app.py`.
Use Streamlit Secrets instead:
```toml
GROQ_API_KEY = "..."
```
Also never commit `.streamlit/secrets.toml` to GitHub.
Troubleshooting
`ModuleNotFoundError: faiss`
Make sure `requirements.txt` contains:
```text
faiss-cpu
```
Then redeploy the Streamlit app.
Groq API key missing
Check:
Streamlit Cloud → Manage app → Settings → Secrets
and make sure the key is exactly:
```toml
GROQ_API_KEY = "your-key"
```
Google Drive errors
The project uses a current gdown version and does not use the removed
`fuzzy=True` parameter.
The app also downloads individual Drive files into a temporary directory so
gdown can preserve the original filename and extension. Without the extension,
the app could download the file successfully but then skip it because it could
not identify whether it was PDF, DOCX, TXT or MD.
If you already have an older copy of the project, replace both `app.py` and
`requirements.txt` with the current versions from this project, then commit
and redeploy.
Google Drive file cannot be loaded
Check that:
The URL is a Google Drive file/folder URL.
The file or folder can be accessed by the app.
The document is PDF, DOCX, TXT or MD.
The Drive URL is not a restricted private resource that the app cannot access.
Scanned PDF
PyMuPDF extracts text from text-based PDFs. A scanned PDF may contain only images, in which case OCR is required. OCR is intentionally not included in this simple version.
Suggested next upgrades
Persistent FAISS index
OCR for scanned PDFs
Better DOCX page/section metadata
CSV/XLSX support
Conversation history
Streaming Groq responses
Source highlighting
Document deletion
Per-document search filters
Authentication
Persistent cloud storage
