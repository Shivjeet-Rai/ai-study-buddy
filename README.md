# 📘 AI Study Buddy

AI Study Buddy is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF study notes and ask natural language questions about them. The system retrieves relevant content from the document and generates grounded answers using a large language model.

---

## 🚀 Features
- Upload PDF-based study notes
- Ask questions in natural language
- Context-aware answers grounded in the document
- Prevents hallucinations by restricting answers to retrieved content
- Clean and interactive UI built with Streamlit

---

## 🧠 How It Works
1. Extracts text from the uploaded PDF
2. Splits text into overlapping chunks
3. Converts chunks into vector embeddings
4. Stores embeddings in a FAISS vector index
5. Retrieves relevant chunks for a question
6. Uses a Gemini LLM to generate grounded answers

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Frontend:** Streamlit
- **PDF Processing:** PyPDF
- **Embeddings:** Hugging Face Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database:** FAISS
- **LLM:** Google Gemini (gemini-2.5-flash-lite)

---

## ⚙️ How to Run Locally (Complete Guide)

### Step 1: Install Python
- Download Python 3.10 or higher from:
  https://www.python.org/downloads/
- During installation, check “Add Python to PATH”
- Verify installation:
```bash
python --version
```

---

### Step 2: Clone the repository
```bash
git clone <your-repo-link>
cd ai-study-buddy
```

---

### Step 3: Create a virtual environment
```bash
python -m venv venv
```

---

### Step 4: Activate the virtual environment (Windows)
```bash
.\venv\Scripts\activate
```

---

### Step 5: Install dependencies
```bash
pip install streamlit pypdf
pip install sentence-transformers faiss-cpu
pip install langchain-community langchain-text-splitters
pip install google-generativeai
```

---

### Step 6: Create a Gemini API key
1. Visit https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click Create API Key
4. Copy the generated key

---

### Step 7: Set the Gemini API key (per terminal session)
```bash
$env:GEMINI_API_KEY="your_api_key_here"
```

---

### Step 8: Run the application
```bash
streamlit run app.py
```

The app will open in your browser at:
http://localhost:8501

---

## 📌 Future Improvements
- Add source citations for retrieved chunks
- Support scanned/image-based PDFs using OCR
- Multi-PDF support
- Conversation history (chat-style interface)

---

## 📄 License
This project is intended for educational and learning purposes.
