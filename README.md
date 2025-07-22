# RAG Bot - Viral Content Analysis Assistant

A custom Retrieval-Augmented Generation (RAG) system that analyzes documents and viral video content (e.g., TikTok/Instagram) to answer questions and extract high-engagement insights using open-source embeddings and LangChain.

## 🚀 What It Does

This RAG bot:
- Accepts your PDF/TXT files or scraped content (e.g., captions, hashtags, comments)
- Embeds them using **open-source sentence transformers**
- Allows you to ask questions like:
  - "What makes this video go viral?"
  - "What emotions or hooks were used?"
  - "Which clips had ridiculous endings?"
- Returns GPT-generated responses **based on actual content**, not hallucinations.

---

## ⚙️ Quick Start

### 1. **Clone and Setup**
```bash
git clone <your-repo-url>
cd rag-bot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 📂 Add Your Data

#### For PDFs and text:
Place them in the `documents/` folder

---

### 4. ⚙️ Setup the Knowledge Base (First Time Only)

```bash
python setup.py
```
This script:

- Loads documents
- Splits them into chunks
- Generates vector embeddings using **HuggingFace**
- Stores them in **ChromaDB**

---

### 5. Run the Bot

####  Command-Line
```bash
python rag_bot.py
```

#### WEB APP
```bash
streamlit run app.py
```