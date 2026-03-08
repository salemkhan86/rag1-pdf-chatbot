# 🧠 RAG PDF Chatbot with Streamlit

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload multiple PDF documents and ask questions about their content.

The system retrieves relevant document chunks and generates answers using a Large Language Model.

---

## 🚀 Features

* 📚 Upload multiple PDFs
* 🔍 Semantic document search using embeddings
* 🧩 Chunking with RecursiveCharacterTextSplitter
* 🧠 Retrieval-Augmented Generation (RAG)
* 💬 Conversational chat history
* ⚡ Powered by Groq LLM
* 🗂 Vector database using Chroma
* 🌐 Streamlit web interface

---

## 🏗 Project Architecture

PDF Upload
→ Document Loader
→ Text Chunking
→ Embeddings (Sentence Transformers)
→ Vector Database (Chroma)
→ Retriever
→ Groq LLM
→ Answer Generation

---

## 🛠 Technologies Used

* Python
* Streamlit
* LangChain
* Groq API
* Sentence Transformers
* Chroma Vector Database
* PyPDF

---

## 📦 Installation

Clone the repository:

```
git clone https://github.com/yourusername/rag-pdf-chatbot.git
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app.py
```

---

## 🔑 Environment Variables

Create a `.env` file and add:

```
GROQ_API_KEY=your_api_key_here
```

---

## 📄 Example Use Case

1. Upload one or more PDF documents
2. Ask questions about the content
3. The system retrieves relevant information and generates answers

---

## 🌍 Deployment

This project can be deployed using:

* Streamlit Cloud
* Docker
* VPS / Cloud servers

---

## 👨‍💻 Author

Muhammad Salem
