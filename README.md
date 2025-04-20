# 🧠 PDF Emotion Analyzer & Chatbot

An AI-powered web app that combines two powerful features into one:

1. **Sentiment Analysis of PDF Content**
2. **Interactive Chatbot to Converse with PDF**

---

## 🔍 Features

### 📑 Sentiment Analysis

- Upload any PDF and get its overall sentiment (Happy, Sad, Fear, Anger, Love)
- Built using a fine-tuned **ModernBERT model**
- Extended training dataset using **Groq LLM** to generate long and diverse samples
- Custom PyTorch layers added for better performance and control

### 💬 Chat with your PDF

- Ask questions about any uploaded document
- Backed by **LangChain** with:
  - Chat history
  - Document retrievers
  - Prompt engineering for accurate and contextual answers

---

## ⚙️ Tech Stack

- **NLP Model**: ModernBERT + PyTorch
- **Embedding**: ModernBERT + LangChain
- **LLM Tools**: Groq (for dataset generation)
- **Chat Backend**: LangChain + FastAPI
- **Frontend**: Streamlit (with dynamic UI flow)
- **Deployment Ready**

---

## 🧪 How to Use

1. Upload your PDF
2. Choose:
   - 📈 Get Sentiment
   - 💬 Ask Questions
3. Enjoy the insights or chat with your PDF!

---
