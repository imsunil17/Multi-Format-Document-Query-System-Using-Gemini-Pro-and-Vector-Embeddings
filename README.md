# Multi-Format-Document-Query-System-Using-Gemini-Pro-and-Vector-Embeddings
This is a mini-project where we can chat with our uploaded pdf
Abstract The exponential growth of digital documents across varied formats poses significant challenges in efficient information retrieval. This study introduces an intelligent, multi-format document query system that leverages Google Gemini Pro, FAISS vector stores, and Google Generative AI embeddings to enable natural language interaction with documents. The system supports diverse file types such as PDFs, DOCX, TXT, CSV, and image-based formats (JPEG, PNG), incorporating OCR for text extraction from images. The methodology combines document chunking, semantic embedding, and similarity-based retrieval to provide context-aware, conversational answers to user queries. Evaluated on a range of academic and real-world datasets, the system achieves high relevancy and user satisfaction scores while significantly reducing search time. This work underscores the potential of large language models and embeddings in transforming document-based knowledge access, setting the stage for further innovations in document-centric NLP applications.

Keywords Natural Language Processing, Vector Embeddings, Gemini Pro, FAISS, OCR, Conversational AI

# 1. Introduction
With the surge in digital documentation, organizations often grapple with extracting meaningful information from diverse data sources. Traditional search systems fail to deliver context-aware answers, especially when handling varied file types. The need for an interactive system that understands user queries and fetches information across documents is critical. This paper proposes a conversational document query system powered by Gemini Pro and Google embeddings, capable of understanding and answering natural language questions across multiple file formats. Our contributions include (1) a unified processing pipeline for text and image-based documents, (2) integration of OCR to extract text from visual data, and (3) conversational query handling using vector-based similarity and LLMs. The rest of the paper is structured as follows: Section 2 reviews related work, Section 3 details the methodology, Section 4 describes the dataset, Section 5 presents experimental results, Section 6 discusses findings, and Section 7 concludes with future directions.

# 2. Related Work
Recent advancements in document intelligence have seen the rise of systems such as Haystack, LangChain, and OpenAI's ChatGPT integrated search tools. Traditional QA systems based on TF-IDF and rule-based heuristics were limited by scope and semantics. With the advent of transformers, semantic embeddings from BERT, Sentence-BERT, and more recently Google’s embedding models have significantly improved contextual understanding. Our approach differs in its hybrid multi-file format support, combining textual and visual data extraction with state-of-the-art conversational models. The integration of Gemini Pro with FAISS for real-time, chunk-based semantic search provides a scalable and intelligent solution, surpassing many existing models in flexibility and ease of use.

# 3. Methodology
Problem Definition: Build a system that enables querying across multiple document types using natural language, returning accurate, context-rich answers. Architecture: • Input: PDF, DOCX, TXT, CSV, PNG, JPG • Preprocessing: OCR (Tesseract) for images, text extraction for others • Chunking: Recursive Character Splitter • Embedding: GoogleGenerativeAIEmbeddings (model="embedding-001") • Vector DB: FAISS • Language Model: ChatGoogleGenerativeAI (Gemini Pro) • Retrieval: Similarity search with context feeding into LLM Preprocessing: • Image Text Extraction via Tesseract • Text Normalization (lowercasing, whitespace removal) • Chunk size: 10000 characters with 1000 overlap Training Setup: • Gemini Pro model hosted via LangChain • Default API temperature: 0.3 for factual accuracy • FAISS index persisted locally for efficient retrieval

# 4. Dataset
We tested the system on a variety of document sets, including: • Academic research papers (PDF) • Government policies (DOCX, TXT) • Business reports (CSV) • Scanned invoices and forms (JPG/PNG) Each document underwent format-specific parsing. Training and evaluation were simulated by user-driven QA sessions. The dataset was split logically into 70% for indexing and 30% for evaluation based on retrieval relevance.

# 5. Experiments and Results
Metrics Used: Precision@k, Recall@k, F1-score, User Rating (1–5 scale) Setup: Windows 11, Python 3.11, Streamlit frontend, LangChain backend, FAISS, Gemini API

<img width="544" height="270" alt="chat1" src="https://github.com/user-attachments/assets/bce065ed-b277-4125-9857-dd4beaf623c5" />

Error Analysis: Most hallucinations occurred in OCR-derived data due to image noise. These is mitigated by pre-filtering or confidence-based filtering.

# 6. Discussion
The system demonstrates high adaptability across file types and delivers accurate results in real time. OCR integration expands the utility to scanned documents, though challenges remain in noisy or handwritten texts. The use of Gemini Pro ensures semantic integrity and context preservation, especially in multi-turn dialogue. The local FAISS store offers latency-free performance even on modest hardware.

# 7. Conclusion and Future Work
This work presents a versatile, conversational interface to interact with multiple document types. By combining OCR, embeddings, and advanced LLMs, the system provides a robust solution for knowledge extraction. Future enhancements include: support for audio/video transcripts, handwriting recognition, and multi-user chat histories with persistent memory.

# 8. References
Vaswani et al., “Attention is All You Need,” 2017.
Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers,” 2018.
LangChain Documentation: https://docs.langchain.com/
Tesseract OCR: https://github.com/tesseract-ocr/tesseract
Gemini Pro API by Google: https://ai.google.dev/
FAISS Library by Facebook: https://github.com/facebookresearch/faiss
# Workflow 
<img width="895" height="560" alt="chat3" src="https://github.com/user-attachments/assets/f0136400-2f79-4bb2-b20a-e1bc66a1d5c2" />
# Sample Output

<img width="909" height="590" alt="chat4" src="https://github.com/user-attachments/assets/bb75920c-865f-45b4-bb95-9df5da1d9f94" />

