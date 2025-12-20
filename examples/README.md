# Examples

This directory contains examples and demonstrations of AI applications in finance.

## Main Example: RAG_quantum.ipynb

### Overview

The `RAG_quantum.ipynb` notebook demonstrates a **Retrieval-Augmented Generation (RAG)** system for IBM Quantum Computing documentation using IBM Cloud services. This example showcases how to build an intelligent question-answering system that can retrieve and synthesize information from quantum computing research papers.

### What It Does

The notebook implements a complete RAG pipeline that:

1. **Document Processing**: Uses Docling to convert and parse PDF research papers from arXiv about quantum computing
2. **Document Chunking**: Employs HybridChunker to split documents into semantically meaningful chunks for efficient retrieval
3. **Embedding Generation**: Utilizes IBM Granite's lightweight embedding model (`granite-embedding-30m-english`) to convert text chunks into vector representations
4. **Vector Storage**: Stores embeddings in a Milvus vector database for fast similarity search
5. **Question Answering**: Integrates with IBM Watsonx AI (using `granite-4-h-small` model) to generate accurate answers based on retrieved context

### Key Technologies Used

- **IBM Watsonx AI**: Large language model for generating responses
- **IBM Granite Embeddings**: Lightweight, efficient embedding model for semantic search
- **Milvus**: Vector database for storing and retrieving document embeddings
- **Docling**: Advanced document parsing and conversion library
- **LangChain**: Framework for building LLM applications with RAG capabilities
- **Hugging Face Transformers**: For tokenization and model management

### Use Case

This notebook is particularly valuable for:
- Building question-answering systems for technical documentation
- Demonstrating quantum computing concepts in finance and other domains
- Showcasing integration between IBM Cloud AI services
- Learning how to implement production-ready RAG systems

### Source Documents

The example processes quantum computing research papers from arXiv, including:
- Papers on quantum algorithms and applications
- Quantum computing in various domains
- Recent advances in quantum technology

### Requirements

To run this notebook, you need:
- IBM Cloud account with Watsonx AI access
- IBM Watsonx API key and project ID
- Environment variables: `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`
- Python packages: `langchain-ibm`, `langchain-huggingface`, `langchain-milvus`, `docling`, `transformers`

### Output

The notebook creates an interactive RAG system where users can:
- Ask questions about quantum computing concepts
- Get answers grounded in the source research papers
- Retrieve relevant context from the knowledge base

---

## Additional Files

Other Python scripts in the `extra` folder contain various experiments and utilities related to quantum computing financial analysis and web scraping. See the `extra` folder for more details.
