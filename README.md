# AI Examples

A collection of AI and machine learning examples demonstrating practical implementations of cutting-edge technologies including Retrieval-Augmented Generation (RAG), quantum computing, and GPU-optimized training pipelines.

## 🌟 Overview

This repository showcases production-ready AI applications and experiments, with a focus on:
- **Quantum Computing & AI**: RAG systems for quantum computing documentation
- **Deep Learning Infrastructure**: GPU-optimized training pipelines with modular architecture
- **IBM Cloud AI Integration**: Examples using IBM Watsonx AI and Granite models
- **Modern ML Practices**: Best practices for scalable, configurable AI systems

## 📁 Repository Structure

### 🔬 [examples/](./examples/)
Contains practical AI examples and demonstrations, with a primary focus on quantum computing applications in finance.

**Main Highlight: RAG_quantum.ipynb**
- Complete RAG (Retrieval-Augmented Generation) pipeline for quantum computing documentation
- Uses IBM Watsonx AI with Granite models for question-answering
- Implements vector similarity search with Milvus database
- Processes research papers from arXiv using Docling
- Technologies: IBM Watsonx, Granite Embeddings, LangChain, Hugging Face Transformers

**Additional Files**
- `extra/`: Experimental scripts for quantum computing financial analysis using OpenAI's API
- Various implementations of web scraping and content analysis tools

[📖 View Examples Documentation](./examples/README.md)

### ⚡ [gpu-training-pipeline/](./gpu-training-pipeline/)
A modular, production-ready deep learning training pipeline optimized for GPU training.

**Key Features:**
- Modular architecture with Hydra configuration management
- Support for text classification with Hugging Face Transformers
- Docker containerization with NVIDIA CUDA runtime
- Gradient accumulation and learning rate scheduling
- Optional Weights & Biases integration for experiment tracking
- Configurable models (DistilBERT, BERT, RoBERTa, etc.)

**Use Cases:**
- Text classification tasks
- Sentiment analysis
- Custom NLP model training
- GPU-accelerated training experiments

[📖 View GPU Pipeline Documentation](./gpu-training-pipeline/README.md)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- (Optional) CUDA-capable GPU for training pipeline
- (Optional) IBM Cloud account with Watsonx AI access for RAG examples

### Running the RAG Example
```bash
cd examples
# Set up environment variables (never commit actual credentials!)
export WATSONX_APIKEY="your_api_key"
export WATSONX_PROJECT_ID="your_project_id"
export WATSONX_URL="your_watsonx_url"

# Launch Jupyter and open RAG_quantum.ipynb
jupyter notebook RAG_quantum.ipynb
```

### Running the GPU Training Pipeline
```bash
cd gpu-training-pipeline

# Install dependencies
pip install -r requirements.txt

# Run training with default configuration
python train.py

# Or use Docker
docker build -t gpu-training-pipeline .
docker run --gpus all gpu-training-pipeline
```

## 🔧 Key Technologies

- **IBM Watsonx AI**: Enterprise-grade AI platform for LLMs
- **IBM Granite Models**: Efficient embedding and language models
- **PyTorch**: Deep learning framework for GPU training
- **Hugging Face Transformers**: State-of-the-art NLP models
- **LangChain**: Framework for building LLM applications
- **Milvus**: High-performance vector database
- **Docling**: Advanced document parsing and conversion
- **Hydra**: Configuration management for ML experiments
- **Docker**: Containerization for reproducible environments

## 📚 Use Cases

### Quantum Computing RAG System
- Build intelligent Q&A systems for technical documentation
- Process and understand quantum computing research papers
- Demonstrate AI applications in emerging technologies
- Showcase integration between IBM Cloud AI services

### GPU Training Pipeline
- Train custom text classification models
- Experiment with different transformer architectures
- Scale training with GPU acceleration
- Manage ML experiments with structured configurations

## 🤝 Contributing

This repository serves as a demonstration of AI capabilities and best practices. Feel free to explore, learn, and adapt these examples for your own projects.

## 📄 License

Please refer to individual project directories for specific licensing information.

## 🔗 Additional Resources

- [IBM Watsonx AI Documentation](https://www.ibm.com/products/watsonx-ai)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [LangChain Documentation](https://python.langchain.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Note**: Some examples require API keys and credentials. Ensure you have the necessary access before running the notebooks and scripts.
