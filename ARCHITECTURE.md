# ML Chatbot System Architecture

## Overview

This document describes the current architecture of the Machine Learning chatbot system with RAG (Retrieval-Augmented Generation) capabilities, Vietnamese-English translation, and intelligent response routing.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ML Chatbot System Architecture                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   Open WebUI │◄──►│   FastAPI       │◄──►│   Milvus        │◄──►│    Etcd      │ │
│  │   Frontend   │    │   Backend       │    │   Vector DB     │    │   Coord.     │ │
│  │   (Port 3000)│    │   (Port 8000)   │    │   (Port 19530)  │    │   (Port 2379) │ │
│  └─────────────┘    └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                   │                     │                     │          │
│           │                   │                     │                     │          │
│           ▼                   ▼                     ▼                     ▼          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      Backend Services Layer                           │ │
│  ├─────────────────────────────────────────────────────────────────────────┤ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │ │
│  │  │   API Layer      │  │  Service Layer   │  │   Data Layer     │           │ │
│  │  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤           │ │
│  │  │ • Chat API       │  │ • RAG Service    │  │ • Document Svc   │           │ │
│  │  │ • WebUI API      │  │ • AI Service     │  │ • Embedding Svc │           │ │
│  │  │ • Document API  │  │ • Translation    │  │ • Vector Repo    │           │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘           │ │
│  │           │                 │                 │                         │ │
│  │           ▼                 ▼                 ▼                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                Core Intelligence Services                           │ │ │
│  │  ├─────────────────────────────────────────────────────────────────────┤ │ │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │ │ │
│  │  │  │ Smart Response  │  │ Query Intent     │  │ Vietnamese      │  │ │ │
│  │  │  │ Router          │  │ Classifier       │  │ Text Service    │  │ │ │
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │ │ │
│  │  └─────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         External Integrations                              │ │
│  ├─────────────────────────────────────────────────────────────────────────┤ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │ │
│  │  │   Gemini AI      │  │   MinIO         │  │   PDF Files     │           │ │
│  │  │   (Text/Embed)   │  │   Object Store  │  │   Ian Goodfellow│           │ │
│  │  │   Google Cloud   │  │   (Port 9000)   │  │   Deep Learning  │           │ │
│  │  │   APIs           │  │                 │  │   Book          │           │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## System Components

### Frontend Layer
- **Open WebUI**: Web-based chat interface (Port 3000)
  - Real-time chat interface
  - Message history and session management
  - User-friendly UI for ML/AI interactions

### Backend Layer

#### API Layer
- **FastAPI Application** (Port 8000)
  - RESTful API endpoints
  - OpenAI-compatible interface
  - Request/response validation
  - CORS and security handling

#### Core Services

**RAG Service**
- Document retrieval from vector database
- Context management for AI responses
- Integration with translation pipeline
- Similarity search and ranking

**AI Service**
- Gemini 2.5 Flash integration
- Response generation with retrieved context
- Prompt engineering and optimization
- Vietnamese-English response handling

**Translation Service**
- Vietnamese to English query translation
- Technical term preservation
- Language detection and routing
- Translation quality validation

**Smart Response Router**
- Query intent classification
- Response strategy selection
- Hybrid response generation
- Context quality assessment

**Query Intent Classifier**
- Vietnamese query analysis
- Intent recognition (definition, comparison, examples, etc.)
- Confidence scoring
- Routing decision logic

**Vietnamese Text Service**
- Vietnamese text processing
- Language detection
- Text normalization
- Special character handling

#### Data Services

**Document Service**
- PDF text extraction (PyPDF2, pdfplumber)
- Content quality assessment
- Document chunking (3000 chars, 500 overlap)
- Metadata extraction

**Embedding Service**
- Text embeddings generation (text-embedding-004)
- Vector encoding for similarity search
- Batch processing support
- Caching optimization

**Vector Repository**
- Milvus vector database integration
- Similarity search operations
- Document chunk storage
- Index management

### Infrastructure Layer

**Milvus Vector Database** (Port 19530)
- Vector similarity search
- Document chunk storage
- Efficient retrieval operations
- Scalable vector operations

**MinIO Object Storage** (Port 9000)
- PDF file storage
- Data persistence
- Backup and recovery
- Scalable object storage

**Etcd Coordination** (Port 2379)
- Milvus cluster coordination
- Configuration management
- Service discovery
- Distributed locking

### External Services

**Google Cloud APIs**
- Gemini 2.5 Flash AI model
- Text embedding generation
- Multimodal capabilities
- High-performance inference

## Data Flow

### Document Indexing Flow
1. PDF Upload → MinIO Storage
2. Text Extraction → Document Service
3. Content Chunking → 3000 char segments
4. Embedding Generation → Text-Embedding-004
5. Vector Storage → Milvus Database

### Query Processing Flow
1. User Query → Open WebUI Frontend
2. Language Detection → Translation Service
3. Vietnamese → English Translation
4. Vector Embedding → Embedding Service
5. Similarity Search → Milvus Database
6. Context Retrieval → RAG Service
7. Intent Classification → Query Intent Classifier
8. Response Generation → AI Service (Gemini)
9. Vietnamese Response → Translation Pipeline

## Key Features

### Multilingual Support
- Vietnamese query processing
- English document retrieval
- Native Vietnamese responses
- Technical term preservation

### Intelligent Routing
- Query intent analysis
- Hybrid response strategies
- Context quality assessment
- Adaptive response generation

### Advanced RAG
- Large language model integration
- Vector similarity search
- Context-aware responses
- Comprehensive document coverage

### Scalable Architecture
- Microservices design
- Containerized deployment
- Distributed vector database
- Object storage integration

## Technology Stack

### Backend
- **Framework**: FastAPI
- **AI Model**: Gemini 2.5 Flash
- **Vector DB**: Milvus v2.3.3
- **Storage**: MinIO
- **Coordination**: Etcd

### Frontend
- **Interface**: Open WebUI
- **Protocol**: HTTP/REST
- **Real-time**: WebSocket support

### Deployment
- **Containerization**: Docker & Docker Compose
- **Networking**: Bridge network
- **Storage**: Persistent volumes
- **Scalability**: Service orchestration