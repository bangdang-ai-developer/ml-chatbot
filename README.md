# ML Chatbot - Andrew Ng Machine Learning Yearning

A RAG-powered chatbot that answers questions from Andrew Ng's "Machine Learning Yearning" PDF.

## Architecture

- **Backend**: Python FastAPI with LangChain RAG
- **Frontend**: React with AI SDK UI chatbot components
- **Vector DB**: Self-hosted Milvus
- **AI Model**: Gemini 2.5 Flash with block_none safety
- **Deployment**: Docker Compose

## Project Structure

```
ml-chatbot/
├── backend/                 # Python FastAPI backend
│   ├── src/
│   │   ├── core/           # Core application logic
│   │   ├── services/       # Business services
│   │   ├── repositories/   # Data access layer
│   │   ├── api/           # API endpoints
│   │   └── models/        # Data models
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   └── utils/
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Features

- RAG-based question answering from PDF content
- Intelligent document chunking and embedding
- Fast response times with vector similarity search
- Clean chatbot UI with typing indicators
- SOLID principles implementation
- Comprehensive error handling
- Docker-based deployment

## Getting Started

1. Clone the repository
2. Copy `.env.example` to `.env` and configure
3. Run `docker-compose up`
4. Access the application at `http://localhost:3000`

## Environment Variables

- `GOOGLE_API_KEY`: Gemini API key
- `MILVUS_HOST`: Milvus server host
- `MILVUS_PORT`: Milvus server port
- `PDF_PATH`: Path to the Andrew Ng PDF file