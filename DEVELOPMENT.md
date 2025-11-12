# Development Guide

## Overview

This guide covers how to set up a development environment for the ML Chatbot project.

## Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- Google Gemini API key
- Andrew Ng's Machine Learning Yearning PDF

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ml-chatbot
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp ../.env.example .env
# Edit .env with your API key
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Copy environment variables
cp ../.env.example .env.local
```

### 4. Development Services

Start the development services:

```bash
# Start Milvus and other services
docker-compose up -d etcd minio milvus

# Or use the development profile
docker-compose --profile dev up
```

## Running the Application

### Development Mode

#### Backend

```bash
cd backend

# Run with auto-reload
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Or run the Python file directly
python -m src.main
```

#### Frontend

```bash
cd frontend

# Start development server
npm run dev
```

### Using Docker for Development

```bash
# Start all services in development mode
docker-compose --profile dev up

# Start specific services
docker-compose --profile dev up backend-dev frontend-dev
```

## Project Structure

```
ml-chatbot/
├── backend/                 # Python FastAPI backend
│   ├── src/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core configuration and utilities
│   │   ├── models/         # Pydantic models
│   │   ├── repositories/   # Data access layer
│   │   └── services/       # Business logic
│   ├── tests/              # Backend tests
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── lib/            # Utility functions
│   │   └── ui/             # UI components (shadcn/ui)
│   └── package.json        # Node.js dependencies
├── docs/                   # Documentation
├── docker-compose.yml      # Production Docker setup
├── docker-compose.dev.yml  # Development Docker setup
└── README.md
```

## Development Workflow

### 1. Making Changes

#### Backend Changes

1. Make changes to source code in `backend/src/`
2. Run tests to verify changes:
   ```bash
   cd backend
   pytest
   ```
3. Test API changes manually or with automated tests
4. Commit changes with descriptive messages

#### Frontend Changes

1. Make changes to source code in `frontend/src/`
2. The development server will auto-reload
3. Run tests if available:
   ```bash
   cd frontend
   npm test
   ```
4. Test UI changes in browser
5. Commit changes

### 2. Code Quality

#### Python (Backend)

- Use type hints
- Follow PEP 8 style guide
- Write comprehensive tests
- Use async/await for I/O operations

#### TypeScript (Frontend)

- Use strict TypeScript settings
- Follow React best practices
- Use functional components with hooks
- Implement proper error handling

### 3. Testing

#### Backend Tests

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_document_service.py

# Run with verbose output
pytest -v
```

#### Frontend Tests

```bash
cd frontend

# Run tests
npm test

# Run with coverage
npm run test:coverage
```

### 4. Debugging

#### Backend Debugging

1. **Using Python Debugger**
   ```python
   import pdb; pdb.set_trace()
   ```

2. **Logging**
   ```python
   import logging
   logger = logging.getLogger(__name__)
   logger.info("Debug message")
   ```

3. **VS Code Debugging**
   - Use Python extension
   - Configure launch.json for debugging

#### Frontend Debugging

1. **Browser Developer Tools**
   - Use React Developer Tools
   - Check network requests
   - Console logging

2. **VS Code Debugging**
   - Use JavaScript Debugger
   - Configure launch.json

## API Development

### Adding New Endpoints

1. Define Pydantic models in `backend/src/models/`
2. Implement business logic in `backend/src/services/`
3. Add API endpoints in `backend/src/api/`
4. Write tests for new endpoints

Example:

```python
# models/example.py
from pydantic import BaseModel

class ExampleRequest(BaseModel):
    message: str

class ExampleResponse(BaseModel):
    response: str

# api/example.py
from fastapi import APIRouter
from ..services.example_service import ExampleService
from ..models.example import ExampleRequest, ExampleResponse

router = APIRouter()

@router.post("/example", response_model=ExampleResponse)
async def example_endpoint(request: ExampleRequest):
    service = ExampleService()
    result = await service.process_example(request.message)
    return ExampleResponse(response=result)
```

## Frontend Development

### Adding New Components

1. Create component in `frontend/src/components/`
2. Use shadcn/ui components when possible
3. Follow TypeScript best practices
4. Add proper error handling

Example:

```typescript
// components/ExampleComponent.tsx
import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'

interface ExampleComponentProps {
  title: string
  content: string
}

export function ExampleComponent({ title, content }: ExampleComponentProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <p>{content}</p>
      </CardContent>
    </Card>
  )
}
```

### State Management

- Use React hooks for local state
- Consider Context API for global state
- Implement proper loading and error states

## Database Development

### Milvus Management

```bash
# Connect to Milvus container
docker exec -it milvus-standalone bash

# Check collection status
python -c "
from pymilvus import connections, utility
connections.connect('default', host='localhost', port='19530')
print(utility.list_collections())
"
```

### Vector Operations

- Test embedding generation
- Verify vector similarity search
- Monitor database performance

## Environment Configuration

### Development Environment Variables

```bash
# .env
GOOGLE_API_KEY=your_development_api_key
MILVUS_HOST=localhost
MILVUS_PORT=19530
PDF_PATH=./andrew-ng-machine-learning-yearning.pdf
CHUNK_SIZE=500
CHUNK_OVERLAP=100
SIMILARITY_THRESHOLD=0.6
```

### Frontend Environment

```bash
# frontend/.env.local
VITE_API_URL=http://localhost:8000
```

## Common Development Tasks

### Indexing the PDF

```bash
# Start services
docker-compose --profile dev up

# Index document (in another terminal)
curl -X POST http://localhost:8000/api/v1/index-document

# Check status
curl -X GET http://localhost:8000/api/v1/indexing-status
```

### Resetting the Database

```bash
# Stop services
docker-compose down

# Remove volumes
docker volume rm ml-chatbot_milvus_data
docker volume rm ml-chatbot_etcd_data
docker volume rm ml-chatbot_minio_data

# Restart services
docker-compose up -d
```

### Viewing Logs

```bash
# View all logs
docker-compose logs

# Follow specific service logs
docker-compose logs -f backend-dev
docker-compose logs -f frontend-dev

# View logs from last hour
docker-compose logs --since=1h backend
```

## Performance Optimization

### Backend Optimization

1. **Async Operations**
   - Use async/await properly
   - Implement connection pooling
   - Cache frequently accessed data

2. **Vector Search**
   - Optimize similarity threshold
   - Adjust chunk sizes
   - Use appropriate indexing

### Frontend Optimization

1. **React Performance**
   - Use React.memo for expensive components
   - Implement code splitting
   - Optimize re-renders

2. **Network**
   - Implement request caching
   - Use proper loading states
   - Optimize bundle size

## Contributing

### Code Style

- Use consistent formatting (Black for Python, Prettier for TypeScript)
- Write descriptive commit messages
- Include tests with new features
- Update documentation

### Pull Request Process

1. Create feature branch
2. Make changes and test
3. Update tests and documentation
4. Submit pull request with description
5. Address review feedback

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check what's using ports
   netstat -tulpn | grep :8000
   lsof -i :3000
   ```

2. **Memory Issues**
   ```bash
   # Check Docker memory usage
   docker stats
   ```

3. **API Key Issues**
   ```bash
   # Verify API key is set
   echo $GOOGLE_API_KEY
   ```

### Getting Help

- Check application logs
- Review API documentation
- Search existing issues
- Ask in development discussions