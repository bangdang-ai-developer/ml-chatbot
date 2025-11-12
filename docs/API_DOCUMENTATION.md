# ML Chatbot API Documentation

## Overview

This document describes the REST API for the Machine Learning Chatbot, which provides intelligent answers to questions about Andrew Ng's "Machine Learning Yearning" book using RAG (Retrieval-Augmented Generation).

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. This may change in production deployments.

## Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `400` - Bad Request (validation errors)
- `500` - Internal Server Error

Error responses follow this format:

```json
{
  "detail": "Error description"
}
```

## Endpoints

### Health Check

Check if the API is running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "ML Chatbot API",
  "version": "1.0.0"
}
```

### Chat

Send a message to the chatbot and get a response.

**Endpoint:** `POST /api/v1/chat`

**Request Body:**
```json
{
  "message": "What is machine learning?",
  "session_id": "optional_session_identifier"
}
```

**Response:**
```json
{
  "message": "Machine learning is the science of getting computers to act without being explicitly programmed...",
  "session_id": "generated_or_provided_session_id",
  "sources": [
    {
      "chunk_id": "chunk_123",
      "page_number": 1,
      "chapter": "Chapter 1",
      "section": "Introduction",
      "confidence": 0.85
    }
  ],
  "confidence": 0.85
}
```

### Index Document

Start indexing the ML Yearning PDF document.

**Endpoint:** `POST /api/v1/index-document`

**Response:**
```json
{
  "message": "Document indexing started in background"
}
```

### Get Indexing Status

Check the status of document indexing.

**Endpoint:** `GET /api/v1/indexing-status`

**Response:**
```json
{
  "status": "completed",
  "indexed_chunks": 245,
  "last_updated": "2024-01-15T10:30:00Z"
}
```

### Get Statistics

Get statistics about the chatbot and indexed documents.

**Endpoint:** `GET /api/v1/stats`

**Response:**
```json
{
  "indexed_documents": true,
  "vector_store": "Milvus",
  "embedding_model": "text-embedding-004",
  "similarity_threshold": 0.7
}
```

### Reindex Documents

Clear and reindex all documents.

**Endpoint:** `POST /api/v1/reindex`

**Response:**
```json
{
  "message": "Document reindexing started in background"
}
```

## Data Models

### ChatRequest

```typescript
interface ChatRequest {
  message: string;           // User's question
  session_id?: string;      // Optional session identifier
}
```

### ChatResponse

```typescript
interface ChatResponse {
  message: string;          // AI-generated response
  session_id: string;       // Session identifier
  sources?: Source[];       // Source references
  confidence?: number;      // Overall confidence score (0-1)
}
```

### Source

```typescript
interface Source {
  chunk_id: string;         // Unique chunk identifier
  page_number?: number;     // Page number from PDF
  chapter?: string;         // Chapter name
  section?: string;         // Section name
  confidence: number;       // Match confidence (0-1)
}
```

## Rate Limiting

Currently, there are no rate limits implemented. Consider adding rate limiting for production use.

## Usage Examples

### Using curl

```bash
# Start a chat session
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the difference between training set and development set?",
    "session_id": "user123"
  }'

# Check indexing status
curl -X GET "http://localhost:8000/api/v1/indexing-status"

# Get chatbot statistics
curl -X GET "http://localhost:8000/api/v1/stats"
```

### Using Python

```python
import requests

# Chat with the bot
response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "message": "When should I collect more data?",
        "session_id": "user123"
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"Response: {data['message']}")
    print(f"Confidence: {data.get('confidence', 'N/A')}")
    for source in data.get('sources', []):
        print(f"Source: {source}")
```

### Using JavaScript

```javascript
// Chat with the bot
async function chatWithBot(message) {
  const response = await fetch('/api/v1/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: message,
      session_id: 'user123'
    })
  });

  if (response.ok) {
    const data = await response.json();
    console.log('Response:', data.message);
    console.log('Confidence:', data.confidence);
    console.log('Sources:', data.sources);
  } else {
    console.error('Error:', await response.text());
  }
}

chatWithBot('What is error analysis?');
```

## Error Scenarios

### Common Errors

1. **Invalid Query (400)**
   ```json
   {
     "detail": "Invalid query. Please ask a question related to machine learning."
   }
   ```

2. **Internal Server Error (500)**
   ```json
   {
     "detail": "Embedding generation failed: API quota exceeded"
   }
   ```

3. **Validation Error (422)**
   ```json
   {
     "detail": [
       {
         "loc": ["body", "message"],
         "msg": "field required",
         "type": "value_error.missing"
       }
     ]
   }
   ```

### Troubleshooting

- **404 Error**: Check that the endpoint URL is correct
- **500 Error**: Check server logs for detailed error information
- **Connection Refused**: Ensure the API server is running on port 8000
- **Slow Responses**: Check Milvus connection and API key validity