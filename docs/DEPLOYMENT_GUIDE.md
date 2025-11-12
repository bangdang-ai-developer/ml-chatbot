# Deployment Guide

## Overview

This guide covers how to deploy the ML Chatbot application using Docker Compose.

## Prerequisites

- Docker and Docker Compose installed
- Google Gemini API key
- Andrew Ng's Machine Learning Yearning PDF file
- At least 8GB RAM and 20GB storage

## Quick Start

### 1. Environment Setup

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` and add your Google Gemini API key:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 2. Start the Application

For development:
```bash
docker-compose --profile dev up
```

For production:
```bash
docker-compose up -d
```

### 3. Index the Document

Once services are running, index the PDF:

```bash
# Index the document
curl -X POST http://localhost:8000/api/v1/index-document

# Check indexing status
curl -X GET http://localhost:8000/api/v1/indexing-status
```

### 4. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture

### Services

1. **Milvus** - Vector database (ports 19530, 9091)
2. **ETCD** - Configuration management
3. **MinIO** - Object storage (ports 9000, 9001)
4. **Backend** - FastAPI application (port 8000)
5. **Frontend** - React application (port 3000)
6. **Nginx** - Reverse proxy (port 80) - Production only

### Data Flow

1. PDF is processed and chunked
2. Text chunks are embedded using Gemini
3. Embeddings stored in Milvus
4. User queries are embedded
5. Similar documents retrieved from Milvus
6. Gemini generates response using retrieved context

## Environment Variables

### Required

- `GOOGLE_API_KEY` - Your Google Gemini API key

### Optional

- `MILVUS_HOST` - Milvus server host (default: localhost)
- `MILVUS_PORT` - Milvus server port (default: 19530)
- `PDF_PATH` - Path to PDF file (default: /app/data/andrew-ng-machine-learning-yearning.pdf)
- `CHUNK_SIZE` - Text chunk size (default: 1000)
- `CHUNK_OVERLAP` - Chunk overlap (default: 200)
- `SIMILARITY_THRESHOLD` - Minimum similarity score (default: 0.7)
- `MAX_RETRIEVED_DOCS` - Maximum documents to retrieve (default: 5)

## Configuration

### Development vs Production

#### Development (`docker-compose.dev.yml`)
- Hot reloading enabled
- Debug mode
- Lower security settings
- Direct host communication

#### Production (`docker-compose.yml`)
- Optimized builds
- Security headers
- Nginx reverse proxy
- Resource limits
- Health checks

### Scaling

For horizontal scaling, modify `docker-compose.yml`:

```yaml
backend:
  deploy:
    replicas: 3
  # ... other config

frontend:
  deploy:
    replicas: 2
  # ... other config
```

## Monitoring

### Health Checks

All services include health checks:

```bash
# Check all services
docker-compose ps

# Check specific service logs
docker-compose logs backend
```

### Monitoring Tools

Consider adding:
- Prometheus for metrics
- Grafana for dashboards
- ELK stack for logging

## Security

### Production Security Measures

1. **Network Security**
   - Use HTTPS with valid certificates
   - Configure firewall rules
   - Limit exposed ports

2. **API Security**
   - Add API key authentication
   - Implement rate limiting
   - Use CORS properly

3. **Data Security**
   - Encrypt data at rest
   - Use secrets management
   - Regular security updates

### SSL/TLS Configuration

1. Place SSL certificates in `nginx/ssl/`:
   ```
   nginx/ssl/cert.pem
   nginx/ssl/key.pem
   ```

2. Configure Nginx for HTTPS

## Backup and Recovery

### Data Backup

1. **Milvus Data**
   ```bash
   docker exec milvus-standalone cp -r /var/lib/milvus /backup/milvus-$(date +%Y%m%d)
   ```

2. **Application Data**
   ```bash
   docker-compose down
   docker volume ls
   # Backup volumes as needed
   ```

### Recovery

1. Restore volumes from backup
2. Restart services
3. Verify data integrity

## Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   # Check service resource usage
   docker stats
   ```

2. **Milvus Connection Issues**
   ```bash
   # Check Milvus logs
   docker-compose logs milvus

   # Verify connectivity
   docker exec backend curl http://milvus:19530/health
   ```

3. **API Key Issues**
   ```bash
   # Verify API key in environment
   docker-compose exec backend printenv | grep GOOGLE_API_KEY
   ```

### Performance Optimization

1. **Vector Database**
   - Adjust index parameters
   - Optimize embedding dimensions
   - Use appropriate search parameters

2. **Application**
   - Implement caching
   - Optimize chunk sizes
   - Use connection pooling

## Maintenance

### Regular Tasks

1. **Updates**
   ```bash
   # Update Docker images
   docker-compose pull
   docker-compose up -d
   ```

2. **Cleanup**
   ```bash
   # Remove unused images and containers
   docker system prune -a
   ```

3. **Monitoring**
   - Check disk usage
   - Monitor memory usage
   - Review logs for errors

### Logging

Logs are configured with different levels:

```bash
# View all logs
docker-compose logs

# Follow specific service logs
docker-compose logs -f backend

# View logs from last hour
docker-compose logs --since=1h
```

## Production Deployment

### Pre-deployment Checklist

- [ ] Environment variables configured
- [ ] SSL certificates in place
- [ ] Resource limits set
- [ ] Backup strategy planned
- [ ] Monitoring configured
- [ ] Security measures implemented

### Deployment Steps

1. **Prepare Server**
   ```bash
   # Install Docker and Docker Compose
   # Configure firewall
   # Set up SSL certificates
   ```

2. **Deploy Application**
   ```bash
   # Clone repository
   git clone <repo-url>
   cd ml-chatbot

   # Configure environment
   cp .env.example .env
   # Edit .env with production values

   # Deploy
   docker-compose -f docker-compose.yml --profile production up -d
   ```

3. **Post-deployment**
   ```bash
   # Verify services
   docker-compose ps

   # Index document
   curl -X POST http://localhost/api/v1/index-document

   # Test application
   # Configure monitoring
   # Set up backups
   ```