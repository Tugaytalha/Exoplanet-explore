# 🐳 Docker Deployment Guide

## Overview

This guide explains how to build and run the Exoplanet Explorer API using Docker. The container automatically:
1. ✅ Fetches data from NASA
2. ✅ Trains the XGBoost model
3. ✅ Runs the FastAPI backend

---

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 2: Using Docker CLI

```bash
# Build image
docker build -t exoplanet-api .

# Run container
docker run -d -p 8000:8000 --name exoplanet-api exoplanet-api

# View logs
docker logs -f exoplanet-api

# Stop container
docker stop exoplanet-api
docker rm exoplanet-api
```

---

## 📋 Prerequisites

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 1.29 or higher (if using compose)
- **Disk Space**: ~2GB free
- **RAM**: 4GB minimum, 8GB recommended
- **Time**: First run takes 10-20 minutes (data fetch + model training)

---

## 🛠️ Build Instructions

### Build the Image

```bash
# Build with default settings
docker build -t exoplanet-api .

# Build with specific tag
docker build -t exoplanet-api:v3.0 .

# Build with no cache (fresh build)
docker build --no-cache -t exoplanet-api .
```

**Build time**: ~5-10 minutes (depending on internet speed)

---

## 🏃 Running the Container

### Basic Run

```bash
docker run -d \
  --name exoplanet-api \
  -p 8000:8000 \
  exoplanet-api
```

### Run with Volume Mounts (Persist Data)

```bash
docker run -d \
  --name exoplanet-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model_outputs:/app/model_outputs \
  exoplanet-api
```

**Benefits:**
- Data and models persist across container restarts
- Faster subsequent startups (skips data fetch and training)

### Run Without Training (Fast Start)

```bash
docker run -d \
  --name exoplanet-api \
  -p 8000:8000 \
  -e SKIP_TRAINING=true \
  exoplanet-api
```

**Note**: API will use dataset dispositions instead of ML predictions

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_TRAINING` | `false` | Skip model training (faster startup) |

**Example:**
```bash
docker run -d \
  -e SKIP_TRAINING=true \
  -p 8000:8000 \
  exoplanet-api
```

### Port Configuration

Change the host port (container always uses 8000):

```bash
# Run on port 8080
docker run -d -p 8080:8000 exoplanet-api

# Run on port 3000
docker run -d -p 3000:8000 exoplanet-api
```

---

## 📊 Startup Process

### What Happens During Startup?

1. **Check Data** (0-5 minutes)
   - If data exists: Skip
   - If not: Fetch from NASA

2. **Check Model** (0-15 minutes)
   - If model exists: Skip
   - If not and `SKIP_TRAINING=false`: Train model
   - If not and `SKIP_TRAINING=true`: Skip (use dataset values)

3. **Start API** (~10 seconds)
   - Load model (if available)
   - Predict all KOIs
   - Start FastAPI server

### Startup Logs

**First run (full startup):**
```
================================
Exoplanet Explorer API - Startup
================================

Step 1: Fetching data from NASA...
Fetching KOI data...
Fetching stellar hosts...
✅ Data fetched successfully!

Step 2: Training XGBoost model...
This will take 5-15 minutes...
======================================================================
 KOI DISPOSITION PREDICTION - XGBOOST MODEL TRAINING
======================================================================
...
✅ Model trained successfully!

Step 3: Starting API server...
✅ Loaded model from xgboost_koi_disposition_20251004_192622.joblib
🔮 Predicting dispositions for 59316 KOIs...
✅ Predictions complete!
API will be available at http://localhost:8000
```

**Subsequent runs (with persisted data):**
```
================================
Exoplanet Explorer API - Startup
================================

✅ Data already exists, skipping fetch
✅ Model already exists, skipping training

Step 3: Starting API server...
✅ Loaded model from xgboost_koi_disposition_20251004_192622.joblib
🔮 Predicting dispositions for 59316 KOIs...
✅ Predictions complete!
API will be available at http://localhost:8000
```

---

## 🔍 Monitoring

### View Logs

```bash
# Follow logs in real-time
docker logs -f exoplanet-api

# View last 100 lines
docker logs --tail 100 exoplanet-api

# View logs with timestamps
docker logs -t exoplanet-api
```

### Check Container Status

```bash
# List running containers
docker ps

# Check container health
docker inspect exoplanet-api | grep -A 5 Health
```

### Health Check

The container includes a health check that runs every 30 seconds:

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' exoplanet-api
```

**Possible states:**
- `starting` - Container just started
- `healthy` - API is responding
- `unhealthy` - API is not responding

---

## 🌐 Accessing the API

Once the container is running:

### API Endpoints

```bash
# Root - API info
curl http://localhost:8000/

# Interactive docs (Swagger UI)
Open in browser: http://localhost:8000/docs

# Alternative docs (ReDoc)
Open in browser: http://localhost:8000/redoc

# Get planets
curl http://localhost:8000/planets?limit=10

# Get statistics
curl http://localhost:8000/stats

# Model status
curl http://localhost:8000/model/status
```

---

## 🛑 Stopping and Cleaning Up

### Stop Container

```bash
# Stop (can be restarted)
docker stop exoplanet-api

# Remove container
docker rm exoplanet-api

# Stop and remove
docker stop exoplanet-api && docker rm exoplanet-api
```

### With Docker Compose

```bash
# Stop services
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes
docker-compose down -v
```

### Remove Image

```bash
# Remove image
docker rmi exoplanet-api

# Force remove
docker rmi -f exoplanet-api
```

### Clean Up Everything

```bash
# Remove container, image, and volumes
docker-compose down -v --rmi all
```

---

## 🔄 Updating

### Update Application Code

```bash
# Stop and remove old container
docker-compose down

# Rebuild image
docker-compose build --no-cache

# Start new container
docker-compose up -d
```

### Preserve Data Across Updates

Using volume mounts ensures data and models persist:

```yaml
# In docker-compose.yml
volumes:
  - ./data:/app/data
  - ./model_outputs:/app/model_outputs
```

---

## 🐛 Troubleshooting

### Container Won't Start

**Check logs:**
```bash
docker logs exoplanet-api
```

**Common issues:**
- Port 8000 already in use → Change port: `-p 8080:8000`
- Insufficient memory → Increase Docker memory limit
- Disk space full → Free up space

### API Not Responding

**Check if container is running:**
```bash
docker ps
```

**Check health:**
```bash
docker inspect --format='{{.State.Health.Status}}' exoplanet-api
```

**Restart container:**
```bash
docker restart exoplanet-api
```

### Model Training Fails

**Skip training:**
```bash
docker run -d -e SKIP_TRAINING=true -p 8000:8000 exoplanet-api
```

**Check logs for errors:**
```bash
docker logs exoplanet-api | grep -i error
```

### Data Fetch Fails

**Check internet connection:**
```bash
docker exec exoplanet-api curl -I https://exoplanetarchive.ipac.caltech.edu
```

**Manual data provision:**
```bash
# Copy data to container
docker cp ./data/koi_with_relative_location.csv exoplanet-api:/app/data/
```

---

## 📦 Production Deployment

### Using Docker Compose (Recommended)

```yaml
version: '3.8'

services:
  exoplanet-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SKIP_TRAINING=false
    volumes:
      - ./data:/app/data
      - ./model_outputs:/app/model_outputs
    restart: always
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

### With Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### With SSL (Let's Encrypt)

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

---

## 🔐 Security Best Practices

### 1. Run as Non-Root User

Add to Dockerfile:
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

### 2. Limit Resources

```bash
docker run -d \
  --memory=4g \
  --cpus=2 \
  -p 8000:8000 \
  exoplanet-api
```

### 3. Use Environment Variables for Secrets

```bash
docker run -d \
  -e API_KEY=${API_KEY} \
  -p 8000:8000 \
  exoplanet-api
```

### 4. Network Isolation

```bash
# Create network
docker network create exoplanet-network

# Run in network
docker run -d \
  --network exoplanet-network \
  --name exoplanet-api \
  exoplanet-api
```

---

## 📊 Performance Tuning

### Memory Optimization

```yaml
services:
  exoplanet-api:
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

### CPU Allocation

```yaml
services:
  exoplanet-api:
    deploy:
      resources:
        limits:
          cpus: '4'
        reservations:
          cpus: '2'
```

### Faster Startup

Use volume mounts to persist data:
```yaml
volumes:
  - ./data:/app/data
  - ./model_outputs:/app/model_outputs
```

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **Training Guide**: `TRAINING_GUIDE.md`
- **API Guide**: `API_GUIDE.md`
- **Docker Docs**: https://docs.docker.com/

---

## ✅ Quick Reference

```bash
# Build
docker build -t exoplanet-api .

# Run (basic)
docker run -d -p 8000:8000 --name exoplanet-api exoplanet-api

# Run (with volumes)
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model_outputs:/app/model_outputs \
  --name exoplanet-api exoplanet-api

# Run (skip training)
docker run -d -p 8000:8000 -e SKIP_TRAINING=true exoplanet-api

# Logs
docker logs -f exoplanet-api

# Stop
docker stop exoplanet-api

# Remove
docker rm exoplanet-api

# Docker Compose
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

**Your containerized exoplanet API is ready! 🚀🐳**

