# 🔧 Docker Quick Fix Guide

## Issue: Missing `koi_disposition` Column

If you see this error:
```
KeyError: 'koi_disposition'
```

This means your data file doesn't have the required column structure.

---

## ✅ Solution 1: Remove Old Data and Restart (Recommended)

### Using Docker Compose

```bash
# Stop containers
docker-compose down

# Remove old data
rm -rf data/koi_with_relative_location.csv

# Restart (will fetch fresh data)
docker-compose up -d

# View logs
docker-compose logs -f
```

### Using Docker CLI

```bash
# Stop and remove container
docker stop exoplanet-api
docker rm exoplanet-api

# Remove old data
rm -rf data/koi_with_relative_location.csv

# Rebuild and run
docker build -t exoplanet-api .
docker run -d -p 8000:8000 --name exoplanet-api exoplanet-api

# View logs
docker logs -f exoplanet-api
```

---

## ✅ Solution 2: Regenerate Data Manually

```bash
# Run fetch.py in container
docker exec exoplanet-api python fetch.py

# Or run locally
python fetch.py

# Restart container
docker restart exoplanet-api
```

---

## ✅ Solution 3: Rebuild Everything

```bash
# Stop and remove everything
docker-compose down -v

# Remove data directory
rm -rf data/ model_outputs/

# Rebuild and start fresh
docker-compose up --build -d

# View logs
docker-compose logs -f
```

---

## 🔍 Verify the Fix

After applying any solution, check the logs:

```bash
# Should see:
# ✅ Data file exists and is valid
# ✅ Loaded model from...
# ✅ Predictions complete!

docker logs exoplanet-api
```

Access the API:
```bash
curl http://localhost:8000/
```

---

## 🎯 Prevention

To avoid this issue in the future:

1. **Always use volume mounts** to persist data:
   ```yaml
   volumes:
     - ./data:/app/data
     - ./model_outputs:/app/model_outputs
   ```

2. **Update data regularly**:
   ```bash
   # Regenerate data
   docker exec exoplanet-api python fetch.py
   docker restart exoplanet-api
   ```

3. **Keep Docker images updated**:
   ```bash
   docker-compose build --no-cache
   docker-compose up -d
   ```

---

## 📊 What the Fix Does

The updated Dockerfile now:
1. ✅ Checks if data file exists
2. ✅ Validates that `koi_disposition` column is present
3. ✅ Automatically regenerates data if column is missing
4. ✅ Provides clear error messages

The updated API now:
1. ✅ Validates data structure on startup
2. ✅ Shows helpful error messages
3. ✅ Lists available columns for debugging

---

## 🚀 Quick Commands

```bash
# Complete reset (removes all data)
docker-compose down -v && rm -rf data/ model_outputs/ && docker-compose up --build -d

# Just regenerate data
rm -rf data/ && docker restart exoplanet-api

# Check if it's working
curl http://localhost:8000/stats
```

---

**Your issue should now be fixed! 🎉**

