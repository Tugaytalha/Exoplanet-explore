# 🐳 Docker Deployment Guide

Bu proje Docker ve Docker Compose ile kolayca çalıştırılabilir.

## 🚀 Hızlı Başlangıç

### 1. Projeyi Çalıştırma

```bash
# Container'ı build et ve çalıştır
docker-compose up --build

# Veya arka planda çalıştırmak için
docker-compose up -d --build
```

### 2. API'ye Erişim

API başarıyla başladığında şu adreslere erişebilirsiniz:

- **API Ana Sayfa**: http://localhost:8000
- **API Dokümantasyonu (Swagger)**: http://localhost:8000/docs
- **Alternatif Dokümantasyon (ReDoc)**: http://localhost:8000/redoc

### 3. Örnek API İstekleri

```bash
# API durumunu kontrol et
curl http://localhost:8000/

# Tüm gezegenleri listele (ilk 10 kayıt)
curl http://localhost:8000/planets?limit=10

# Sadece onaylanmış gezegenleri getir
curl http://localhost:8000/planets?only_confirmed=true&limit=20

# Belirli bir gezegeni ID ile getir
curl http://localhost:8000/planets/10593626

# Model durumunu kontrol et
curl http://localhost:8000/model/status

# İstatistikleri görüntüle
curl http://localhost:8000/stats
```

## 📋 Gereksinimler

- Docker Desktop (Mac/Windows) veya Docker Engine (Linux)
- Docker Compose

## 🔧 Docker Compose Komutları

```bash
# Container'ı başlat
docker-compose up

# Container'ı arka planda başlat
docker-compose up -d

# Container'ı durdur
docker-compose down

# Container'ı durdur ve volume'leri sil
docker-compose down -v

# Logları görüntüle
docker-compose logs

# Logları canlı takip et
docker-compose logs -f

# Container içinde komut çalıştır
docker-compose exec exoplanet-api bash

# Modeli yeniden eğit (container içinde)
docker-compose exec exoplanet-api python train_koi_disposition.py
```

## 🏗️ Nasıl Çalışır?

### Container Başlangıç Süreci

1. **Container Başlatılır**: Docker compose, Dockerfile'ı kullanarak container'ı oluşturur
2. **Entrypoint Script Çalışır**: `entrypoint.sh` scripti otomatik olarak çalışır
3. **Veri Kontrolü**: 
   - Eğer `data/koi_with_relative_location.csv` yoksa, `fetch.py` çalıştırılır
   - NASA Exoplanet Archive'dan veriler indirilir
4. **Model Kontrolü**: 
   - Eğer eğitilmiş model varsa yüklenir
   - Yoksa API, dataset'teki mevcut disposition'ları kullanır
5. **API Başlatılır**: FastAPI uygulaması uvicorn ile başlatılır

### Veri Kalıcılığı

Docker Compose, şu dizinleri volume olarak mount eder:
- `./data` → Container içinde `/app/data`
- `./model_outputs` → Container içinde `/app/model_outputs`

Bu sayede:
- ✅ İndirilen veriler container yeniden başlatıldığında kaybolmaz
- ✅ Eğitilmiş modeller kalıcı olarak saklanır
- ✅ Container durdurulup başlatılırsa veriler korunur

## 🧪 Model Eğitimi

Container içinde model eğitmek için:

```bash
# Container'a bağlan
docker-compose exec exoplanet-api bash

# Modeli eğit
python train_koi_disposition.py

# Container'dan çık
exit
```

Eğitilen model `model_outputs/` dizinine kaydedilir ve API otomatik olarak en son modeli kullanır.

## 🔍 Sorun Giderme

### Port 8000 Kullanımda Hatası

Eğer 8000 portu başka bir uygulama tarafından kullanılıyorsa, `docker-compose.yml` dosyasında port numarasını değiştirebilirsiniz:

```yaml
ports:
  - "8080:8000"  # Host'ta 8080, container'da 8000
```

### Container Başlamıyor

```bash
# Logları kontrol et
docker-compose logs

# Container'ı yeniden build et
docker-compose up --build --force-recreate
```

### Veri İndirme Hatası

Eğer `fetch.py` çalışırken hata alıyorsanız:

```bash
# Container'a bağlan
docker-compose exec exoplanet-api bash

# Manuel olarak çalıştır
python fetch.py
```

### Memory/CPU Sorunları

Docker Desktop'ta resource limitleri artırın:
- Settings → Resources → Memory: En az 4GB
- Settings → Resources → CPUs: En az 2 CPU

## 📦 Development Modu

Kod değişikliklerini hot reload ile test etmek için `docker-compose.yml` dosyasını şu şekilde güncelleyin:

```yaml
services:
  exoplanet-api:
    # ... diğer ayarlar
    volumes:
      - ./data:/app/data
      - ./model_outputs:/app/model_outputs
      - ./api.py:/app/api.py  # Hot reload için
    command: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## 🌟 Üretim Ortamı için Öneriler

1. **Güvenlik**: 
   - API key authentication ekleyin
   - HTTPS kullanın (nginx reverse proxy)
   
2. **Performans**:
   - Uvicorn worker sayısını artırın: `--workers 4`
   - Gunicorn kullanın: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app`

3. **Monitoring**:
   - Health check endpoint'lerini izleyin
   - Log aggregation ekleyin (ELK, Datadog, etc.)

4. **Ölçeklendirme**:
   - Docker Swarm veya Kubernetes kullanın
   - Load balancer ekleyin

## 📚 İlgili Dosyalar

- `Dockerfile` - Container image tanımı
- `docker-compose.yml` - Servis orchestration
- `entrypoint.sh` - Container başlangıç scripti
- `requirements.txt` - Python bağımlılıkları

## 💡 İpuçları

- Container ilk kez başlatıldığında veri indirmesi 2-5 dakika sürebilir
- Model eğitimi 10-20 dakika sürer
- API, model olmadan da çalışabilir (dataset disposition'ları kullanır)
- Volume'leri silmek için: `docker-compose down -v` (dikkatli kullanın!)

---

**🎉 Başarıyla çalıştırıldı! API'nizin tadını çıkarın!**
