# 🚀 DocuChat Complete Deployment Guide

## 📋 **Table of Contents**

1. [Quick Start](#quick-start)
2. [Windows Deployment](#windows-deployment)
3. [Linux Deployment](#linux-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Production Considerations](#production-considerations)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+ installed
- At least 4GB RAM (8GB recommended)
- 10GB free disk space

### **1. Install Dependencies**
```bash
# Clone or download DocuChat
cd DocuChat

# Install Python dependencies
pip install -r requirements.txt
```

### **2. Install & Start Ollama**
```bash
# Install Ollama (https://ollama.ai)
# Windows: Download installer
# Linux: curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull required model (in new terminal)
ollama pull gemma3:270m
```

### **3. Launch DocuChat Web UI**
```bash
# Launch web application
python web/run_web_app.py

# Open browser to: http://localhost:7860
```

---

## 🖥️ **Windows Deployment**

### **Option 1: Development Setup**
```powershell
# 1. Install Python from Microsoft Store or python.org
# 2. Install Ollama from https://ollama.ai

# Setup DocuChat
cd C:\DocuChat
pip install -r requirements.txt

# Start services
ollama serve  # Keep this terminal open
python web/run_web_app.py  # In new terminal

# Access: http://localhost:7860
```

### **Option 2: Windows Service (Production)**
```powershell
# Install Windows service dependencies
pip install pywin32

# Setup Windows service
python deployment/deploy_windows_service.py setup

# Install as Administrator
.\install_service.bat

# Service will auto-start on system boot
# Access: http://localhost:7860
```

**Service Management:**
```powershell
# Control Windows service
net start DocuChatWebUI
net stop DocuChatWebUI
net restart DocuChatWebUI

# View service status
sc query DocuChatWebUI

# Remove service
python deployment/deploy_windows_service.py remove
```

### **Option 3: Docker Desktop (Windows)**
```powershell
# Install Docker Desktop
# Start Docker Desktop

# Build and run
docker-compose up -d

# Access: http://localhost:7860
```

---

## 🐧 **Linux Deployment**

### **Option 1: Automated Installation (Recommended)**
```bash
# Make script executable
chmod +x deployment/deploy_linux.sh

# Install everything automatically
./deployment/deploy_linux.sh install

# Access: http://your-server-ip
```

**What the script does:**
- ✅ Installs system dependencies
- ✅ Sets up Ollama with model
- ✅ Creates dedicated user account
- ✅ Installs DocuChat as systemd service
- ✅ Configures Nginx reverse proxy
- ✅ Sets up firewall rules
- ✅ Auto-starts on system boot

### **Option 2: Manual Installation**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip python3-venv nginx

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull gemma3:270m

# Setup DocuChat
sudo mkdir -p /opt/docuchat
sudo cp -r . /opt/docuchat/
cd /opt/docuchat

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run application
python web/web_app.py
```

### **Service Management (systemd)**
```bash
# Service control
sudo systemctl start docuchat-web
sudo systemctl stop docuchat-web
sudo systemctl restart docuchat-web
sudo systemctl status docuchat-web

# Enable/disable auto-start
sudo systemctl enable docuchat-web
sudo systemctl disable docuchat-web

# View logs
sudo journalctl -u docuchat-web -f

# Configuration files
/etc/systemd/system/docuchat-web.service
/etc/nginx/sites-available/docuchat
```

### **Maintenance Commands**
```bash
# Check status
./deployment/deploy_linux.sh status

# View logs
./deployment/deploy_linux.sh logs

# Uninstall completely
./deployment/deploy_linux.sh uninstall
```

---

## 🐳 **Docker Deployment**

### **Option 1: Docker Compose (Recommended)**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up -d --build
```

**Services included:**
- `docuchat-web`: Web UI application
- `ollama`: LLM service with models
- `nginx`: Reverse proxy (optional)

### **Option 2: Docker Run**
```bash
# Build image
docker build -t docuchat-web .

# Run container
docker run -d \
  --name docuchat-web \
  -p 7860:7860 \
  -v $(pwd)/chroma_web:/app/chroma_web \
  -v $(pwd)/documents:/app/documents:ro \
  --env OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  docuchat-web

# Start Ollama separately
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  ollama/ollama:latest

# Pull model
docker exec ollama ollama pull gemma3:270m
```

### **Docker Configuration**

**Environment Variables:**
```bash
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
OLLAMA_BASE_URL=http://ollama:11434
PYTHONUNBUFFERED=1
```

**Volume Mounts:**
```bash
# Persistent data
./chroma_web:/app/chroma_web      # Vector database
./documents:/app/documents:ro    # Document directory (read-only)
./logs:/app/logs                 # Application logs

# Ollama data
ollama-data:/root/.ollama        # Models and configuration
```

---

## ☁️ **Cloud Deployment**

### **Option 1: Automated Cloud Setup**
```bash
# Basic cloud deployment
chmod +x deployment/deploy_cloud.sh
./deployment/deploy_cloud.sh

# With custom domain and SSL
./deployment/deploy_cloud.sh \
  --domain docuchat.example.com \
  --email admin@example.com \
  --ssl
```

### **Option 2: Platform-Specific**

#### **AWS EC2**
```bash
# Launch Ubuntu 20.04+ instance
# Security Group: Allow ports 22, 80, 443

# SSH to instance
ssh -i key.pem ubuntu@ec2-instance

# Run deployment
wget https://github.com/user/docuchat/deployment/deploy_cloud.sh
chmod +x deployment/deploy_cloud.sh
./deployment/deploy_cloud.sh --provider aws
```

#### **Google Cloud Platform**
```bash
# Create Compute Engine instance
gcloud compute instances create docuchat-vm \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --machine-type=e2-standard-2 \
  --tags=http-server,https-server

# SSH and deploy
gcloud compute ssh docuchat-vm
./deployment/deploy_cloud.sh --provider gcp
```

#### **Azure**
```bash
# Create VM
az vm create \
  --resource-group DocuChatRG \
  --name docuchat-vm \
  --image UbuntuLTS \
  --size Standard_B2s \
  --admin-username azureuser

# SSH and deploy
ssh azureuser@vm-ip-address
./deployment/deploy_cloud.sh --provider azure
```

#### **DigitalOcean**
```bash
# Create droplet via web interface or CLI
doctl compute droplet create docuchat \
  --image ubuntu-20-04-x64 \
  --size s-2vcpu-4gb \
  --region nyc1

# SSH and deploy
ssh root@droplet-ip
./deployment/deploy_cloud.sh --provider digitalocean
```

### **Cloud Configuration Features**
- ✅ SSL certificate with Let's Encrypt
- ✅ Nginx reverse proxy with security headers
- ✅ Rate limiting and DDoS protection
- ✅ Automated monitoring and alerting
- ✅ Daily backups to cloud storage
- ✅ Auto-restart on failures
- ✅ Resource usage monitoring

---

## 🔒 **Production Considerations**

### **Security**

#### **SSL/TLS Certificate**
```bash
# Automatic with Let's Encrypt
sudo certbot --nginx \
  --email your-email@domain.com \
  --domains your-domain.com \
  --agree-tos --non-interactive

# Manual certificate
sudo certbot certonly --nginx -d your-domain.com
```

#### **Firewall Configuration**
```bash
# Ubuntu/Debian
sudo ufw enable
sudo ufw allow 'Nginx Full'
sudo ufw allow ssh

# CentOS/RHEL
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

#### **Access Control**
```nginx
# Add to Nginx config for IP restriction
location / {
    allow 192.168.1.0/24;  # Local network
    allow 10.0.0.0/8;      # VPN
    deny all;
    
    proxy_pass http://127.0.0.1:7860;
    # ... other proxy settings
}
```

### **Performance Optimization**

#### **Resource Allocation**
```bash
# Recommended server specs:
# - CPU: 4+ cores
# - RAM: 8GB+ (16GB for large document sets)
# - Storage: 50GB+ SSD
# - Network: 100Mbps+
```

#### **Caching Configuration**
```nginx
# Nginx caching
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}

# Proxy caching
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=docuchat:10m;
location / {
    proxy_cache docuchat;
    proxy_cache_valid 200 1m;
    # ... other settings
}
```

#### **Database Optimization**
```python
# ChromaDB optimization in web_app.py
"hnsw:space": "cosine",
"hnsw:search_ef": 100,
"hnsw:M": 16
```

### **Monitoring & Logging**

#### **Health Checks**
```bash
# Built-in health check endpoint
curl http://localhost:7860/health

# Service monitoring
systemctl status docuchat-web
journalctl -u docuchat-web -f
```

#### **Log Rotation**
```bash
# Configure logrotate
sudo cat > /etc/logrotate.d/docuchat << EOF
/opt/docuchat/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    notifempty
    create 0644 docuchat docuchat
    postrotate
        systemctl reload docuchat-web
    endscript
}
EOF
```

#### **Monitoring Script**
```bash
# Automated monitoring (created by deploy scripts)
/usr/local/bin/docuchat-monitor.sh

# Monitor logs
tail -f /var/log/docuchat-monitor.log
```

### **Backup & Recovery**

#### **Automated Backups**
```bash
# Daily backup script (created by deploy scripts)
/usr/local/bin/docuchat-backup.sh

# Manual backup
sudo tar -czf docuchat-backup.tar.gz \
  /opt/docuchat/chroma_web \
  /opt/docuchat/logs \
  /etc/nginx/sites-available/docuchat \
  /etc/systemd/system/docuchat-web.service
```

#### **Recovery Process**
```bash
# Restore from backup
sudo systemctl stop docuchat-web
sudo tar -xzf docuchat-backup.tar.gz -C /
sudo systemctl start docuchat-web
```

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **1. Ollama Connection Failed**
```bash
# Check Ollama service
systemctl status ollama
curl http://localhost:11434/api/tags

# Restart Ollama
sudo systemctl restart ollama
ollama pull gemma3:270m

# Check firewall
sudo ufw status
sudo ufw allow 11434/tcp
```

#### **2. Gradio Port Already in Use**
```bash
# Find process using port 7860
sudo lsof -i :7860
sudo netstat -tulpn | grep 7860

# Kill process
sudo kill -9 <process_id>

# Use different port
python web/web_app.py --port 7861
```

#### **3. Permission Denied**
```bash
# Fix ownership
sudo chown -R docuchat:docuchat /opt/docuchat

# Fix permissions
sudo chmod -R 755 /opt/docuchat
sudo chmod +x /opt/docuchat/web_app.py
```

#### **4. Out of Memory**
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Add swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Monitor memory
watch -n 1 'free -m'
```

#### **5. SSL Certificate Issues**
```bash
# Check certificate status
sudo certbot certificates

# Renew certificate
sudo certbot renew --dry-run
sudo certbot renew

# Fix nginx config
sudo nginx -t
sudo systemctl reload nginx
```

### **Log Locations**
```bash
# Application logs
/opt/docuchat/logs/web_app.log

# Service logs  
sudo journalctl -u docuchat-web.service

# Nginx logs
/var/log/nginx/access.log
/var/log/nginx/error.log

# System logs
/var/log/syslog
/var/log/messages  # CentOS/RHEL
```

### **Debug Mode**
```bash
# Run in debug mode
export GRADIO_DEBUG=1
python web/web_app.py

# Increase log verbosity
export DOCUCHAT_LOG_LEVEL=DEBUG
systemctl restart docuchat-web
```

### **Performance Issues**
```bash
# Check system resources
top
htop
iotop
df -h

# Optimize ChromaDB
# Reduce chunk_size in settings
# Limit top_k results
# Clean old embeddings

# Monitor network
iftop
netstat -i
```

---

## 📞 **Support**

### **Getting Help**
- **Documentation**: Check `docs/UI.md` for detailed technical specs
- **Logs**: Always check logs first (`journalctl -u docuchat-web -f`)
- **Health Check**: Visit `http://your-domain/health`
- **Community**: GitHub Issues and Discussions

### **Reporting Issues**
1. Check logs for error messages
2. Include system information (`uname -a`, `python --version`)
3. Provide deployment method (Docker, Linux, Windows)
4. Include relevant configuration files
5. Describe steps to reproduce

---

## ✅ **Deployment Checklist**

### **Pre-deployment**
- [ ] System meets minimum requirements (4GB RAM, 10GB storage)
- [ ] Python 3.8+ installed
- [ ] Ollama installed and accessible
- [ ] Required ports available (7860, 11434)
- [ ] Domain name configured (if using SSL)

### **Post-deployment**
- [ ] Web UI accessible at configured URL
- [ ] Can upload and process documents
- [ ] Chat functionality works with citations
- [ ] SSL certificate valid (if enabled)
- [ ] Service auto-starts on boot
- [ ] Monitoring and backups configured
- [ ] Firewall rules properly configured
- [ ] Performance acceptable for expected load

### **Security Review**
- [ ] Default passwords changed
- [ ] Unnecessary ports closed
- [ ] SSL/TLS properly configured
- [ ] Access logs enabled
- [ ] Regular updates scheduled
- [ ] Backup and recovery tested

---

**🎉 Congratulations! Your DocuChat deployment is complete!**

Access your new document Q&A system and start chatting with your documents through the beautiful web interface!