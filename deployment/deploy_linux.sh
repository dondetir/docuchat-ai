#!/bin/bash
# DocuChat Linux Deployment Script
# Supports Ubuntu 20.04+, Debian 11+, CentOS 8+, RHEL 8+

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="docuchat-web"
APP_DIR="/opt/docuchat"
APP_USER="docuchat"
SERVICE_NAME="docuchat-web.service"
PYTHON_VERSION="3.9"

echo -e "${BLUE}🚀 DocuChat Linux Deployment Script${NC}"
echo "=================================================="

# Function to detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo $ID
    elif type lsb_release >/dev/null 2>&1; then
        lsb_release -si | tr '[:upper:]' '[:lower:]'
    else
        echo "unknown"
    fi
}

# Function to install system dependencies
install_system_deps() {
    local distro=$(detect_distro)
    
    echo -e "${YELLOW}📦 Installing system dependencies...${NC}"
    
    case $distro in
        ubuntu|debian)
            sudo apt update
            sudo apt install -y \
                python3 \
                python3-pip \
                python3-venv \
                curl \
                wget \
                git \
                nginx \
                supervisor \
                ufw
            ;;
        centos|rhel|fedora)
            sudo yum update -y
            sudo yum install -y \
                python3 \
                python3-pip \
                curl \
                wget \
                git \
                nginx \
                supervisor \
                firewalld
            ;;
        *)
            echo -e "${RED}❌ Unsupported distribution: $distro${NC}"
            exit 1
            ;;
    esac
    
    echo -e "${GREEN}✅ System dependencies installed${NC}"
}

# Function to install Ollama
install_ollama() {
    echo -e "${YELLOW}🤖 Installing Ollama...${NC}"
    
    if command -v ollama >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Ollama already installed${NC}"
        return
    fi
    
    # Install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh
    
    # Start Ollama service
    sudo systemctl enable ollama
    sudo systemctl start ollama
    
    # Wait for service to be ready
    echo -e "${YELLOW}⏳ Waiting for Ollama service...${NC}"
    sleep 10
    
    # Pull required model
    echo -e "${YELLOW}📥 Downloading gemma3:270m model...${NC}"
    ollama pull gemma3:270m
    
    echo -e "${GREEN}✅ Ollama installed and model ready${NC}"
}

# Function to create application user
create_app_user() {
    echo -e "${YELLOW}👤 Creating application user...${NC}"
    
    if id "$APP_USER" &>/dev/null; then
        echo -e "${GREEN}✅ User $APP_USER already exists${NC}"
        return
    fi
    
    sudo useradd -r -s /bin/bash -d $APP_DIR $APP_USER
    sudo mkdir -p $APP_DIR
    sudo chown $APP_USER:$APP_USER $APP_DIR
    
    echo -e "${GREEN}✅ User $APP_USER created${NC}"
}

# Function to setup application
setup_application() {
    echo -e "${YELLOW}⚙️ Setting up DocuChat application...${NC}"
    
    # Copy application files
    sudo cp -r . $APP_DIR/
    sudo chown -R $APP_USER:$APP_USER $APP_DIR
    
    # Create virtual environment
    sudo -u $APP_USER python3 -m venv $APP_DIR/venv
    
    # Install Python dependencies
    sudo -u $APP_USER $APP_DIR/venv/bin/pip install --upgrade pip
    sudo -u $APP_USER $APP_DIR/venv/bin/pip install -r $APP_DIR/requirements.txt
    
    # Create required directories
    sudo -u $APP_USER mkdir -p $APP_DIR/{logs,chroma,documents}
    
    echo -e "${GREEN}✅ Application setup complete${NC}"
}

# Function to create systemd service
create_systemd_service() {
    echo -e "${YELLOW}🔧 Creating systemd service...${NC}"
    
    cat > /tmp/$SERVICE_NAME << EOF
[Unit]
Description=DocuChat Web UI - RAG Document Q&A System
After=network.target ollama.service
Wants=ollama.service
Requires=network.target

[Service]
Type=simple
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR
Environment=PYTHONPATH=$APP_DIR/src
Environment=GRADIO_SERVER_NAME=0.0.0.0
Environment=GRADIO_SERVER_PORT=7860
ExecStart=$APP_DIR/venv/bin/python $APP_DIR/web/web_app.py
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$APP_NAME

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$APP_DIR

[Install]
WantedBy=multi-user.target
EOF

    sudo mv /tmp/$SERVICE_NAME /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable $SERVICE_NAME
    
    echo -e "${GREEN}✅ Systemd service created${NC}"
}

# Function to configure Nginx reverse proxy
configure_nginx() {
    echo -e "${YELLOW}🌐 Configuring Nginx reverse proxy...${NC}"
    
    cat > /tmp/docuchat-nginx << 'EOF'
server {
    listen 80;
    server_name _;
    
    # Redirect HTTP to HTTPS (uncomment for production)
    # return 301 https://$server_name$request_uri;
    
    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Gradio
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts for large file uploads
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:7860/;
        access_log off;
    }
}

# HTTPS configuration (uncomment and configure for production)
# server {
#     listen 443 ssl http2;
#     server_name your-domain.com;
#     
#     ssl_certificate /path/to/ssl/cert.pem;
#     ssl_certificate_key /path/to/ssl/key.pem;
#     
#     location / {
#         proxy_pass http://127.0.0.1:7860;
#         # ... same proxy settings as above
#     }
# }
EOF

    sudo mv /tmp/docuchat-nginx /etc/nginx/sites-available/docuchat
    sudo ln -sf /etc/nginx/sites-available/docuchat /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    
    # Test Nginx configuration
    sudo nginx -t
    sudo systemctl enable nginx
    sudo systemctl restart nginx
    
    echo -e "${GREEN}✅ Nginx configured${NC}"
}

# Function to configure firewall
configure_firewall() {
    echo -e "${YELLOW}🔥 Configuring firewall...${NC}"
    
    local distro=$(detect_distro)
    
    case $distro in
        ubuntu|debian)
            sudo ufw --force enable
            sudo ufw allow ssh
            sudo ufw allow 'Nginx Full'
            sudo ufw allow 7860/tcp  # Direct access port
            sudo ufw status
            ;;
        centos|rhel|fedora)
            sudo systemctl enable firewalld
            sudo systemctl start firewalld
            sudo firewall-cmd --permanent --add-service=ssh
            sudo firewall-cmd --permanent --add-service=http
            sudo firewall-cmd --permanent --add-service=https
            sudo firewall-cmd --permanent --add-port=7860/tcp
            sudo firewall-cmd --reload
            ;;
    esac
    
    echo -e "${GREEN}✅ Firewall configured${NC}"
}

# Function to start services
start_services() {
    echo -e "${YELLOW}🚀 Starting DocuChat services...${NC}"
    
    # Start DocuChat
    sudo systemctl start $SERVICE_NAME
    
    # Check status
    sleep 5
    if sudo systemctl is-active --quiet $SERVICE_NAME; then
        echo -e "${GREEN}✅ DocuChat service started successfully${NC}"
    else
        echo -e "${RED}❌ DocuChat service failed to start${NC}"
        sudo journalctl -u $SERVICE_NAME --no-pager -l
        exit 1
    fi
}

# Function to show deployment summary
show_summary() {
    echo ""
    echo -e "${GREEN}🎉 DocuChat deployment completed!${NC}"
    echo "=================================================="
    echo -e "${BLUE}Service Status:${NC}"
    echo "  - DocuChat Web UI: $(sudo systemctl is-active $SERVICE_NAME)"
    echo "  - Ollama Service: $(sudo systemctl is-active ollama)"
    echo "  - Nginx Proxy: $(sudo systemctl is-active nginx)"
    echo ""
    echo -e "${BLUE}Access URLs:${NC}"
    echo "  - Direct: http://localhost:7860"
    echo "  - Via Nginx: http://$(hostname -I | awk '{print $1}')"
    echo ""
    echo -e "${BLUE}Service Management:${NC}"
    echo "  sudo systemctl start/stop/restart $SERVICE_NAME"
    echo "  sudo journalctl -u $SERVICE_NAME -f"
    echo ""
    echo -e "${BLUE}Configuration:${NC}"
    echo "  - App Directory: $APP_DIR"
    echo "  - Logs: $APP_DIR/logs/"
    echo "  - Vector DB: $APP_DIR/chroma/"
    echo "  - Documents: $APP_DIR/documents/"
}

# Main deployment function
main() {
    # Check if running as root for system setup
    if [[ $EUID -eq 0 ]]; then
        echo -e "${RED}❌ Don't run this script as root directly${NC}"
        echo "   This script will use sudo when needed"
        exit 1
    fi
    
    # Check if we can sudo
    if ! sudo -n true 2>/dev/null; then
        echo -e "${YELLOW}⚠️ This script requires sudo access${NC}"
        sudo echo "Sudo access confirmed"
    fi
    
    echo -e "${BLUE}🔍 Detected OS: $(detect_distro)${NC}"
    
    # Installation steps
    install_system_deps
    install_ollama
    create_app_user
    setup_application
    create_systemd_service
    configure_nginx
    configure_firewall
    start_services
    show_summary
    
    echo -e "${GREEN}✅ All done! DocuChat is running at http://$(hostname -I | awk '{print $1}')${NC}"
}

# Parse command line arguments
case "${1:-install}" in
    install)
        main
        ;;
    uninstall)
        echo -e "${YELLOW}🗑️ Uninstalling DocuChat...${NC}"
        sudo systemctl stop $SERVICE_NAME || true
        sudo systemctl disable $SERVICE_NAME || true
        sudo rm -f /etc/systemd/system/$SERVICE_NAME
        sudo systemctl daemon-reload
        sudo rm -f /etc/nginx/sites-available/docuchat
        sudo rm -f /etc/nginx/sites-enabled/docuchat
        sudo systemctl restart nginx
        sudo userdel -r $APP_USER || true
        sudo rm -rf $APP_DIR
        echo -e "${GREEN}✅ DocuChat uninstalled${NC}"
        ;;
    status)
        echo -e "${BLUE}📊 DocuChat Service Status:${NC}"
        sudo systemctl status $SERVICE_NAME --no-pager -l
        ;;
    logs)
        echo -e "${BLUE}📋 DocuChat Service Logs:${NC}"
        sudo journalctl -u $SERVICE_NAME -f
        ;;
    *)
        echo "Usage: $0 {install|uninstall|status|logs}"
        echo ""
        echo "  install   - Install and configure DocuChat (default)"
        echo "  uninstall - Remove DocuChat completely"
        echo "  status    - Show service status"
        echo "  logs      - Show service logs"
        exit 1
        ;;
esac