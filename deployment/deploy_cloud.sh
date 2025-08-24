#!/bin/bash
# DocuChat Cloud Deployment Script
# Supports AWS, GCP, Azure, DigitalOcean, and other cloud providers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}☁️ DocuChat Cloud Deployment Script${NC}"
echo "=================================================="

# Default configuration
DOMAIN_NAME=""
EMAIL=""
ENABLE_SSL=false
CLOUD_PROVIDER=""
INSTANCE_SIZE="medium"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)
            DOMAIN_NAME="$2"
            shift 2
            ;;
        --email)
            EMAIL="$2"
            shift 2
            ;;
        --ssl)
            ENABLE_SSL=true
            shift
            ;;
        --provider)
            CLOUD_PROVIDER="$2"
            shift 2
            ;;
        --size)
            INSTANCE_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --domain DOMAIN    Domain name for the deployment"
            echo "  --email EMAIL      Email for SSL certificate"
            echo "  --ssl              Enable SSL/TLS with Let's Encrypt"
            echo "  --provider CLOUD   Cloud provider (aws|gcp|azure|do)"
            echo "  --size SIZE        Instance size (small|medium|large)"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --domain docuchat.example.com --email admin@example.com --ssl"
            echo "  $0 --provider aws --size large"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Function to detect cloud provider
detect_cloud_provider() {
    if [ -n "$CLOUD_PROVIDER" ]; then
        echo $CLOUD_PROVIDER
        return
    fi
    
    # Try to detect based on metadata services
    if curl -s --connect-timeout 3 http://169.254.169.254/latest/meta-data/ >/dev/null 2>&1; then
        echo "aws"
    elif curl -s --connect-timeout 3 -H "Metadata-Flavor: Google" http://169.254.169.254/computeMetadata/v1/ >/dev/null 2>&1; then
        echo "gcp"
    elif curl -s --connect-timeout 3 -H "Metadata: true" http://169.254.169.254/metadata/instance >/dev/null 2>&1; then
        echo "azure"
    elif curl -s --connect-timeout 3 http://169.254.169.254/metadata/v1/ >/dev/null 2>&1; then
        echo "digitalocean"
    else
        echo "unknown"
    fi
}

# Function to get public IP
get_public_ip() {
    local provider=$(detect_cloud_provider)
    
    case $provider in
        aws)
            curl -s http://169.254.169.254/latest/meta-data/public-ipv4
            ;;
        gcp)
            curl -s -H "Metadata-Flavor: Google" \
                http://169.254.169.254/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip
            ;;
        azure)
            curl -s -H "Metadata: true" \
                "http://169.254.169.254/metadata/instance/network/interface/0/ipv4/ipAddress/0/publicIpAddress?api-version=2017-08-01&format=text"
            ;;
        digitalocean)
            curl -s http://169.254.169.254/metadata/v1/interfaces/public/0/ipv4/address
            ;;
        *)
            curl -s https://ipinfo.io/ip
            ;;
    esac
}

# Function to install SSL certificate with Let's Encrypt
install_ssl_certificate() {
    if [ -z "$DOMAIN_NAME" ] || [ -z "$EMAIL" ]; then
        echo -e "${YELLOW}⚠️ Domain name and email required for SSL certificate${NC}"
        echo "   Use --domain and --email options"
        return 1
    fi
    
    echo -e "${YELLOW}🔒 Installing SSL certificate...${NC}"
    
    # Install certbot
    sudo apt update
    sudo apt install -y certbot python3-certbot-nginx
    
    # Get certificate
    sudo certbot --nginx \
        --non-interactive \
        --agree-tos \
        --email "$EMAIL" \
        --domains "$DOMAIN_NAME" \
        --redirect
    
    # Setup auto-renewal
    sudo systemctl enable certbot.timer
    sudo systemctl start certbot.timer
    
    echo -e "${GREEN}✅ SSL certificate installed${NC}"
}

# Function to configure Nginx for cloud deployment
configure_cloud_nginx() {
    echo -e "${YELLOW}🌐 Configuring Nginx for cloud deployment...${NC}"
    
    local server_name="${DOMAIN_NAME:-_}"
    local public_ip=$(get_public_ip)
    
    cat > /tmp/docuchat-cloud-nginx << EOF
# DocuChat Cloud Nginx Configuration
server {
    listen 80;
    server_name $server_name;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    
    # Rate limiting
    limit_req zone=api burst=20 nodelay;
    
    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # Buffer settings for better performance
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:7860/;
        access_log off;
    }
    
    # Static file caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}

# Rate limiting configuration
limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/m;
EOF

    sudo mv /tmp/docuchat-cloud-nginx /etc/nginx/sites-available/docuchat
    sudo ln -sf /etc/nginx/sites-available/docuchat /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    
    # Test configuration
    sudo nginx -t
    sudo systemctl reload nginx
    
    echo -e "${GREEN}✅ Nginx configured for cloud deployment${NC}"
    echo -e "${BLUE}Public IP: $public_ip${NC}"
    if [ -n "$DOMAIN_NAME" ]; then
        echo -e "${BLUE}Domain: $DOMAIN_NAME${NC}"
    fi
}

# Function to configure cloud firewall
configure_cloud_firewall() {
    echo -e "${YELLOW}🔥 Configuring cloud firewall...${NC}"
    
    local provider=$(detect_cloud_provider)
    
    # Configure local firewall first
    sudo ufw --force enable
    sudo ufw allow ssh
    sudo ufw allow 'Nginx Full'
    
    case $provider in
        aws)
            echo -e "${BLUE}ℹ️ AWS Security Group configuration required:${NC}"
            echo "   - HTTP (80): 0.0.0.0/0"
            echo "   - HTTPS (443): 0.0.0.0/0"
            echo "   - SSH (22): Your IP only"
            ;;
        gcp)
            echo -e "${BLUE}ℹ️ GCP Firewall rules may be required:${NC}"
            echo "   gcloud compute firewall-rules create allow-http --allow tcp:80"
            echo "   gcloud compute firewall-rules create allow-https --allow tcp:443"
            ;;
        azure)
            echo -e "${BLUE}ℹ️ Azure NSG rules may be required:${NC}"
            echo "   Allow HTTP (80) and HTTPS (443) inbound"
            ;;
        digitalocean)
            echo -e "${BLUE}ℹ️ DigitalOcean Firewall configuration:${NC}"
            echo "   HTTP/HTTPS: All IPv4/IPv6"
            echo "   SSH: Your IP only"
            ;;
    esac
    
    echo -e "${GREEN}✅ Local firewall configured${NC}"
}

# Function to setup monitoring
setup_monitoring() {
    echo -e "${YELLOW}📊 Setting up monitoring...${NC}"
    
    # Create monitoring script
    cat > /tmp/docuchat-monitor.sh << 'EOF'
#!/bin/bash
# DocuChat Monitoring Script

LOG_FILE="/var/log/docuchat-monitor.log"
SERVICE_NAME="docuchat-web.service"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> $LOG_FILE
}

# Check DocuChat service
if ! systemctl is-active --quiet $SERVICE_NAME; then
    log_message "ERROR: DocuChat service is down, restarting..."
    systemctl restart $SERVICE_NAME
    if systemctl is-active --quiet $SERVICE_NAME; then
        log_message "INFO: DocuChat service restarted successfully"
    else
        log_message "CRITICAL: DocuChat service failed to restart"
    fi
fi

# Check Ollama service
if ! systemctl is-active --quiet ollama; then
    log_message "ERROR: Ollama service is down, restarting..."
    systemctl restart ollama
    sleep 10  # Wait for Ollama to start
fi

# Check disk space
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 85 ]; then
    log_message "WARNING: Disk usage is ${DISK_USAGE}%"
fi

# Check memory usage
MEM_USAGE=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
if [ $(echo "$MEM_USAGE > 90" | bc) -eq 1 ]; then
    log_message "WARNING: Memory usage is ${MEM_USAGE}%"
fi
EOF

    sudo mv /tmp/docuchat-monitor.sh /usr/local/bin/docuchat-monitor.sh
    sudo chmod +x /usr/local/bin/docuchat-monitor.sh
    
    # Create cron job for monitoring
    echo "*/5 * * * * /usr/local/bin/docuchat-monitor.sh" | sudo crontab -
    
    echo -e "${GREEN}✅ Monitoring setup complete${NC}"
    echo -e "${BLUE}Monitor logs: tail -f /var/log/docuchat-monitor.log${NC}"
}

# Function to create backup script
create_backup_script() {
    echo -e "${YELLOW}💾 Creating backup script...${NC}"
    
    cat > /tmp/docuchat-backup.sh << 'EOF'
#!/bin/bash
# DocuChat Backup Script

BACKUP_DIR="/opt/backups/docuchat"
APP_DIR="/opt/docuchat"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="docuchat_backup_${DATE}.tar.gz"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create backup
tar -czf "${BACKUP_DIR}/${BACKUP_FILE}" \
    --exclude="${APP_DIR}/venv" \
    --exclude="${APP_DIR}/logs/*" \
    -C /opt docuchat

# Keep only last 7 backups
find $BACKUP_DIR -name "docuchat_backup_*.tar.gz" -mtime +7 -delete

echo "Backup created: ${BACKUP_DIR}/${BACKUP_FILE}"
EOF

    sudo mv /tmp/docuchat-backup.sh /usr/local/bin/docuchat-backup.sh
    sudo chmod +x /usr/local/bin/docuchat-backup.sh
    
    # Schedule daily backups
    echo "0 2 * * * /usr/local/bin/docuchat-backup.sh" | sudo crontab -u root -
    
    echo -e "${GREEN}✅ Backup script created${NC}"
    echo -e "${BLUE}Manual backup: sudo /usr/local/bin/docuchat-backup.sh${NC}"
}

# Function to optimize for cloud deployment
optimize_for_cloud() {
    echo -e "${YELLOW}⚡ Optimizing for cloud deployment...${NC}"
    
    # Configure swap if not present (helps with memory management)
    if [ ! -f /swapfile ]; then
        sudo fallocate -l 2G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi
    
    # Optimize kernel parameters
    cat >> /tmp/cloud-sysctl.conf << 'EOF'
# DocuChat Cloud Optimizations
vm.swappiness=10
net.core.rmem_max=16777216
net.core.wmem_max=16777216
net.ipv4.tcp_rmem=4096 87380 16777216
net.ipv4.tcp_wmem=4096 65536 16777216
EOF
    
    sudo mv /tmp/cloud-sysctl.conf /etc/sysctl.d/99-docuchat.conf
    sudo sysctl -p /etc/sysctl.d/99-docuchat.conf
    
    echo -e "${GREEN}✅ Cloud optimizations applied${NC}"
}

# Function to show cloud deployment summary
show_cloud_summary() {
    local public_ip=$(get_public_ip)
    local provider=$(detect_cloud_provider)
    
    echo ""
    echo -e "${GREEN}🎉 DocuChat Cloud Deployment Complete!${NC}"
    echo "=================================================="
    echo -e "${BLUE}Cloud Provider:${NC} $provider"
    echo -e "${BLUE}Public IP:${NC} $public_ip"
    if [ -n "$DOMAIN_NAME" ]; then
        echo -e "${BLUE}Domain:${NC} $DOMAIN_NAME"
        echo -e "${BLUE}URL:${NC} http${ENABLE_SSL:+s}://$DOMAIN_NAME"
    else
        echo -e "${BLUE}URL:${NC} http://$public_ip"
    fi
    echo ""
    echo -e "${BLUE}Service Management:${NC}"
    echo "  sudo systemctl status docuchat-web.service"
    echo "  sudo journalctl -u docuchat-web.service -f"
    echo ""
    echo -e "${BLUE}Monitoring:${NC}"
    echo "  tail -f /var/log/docuchat-monitor.log"
    echo ""
    echo -e "${BLUE}Backups:${NC}"
    echo "  sudo /usr/local/bin/docuchat-backup.sh"
    echo "  Backups stored in: /opt/backups/docuchat/"
    echo ""
    echo -e "${BLUE}SSL Certificate (if enabled):${NC}"
    if [ "$ENABLE_SSL" = true ]; then
        echo "  Auto-renewal: systemctl status certbot.timer"
        echo "  Manual renewal: sudo certbot renew"
    else
        echo "  Enable with: sudo certbot --nginx"
    fi
}

# Main cloud deployment function
main() {
    local provider=$(detect_cloud_provider)
    echo -e "${BLUE}🔍 Detected Cloud Provider: $provider${NC}"
    
    # Run base Linux installation first
    if [ -f "./deployment/deploy_linux.sh" ]; then
        echo -e "${YELLOW}📋 Running base Linux installation...${NC}"
        chmod +x ./deployment/deploy_linux.sh
        ./deployment/deploy_linux.sh install
    elif [ -f "./deploy_linux.sh" ]; then
        # Fallback for running from deployment directory
        echo -e "${YELLOW}📋 Running base Linux installation...${NC}"
        chmod +x ./deploy_linux.sh
        ./deploy_linux.sh install
    else
        echo -e "${RED}❌ deploy_linux.sh not found${NC}"
        echo "   Please run this script from the DocuChat directory or deployment/ directory"
        exit 1
    fi
    
    # Cloud-specific configurations
    configure_cloud_nginx
    configure_cloud_firewall
    
    if [ "$ENABLE_SSL" = true ]; then
        install_ssl_certificate
    fi
    
    setup_monitoring
    create_backup_script
    optimize_for_cloud
    show_cloud_summary
    
    echo -e "${GREEN}✅ Cloud deployment completed successfully!${NC}"
}

# Run main function
main