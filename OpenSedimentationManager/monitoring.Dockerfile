# Monitoring Dockerfile for OpenAcidMineDrainage
FROM prom/prometheus:latest

# Copy Prometheus configuration
COPY monitoring/prometheus.yml /etc/prometheus/prometheus.yml

# Expose Prometheus port
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:9090/-/healthy || exit 1

# Default command is inherited from base image
