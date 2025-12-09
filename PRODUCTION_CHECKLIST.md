# Production Deployment Checklist

Use this checklist to ensure Hypothesis Forge is properly configured for production.

## Pre-Deployment

### Configuration
- [ ] Copy `env.example` to `.env` and configure all variables
- [ ] Set strong passwords for Neo4j
- [ ] Configure API keys (PubMed, etc.)
- [ ] Set appropriate log levels
- [ ] Configure simulation timeouts
- [ ] Set RL agent parameters

### Security
- [ ] Review and update all passwords
- [ ] Enable HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Set up secrets management (AWS Secrets Manager, etc.)
- [ ] Review file permissions
- [ ] Enable XSRF protection (Streamlit)
- [ ] Disable CORS if not needed

### Infrastructure
- [ ] Set up Neo4j instance (local or cloud)
- [ ] Configure database backups
- [ ] Set up monitoring (Prometheus, Datadog, etc.)
- [ ] Configure log aggregation
- [ ] Set up alerting
- [ ] Configure auto-scaling
- [ ] Set up load balancer

### Testing
- [ ] Run all tests: `pytest tests/ -v`
- [ ] Verify test coverage >80%
- [ ] Test data preprocessing
- [ ] Test simulation engine
- [ ] Test hypothesis generation
- [ ] Test dashboard functionality
- [ ] Test health check endpoints
- [ ] Load testing

## Deployment

### Docker
- [ ] Build Docker image: `docker build -t hypothesis-forge -f deploy/Dockerfile .`
- [ ] Test Docker image locally
- [ ] Push to container registry
- [ ] Configure Docker Compose for production
- [ ] Set resource limits (CPU, memory)
- [ ] Configure health checks

### Environment Variables
- [ ] Set all required environment variables
- [ ] Use secrets management for sensitive data
- [ ] Verify environment variables are loaded
- [ ] Test configuration validation

### Services
- [ ] Deploy Neo4j (if not using managed service)
- [ ] Deploy application
- [ ] Configure reverse proxy (nginx, etc.)
- [ ] Set up SSL certificates
- [ ] Configure domain name

## Post-Deployment

### Verification
- [ ] Health check endpoint responds: `/health`
- [ ] Dashboard is accessible
- [ ] Data processing works
- [ ] Simulations run successfully
- [ ] Hypothesis generation works
- [ ] Knowledge graph connects (if enabled)
- [ ] API integrations work (PubMed, arXiv)

### Monitoring
- [ ] Set up application monitoring
- [ ] Monitor CPU and memory usage
- [ ] Monitor disk space
- [ ] Monitor API response times
- [ ] Monitor error rates
- [ ] Set up alerts for critical issues

### Backup
- [ ] Configure automated backups
- [ ] Test backup restoration
- [ ] Document backup procedures
- [ ] Set backup retention policy

### Documentation
- [ ] Document deployment process
- [ ] Document rollback procedure
- [ ] Document troubleshooting steps
- [ ] Document monitoring setup
- [ ] Document backup/restore procedures

## Maintenance

### Regular Tasks
- [ ] Update dependencies monthly
- [ ] Review and rotate passwords quarterly
- [ ] Review logs weekly
- [ ] Monitor disk space
- [ ] Review performance metrics
- [ ] Update documentation as needed

### Security
- [ ] Regular security audits
- [ ] Dependency vulnerability scanning
- [ ] Review access logs
- [ ] Update SSL certificates
- [ ] Review firewall rules

## Troubleshooting

### Common Issues
- **Neo4j connection fails**: Check URI, credentials, and network
- **Dashboard not loading**: Check Streamlit logs and port configuration
- **Simulations timeout**: Increase timeout or optimize simulation code
- **Memory issues**: Adjust batch sizes or add more memory
- **API rate limits**: Adjust rate limiter settings

### Logs
- Application logs: `logs/hypothesis_forge.log`
- Docker logs: `docker logs <container_name>`
- System logs: Check system journal

### Health Checks
- Health endpoint: `http://localhost:8080/health`
- Liveness: `http://localhost:8080/health/live`
- Readiness: `http://localhost:8080/health/ready`

## Support

For issues or questions:
1. Check logs first
2. Review documentation
3. Check GitHub issues
4. Contact maintainers

---

**Last Updated:** 2025-12-09

