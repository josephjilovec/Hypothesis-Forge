# Production Deployment Guide

This guide covers deploying Hypothesis Forge to production environments.

## Prerequisites

- Docker and Docker Compose installed
- Cloud platform account (AWS, DigitalOcean, GCP, etc.)
- Neo4j instance (local or cloud-hosted)
- Domain name (optional, for custom domain)

## Environment Configuration

1. Copy `env.example` to `.env`
2. Set all required environment variables:
   - Neo4j credentials
   - API keys (PubMed, etc.)
   - Security settings

## Docker Deployment

### Build and Run Locally

```bash
docker-compose -f deploy/docker-compose.yml up -d
```

### Production Build

```bash
docker build -t hypothesis-forge:latest -f deploy/Dockerfile .
```

## Cloud Deployment

### AWS Elastic Beanstalk

1. Install EB CLI: `pip install awscli eb-cli`
2. Configure: `eb init -p docker hypothesis-forge --region us-east-1`
3. Create environment: `eb create hypothesis-forge-prod`
4. Set environment variables: `eb setenv NEO4J_URI=... NEO4J_PASSWORD=...`
5. Deploy: `eb deploy`

### DigitalOcean App Platform

1. Connect repository to DigitalOcean
2. Configure build settings:
   - Dockerfile path: `deploy/Dockerfile`
   - Port: 8501
3. Set environment variables
4. Deploy

### Google Cloud Run

```bash
gcloud run deploy hypothesis-forge \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501
```

## Security Checklist

- [ ] Use strong passwords for Neo4j
- [ ] Enable HTTPS/TLS
- [ ] Set up firewall rules
- [ ] Use secrets management (AWS Secrets Manager, etc.)
- [ ] Enable logging and monitoring
- [ ] Set up backup strategy
- [ ] Configure rate limiting
- [ ] Review and update dependencies regularly

## Monitoring

- Set up application monitoring (Datadog, New Relic, etc.)
- Monitor Neo4j performance
- Track API usage and costs
- Set up alerts for errors

## Scaling

- Use load balancer for multiple instances
- Configure auto-scaling based on CPU/memory
- Use managed Neo4j service for better performance
- Consider caching for frequently accessed data

## Backup and Recovery

- Regular backups of Neo4j database
- Backup simulation results and hypotheses
- Test recovery procedures
- Document disaster recovery plan

## Performance Optimization

- Enable Neo4j query caching
- Optimize simulation parameters
- Use CDN for static assets
- Monitor and optimize database queries

## Maintenance

- Regular dependency updates
- Security patches
- Performance monitoring
- Log rotation and cleanup

