# DesignStream AI

A modern AI-powered design and rendering platform for interior design and furniture placement.

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.13+
- Make (optional, for convenience commands)

### Development Setup

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd designstreamaigrok
   cp .env.example .env
   ```

2. **Start development environment:**
   ```bash
   make setup
   # or manually:
   make install
   make up
   make migrate
   make seed
   make minio-setup
   ```

3. **Start the application:**
   ```bash
   make dev
   ```

4. **Access the application:**
   - Application: http://localhost:5002
   - MinIO Console: http://localhost:9001 (admin/minioadmin123)
   - Postgres: localhost:5432 (designstream/designstream_password)

## 🛠️ Development Commands

### Database Operations
```bash
make psql          # Connect to main database
make psql-dev      # Connect to development database
make migrate       # Run migrations
make migrate-dev   # Run migrations on dev database
make seed          # Seed database with sample data
```

### Services Management
```bash
make up            # Start all services
make up-dev        # Start with development database
make down          # Stop all services
make logs          # View all logs
make logs-postgres # View Postgres logs
make logs-minio    # View MinIO logs
```

### MinIO Operations
```bash
make minio-setup   # Setup MinIO buckets
make minio-logs    # View MinIO logs
```

### Testing & Quality
```bash
make test          # Run tests
make clean         # Clean up containers and volumes
make clean-data    # Clean only data volumes
```

## 🏗️ Architecture

### Services
- **Postgres 16** with pgvector extension for AI embeddings
- **MinIO** for S3-compatible object storage
- **Redis** for caching and Celery task queue
- **Flask** application with AI/ML services

### Database Schema
- **Products**: Shopify product data
- **Variants**: Product variants with AI embeddings
- **Images**: Product images with cutout URLs
- **Render Sessions**: User rendering sessions
- **AI Recommendations**: ML recommendation data
- **Analytics**: Usage tracking and events

### AI/ML Features
- **CLIP Embeddings**: Product similarity matching
- **YOLO Segmentation**: Automatic object detection
- **Background Removal**: AI-powered matting
- **Recommendation Engine**: ML-based product suggestions

## 🔧 Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:

```bash
# Database
DATABASE_URL=postgresql+psycopg2://designstream:designstream_password@localhost:5432/designstreamdb

# S3/MinIO
S3_BUCKET=designstream-uploads
S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin123

# Shopify
SHOPIFY_API_KEY=your_api_key
SHOPIFY_API_SECRET=your_api_secret

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
```

## 🧪 Testing

### Run Tests
```bash
make test
```

### Test Coverage
```bash
pytest --cov=app --cov-report=html
```

## 🚀 Deployment

### Docker
```bash
docker build -t designstream-ai .
docker run -p 5002:5002 designstream-ai
```

### Production
The application includes GitHub Actions CI/CD pipeline with:
- Automated testing
- Security scanning
- Docker image building
- Staging/Production deployment

## 📁 Project Structure

```
├── app/                    # Application code
│   ├── routers/           # API routes
│   ├── services/          # Business logic
│   ├── static/            # Static files
│   └── templates/         # HTML templates
├── migrations/            # Database migrations
│   ├── 001_initial_schema.sql
│   ├── 002_add_indexes.sql
│   └── seeds/             # Sample data
├── tests/                 # Test files
├── .github/workflows/     # CI/CD pipelines
├── docker-compose.yml     # Development services
├── Dockerfile            # Application container
├── Makefile              # Development commands
└── requirements.txt      # Python dependencies
```

## 🔍 API Endpoints

### Render Service
- `POST /render/render` - Generate room renderings
- `GET /render/health` - Health check

### Matting Service
- `POST /matting/preview` - Background removal
- `GET /matting/health` - Health check

### Recommender Service
- `POST /recommender/recommend` - Get AI recommendations
- `GET /recommender/health` - Health check

### Widget Service
- `GET /widget.js` - Embeddable widget
- `POST /widget/recommendations` - Widget recommendations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the CI/CD logs for deployment issues

