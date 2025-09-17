.PHONY: help up down logs psql migrate seed test clean install dev

# Default target
help: ## Show this help message
	@echo "DesignStream AI Development Commands"
	@echo "===================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment setup
install: ## Install dependencies
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Development environment
up: ## Start all services (Postgres, MinIO, Redis)
	docker-compose up -d postgres minio redis
	@echo "⏳ Waiting for services to be ready..."
	@sleep 10
	@echo "✅ Services started:"
	@echo "   📊 Postgres: localhost:5432"
	@echo "   🗄️  MinIO: localhost:9000 (admin: minioadmin/minioadmin123)"
	@echo "   🔴 Redis: localhost:6379"

up-dev: ## Start development environment with dev database
	docker-compose up -d postgres-dev minio redis
	@echo "⏳ Waiting for services to be ready..."
	@sleep 10
	@echo "✅ Development services started:"
	@echo "   📊 Postgres Dev: localhost:5433"
	@echo "   🗄️  MinIO: localhost:9000"
	@echo "   🔴 Redis: localhost:6379"

down: ## Stop all services
	docker-compose down
	@echo "✅ All services stopped"

logs: ## Show logs for all services
	docker-compose logs -f

logs-postgres: ## Show Postgres logs
	docker-compose logs -f postgres

logs-minio: ## Show MinIO logs
	docker-compose logs -f minio

# Database operations
psql: ## Connect to Postgres database
	@echo "Connecting to Postgres..."
	@echo "Database: designstreamdb"
	@echo "User: designstream"
	@echo "Password: designstream_password"
	@echo ""
	PGPASSWORD=designstream_password psql -h localhost -p 5432 -U designstream -d designstreamdb

psql-dev: ## Connect to development Postgres database
	@echo "Connecting to Development Postgres..."
	@echo "Database: designstreamdb_dev"
	@echo "User: designstream"
	@echo "Password: designstream_password"
	@echo ""
	PGPASSWORD=designstream_password psql -h localhost -p 5433 -U designstream -d designstreamdb_dev

migrate: ## Run database migrations
	@echo "🔄 Running database migrations..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	. venv/bin/activate && python -c " \
import os; \
from sqlalchemy import create_engine, text; \
from dotenv import load_dotenv; \
load_dotenv(); \
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://designstream:designstream_password@localhost:5432/designstreamdb'); \
try: \
    engine = create_engine(DATABASE_URL); \
    with engine.connect() as conn: \
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;')); \
        conn.commit(); \
        print('✅ pgvector extension enabled'); \
        migration_files = ['migrations/001_initial_schema.sql', 'migrations/002_add_indexes.sql', 'migrations/003_matting_studio_admin.sql']; \
        for migration in migration_files: \
            if os.path.exists(migration): \
                with open(migration, 'r') as f: \
                    sql = f.read(); \
                    conn.execute(text(sql)); \
                    conn.commit(); \
                    print(f'✅ Applied migration: {migration}'); \
        print('✅ All migrations completed successfully'); \
except Exception as e: \
    print(f'❌ Migration failed: {e}'); \
    exit(1); \
"

migrate-dev: ## Run migrations on development database
	@echo "🔄 Running migrations on development database..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	. venv/bin/activate && python -c " \
import os; \
from sqlalchemy import create_engine, text; \
from dotenv import load_dotenv; \
load_dotenv(); \
DATABASE_URL = os.getenv('DATABASE_URL_DEV', 'postgresql+psycopg2://designstream:designstream_password@localhost:5433/designstreamdb_dev'); \
try: \
    engine = create_engine(DATABASE_URL); \
    with engine.connect() as conn: \
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;')); \
        conn.commit(); \
        print('✅ pgvector extension enabled'); \
        migration_files = ['migrations/001_initial_schema.sql', 'migrations/002_add_indexes.sql', 'migrations/003_matting_studio_admin.sql']; \
        for migration in migration_files: \
            if os.path.exists(migration): \
                with open(migration, 'r') as f: \
                    sql = f.read(); \
                    conn.execute(text(sql)); \
                    conn.commit(); \
                    print(f'✅ Applied migration: {migration}'); \
        print('✅ All migrations completed successfully'); \
except Exception as e: \
    print(f'❌ Migration failed: {e}'); \
    exit(1); \
"

seed: ## Seed database with sample data
	@echo "🌱 Seeding database with sample data..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	. venv/bin/activate && python -c " \
import os; \
from sqlalchemy import create_engine, text; \
from dotenv import load_dotenv; \
load_dotenv(); \
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://designstream:designstream_password@localhost:5432/designstreamdb'); \
try: \
    engine = create_engine(DATABASE_URL); \
    with engine.connect() as conn: \
        seed_files = ['migrations/seeds/001_sample_products.sql', 'migrations/seeds/002_sample_variants.sql']; \
        for seed in seed_files: \
            if os.path.exists(seed): \
                with open(seed, 'r') as f: \
                    sql = f.read(); \
                    conn.execute(text(sql)); \
                    conn.commit(); \
                    print(f'✅ Applied seed: {seed}'); \
        print('✅ Database seeded successfully'); \
except Exception as e: \
    print(f'❌ Seeding failed: {e}'); \
    exit(1); \
"

# Application
dev: ## Start development server
	@echo "🚀 Starting development server..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	. venv/bin/activate && python run.py

test: ## Run tests
	@echo "🧪 Running tests..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	. venv/bin/activate && python -m pytest tests/ -v

# MinIO operations
minio-setup: ## Setup MinIO buckets
	@echo "🗄️  Setting up MinIO buckets..."
	@docker-compose exec minio sh -c "
		mc alias set local http://localhost:9000 minioadmin minioadmin123
		mc mb local/designstream-uploads --ignore-existing
		mc mb local/designstream-renders --ignore-existing
		mc mb local/designstream-ci --ignore-existing
		mc policy set public local/designstream-uploads
		mc policy set public local/designstream-renders
		mc policy set public local/designstream-ci
	"
	@echo "✅ MinIO buckets created and configured"

minio-logs: ## Show MinIO logs
	docker-compose logs -f minio

# Cleanup
clean: ## Clean up containers and volumes
	docker-compose down -v
	docker system prune -f
	@echo "✅ Cleanup completed"

clean-data: ## Clean up only data volumes (keep containers)
	docker-compose down -v
	@echo "✅ Data volumes cleaned"

# Full setup
setup: install up migrate seed minio-setup ## Complete development setup
	@echo "🎉 Development environment ready!"
	@echo "Run 'make dev' to start the application"

# CI/CD helpers
ci-up: ## Start services for CI
	docker-compose up -d postgres minio redis
	@sleep 15

ci-migrate: ## Run migrations for CI
	@echo "🔄 Running CI migrations..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	. venv/bin/activate && python -c " \
import os; \
from sqlalchemy import create_engine, text; \
from dotenv import load_dotenv; \
load_dotenv(); \
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://designstream:designstream_password@localhost:5432/designstreamdb'); \
try: \
    engine = create_engine(DATABASE_URL); \
    with engine.connect() as conn: \
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;')); \
        conn.commit(); \
        migration_files = ['migrations/001_initial_schema.sql', 'migrations/002_add_indexes.sql', 'migrations/003_matting_studio_admin.sql']; \
        for migration in migration_files: \
            if os.path.exists(migration): \
                with open(migration, 'r') as f: \
                    sql = f.read(); \
                    conn.execute(text(sql)); \
                    conn.commit(); \
        print('✅ CI migrations completed'); \
except Exception as e: \
    print(f'❌ CI migration failed: {e}'); \
    exit(1); \
"

ci-setup-minio: ## Setup MinIO for CI
	@echo "🗄️  Setting up MinIO for CI..."
	@docker-compose exec minio sh -c "
		mc alias set local http://localhost:9000 minioadmin minioadmin123
		mc mb local/designstream-ci --ignore-existing
		mc policy set public local/designstream-ci
	"
	@echo "✅ CI MinIO setup completed"
