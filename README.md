# Student Score Prediction System

## Project Overview
AI-powered system to predict student academic performance and provide early intervention recommendations.

## Quick Start
```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Start API server
poetry run uvicorn src.api.main:app --reload
```

## Project Structure
```
studentscore/
├── data/               # Data files
│   ├── raw/           # Raw SQLite database
│   ├── processed/     # Cleaned data
│   └── feature/       # Engineered features
├── models/            # Saved ML models
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
│   ├── data/         # Data processing
│   ├── features/     # Feature engineering
│   ├── models/       # ML models
│   └── api/          # FastAPI application
├── tests/            # Test suites
└── specs/            # Documentation
```

## Database Schema
- SQLite database with 15,900 student records
- 17 features including demographics, academic metrics
- Target variable: `final_test` score

## Key Features
- Data cleaning and preprocessing pipeline
- Feature engineering optimized for academic prediction
- Multiple ML models (Random Forest, XGBoost, etc.)
- REST API for predictions and model management
- Monitoring and performance tracking

## Development Setup
1. Install Poetry for dependency management
2. Set up Python 3.9+ virtual environment
3. Install project dependencies
4. Run test suite to verify setup

## API Documentation
Access OpenAPI documentation at: http://localhost:8000/docs

## Testing
- Unit tests: `poetry run pytest tests/unit`
- Integration tests: `poetry run pytest tests/integration`
- Coverage report: `poetry run pytest --cov`

## Contributing
1. Follow PEP 8 style guide
2. Write tests for new features
3. Update documentation
4. Submit pull request

## License
MIT License

## Contact
Project Maintainer: [Your Name]
