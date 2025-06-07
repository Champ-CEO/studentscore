# Student Score Prediction System

## Project Overview
AI-powered system to predict student academic performance using machine learning. This project implements a comprehensive data science pipeline from raw data processing to feature engineering, following a phased development approach.

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run Phase 3 tests
python test_phase3_simple.py

# Or run comprehensive tests
python -m pytest tests/
```

## Project Structure
```
studentscore/
├── data/                    # Data files
│   ├── raw/                # Raw SQLite database (score.db)
│   ├── processed/          # Cleaned and processed data
│   └── featured/           # Feature engineered datasets
├── notebook/               # EDA and visualization
│   └── visualization/      # Generated plots and charts
├── src/                    # Source code
│   └── data/              # Data processing modules
├── tests/                  # Test suites
│   └── data/              # Data-specific tests
├── specs/                  # Project documentation
│   ├── ai-doc/            # AI documentation
│   └── reports/           # Phase reports
└── test_phase3_simple.py   # Standalone test script
```

## Database Schema
- SQLite database with 15,900 student records
- 17 features including demographics, academic metrics, study habits
- Target variable: `final_test` score (0-100)

## Current Implementation Status

### ✅ Phase 1: Data Collection and Setup
- SQLite database setup and structure analysis
- Initial data exploration and validation

### ✅ Phase 2: Exploratory Data Analysis (EDA)
- Comprehensive statistical analysis
- Correlation analysis and feature relationships
- Data quality assessment and visualization
- Missing data pattern analysis

### ✅ Phase 3: Data Preprocessing and Cleaning
- Age data correction (negative values fixed)
- Categorical data standardization
- Missing data imputation (attendance rate, final test)
- Outlier detection and handling
- Feature engineering foundation
- Comprehensive data validation pipeline

### 🔄 Phase 4: Feature Engineering (In Progress)
- Derived feature creation
- Interaction feature generation
- Advanced preprocessing techniques

### 📋 Phase 5: Model Development (Planned)
- Multiple ML algorithm implementation
- Model training and validation
- Performance optimization

## Development Setup
1. Clone the repository
2. Set up Python 3.9+ virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests to verify setup: `python test_phase3_simple.py`

## Key Data Artifacts

### Processed Data
- `data/processed/final_processed.csv` - Model-ready dataset
- `data/processed/age_corrected.csv` - Age-corrected data
- `data/processed/standardized.csv` - Standardized categorical data
- `data/processed/imputed.csv` - Imputed attendance data

### Feature Engineering
- `data/featured/derived_features.csv` - Initial derived features
- `data/featured/interaction_features.csv` - Interaction features
- Feature definitions and audit trails in JSON format

### EDA Visualizations
- `notebook/visualization/correlation_heatmap.png`
- `notebook/visualization/numerical_features_boxplots.png`
- `notebook/visualization/target_variable_analysis.png`

## Testing
- Simple tests: `python test_phase3_simple.py`
- Comprehensive tests: `python -m pytest tests/`
- Phase 3 specific tests: `python -m pytest tests/test_phase3_implementations.py`
- Data quality tests: `python -m pytest tests/data/`

## Data Quality Metrics
- **Data Completeness**: 100% (after Phase 3 preprocessing)
- **Age Validity**: 100% (corrected 5 negative values)
- **Categorical Consistency**: 100% (standardized format)
- **Missing Data**: 0% (imputed using EDA-informed strategies)
- **Overall Quality Score**: 80.25%

## Documentation
- Phase reports available in `specs/reports/`
- Task tracking in `specs/TASKS.md`
- Project documentation in `specs/ai-doc/`

## Contributing
1. Follow the phased development approach
2. Write tests for new implementations
3. Update phase reports and documentation
4. Maintain audit trails for data processing

## License
MIT License
