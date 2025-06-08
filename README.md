# Student Score Prediction System

## Project Overview
AI-powered system to predict student academic performance using machine learning. This project implements a comprehensive data science pipeline from raw data processing to feature engineering, following a phased development approach.

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run comprehensive tests
python -m pytest tests/

# (Optional) Run specific test suites if needed, e.g.:
# python -m pytest tests/test_phase4_implementations.py
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
│   ├── data/               # Data processing modules
│   └── modeling/           # Modeling scripts (Phase 5)
├── tests/                  # Test suites
│   └── data/               # Data-specific tests
├── specs/                  # Project documentation
│   ├── ai-doc/             # AI documentation
│   └── reports/            # Phase reports
├── logs/                   # Execution logs
└── requirements.txt        # Project dependencies
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

### ✅ Phase 4: Feature Engineering (Completed)
- Derived feature creation (Study Efficiency Score, Academic Support Index)
- Interaction feature generation (Study × Attendance, Sleep × Study Hours)
- Advanced preprocessing techniques (scaling, encoding, outlier handling)
- Feature selection and dimensionality reduction
- Comprehensive data quality validation

### ✅ Phase 5: Model Development (Completed)
- Implemented multiple ML algorithms: Linear Regression, Ridge, Random Forest, XGBoost, Neural Network.
- Addressed and fixed critical data leakage issues, ensuring reliable model evaluation.
- Conducted comprehensive model training, cross-validation, and performance evaluation.
- Selected the best performing model based on MAE, R², and overfitting analysis.
- Generated realistic performance metrics and learning curves for all models.
- Persisted trained models and detailed results.

### 📋 Phase 6: Testing, Validation & Refinement (In Progress)
- **Critical**: Investigate near-perfect Linear Regression performance to ensure validity.
- **Critical**: Implement robust model interpretability (e.g., SHAP, Permutation Importance).
- Conduct comprehensive data validation on the final modeling dataset.
- Perform external validation and robustness analysis on selected models.
- Finalize model documentation and prepare for potential deployment scenarios.

## Development Setup
1. Clone the repository
2. Set up Python 3.9+ virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests to verify setup: `python -m pytest tests/`

## Key Data Artifacts

### Modeling Datasets & Outputs (Phase 5)
- `data/modeling_outputs/clean_dataset_no_leakage.csv` - Final model-ready dataset after data leakage fix.
- `data/modeling_outputs/phase5_complete_fixed_results.json` - Comprehensive results for all implemented models (MAE, RMSE, R², overfitting metrics).
- `data/modeling_outputs/best_model_selection_fixed.json` - Details of the selected best model.
- `data/modeling_outputs/best_model_linear_regression_fixed.joblib` - Saved best performing model (Linear Regression).
- `data/modeling_outputs/models/` - Directory containing all persisted trained model files.
- `data/modeling_outputs/overfitting_plots/` - Directory containing learning curve plots for all models.

### Processed Data (Phases 3 & 4)
- `data/processed/final_processed.csv` - Dataset after initial preprocessing and cleaning (pre-leakage fix).
- `data/processed/age_corrected.csv` - Age-corrected data.
- `data/processed/standardized.csv` - Standardized categorical data.
- `data/processed/imputed.csv` - Imputed attendance data.

### Feature Engineering (Phase 4 Complete)
- `data/featured/derived_features.csv` - Engineered features (Study Efficiency Score, Academic Support Index)
- `data/featured/interaction_features.csv` - Interaction features (Study × Attendance, etc.)
- `data/featured/feature_definitions.json` - Complete feature documentation
- `data/featured/interaction_definitions.json` - Interaction feature specifications
- Feature audit trails and validation logs in JSON format

### EDA Visualizations
- `notebook/visualization/correlation_heatmap.png`
- `notebook/visualization/numerical_features_boxplots.png`
- `notebook/visualization/target_variable_analysis.png`

## Testing
- Comprehensive tests: `python -m pytest tests/`
- Phase-specific tests (example for Phase 4):
  `python -m pytest tests/test_phase4_implementations.py`
- Data-specific tests:
  `python -m pytest tests/data/`

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
