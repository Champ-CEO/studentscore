# AIAP Student Score Prediction Project Brief

## Project Overview

This project is part of the AI Singapore (AISG) Technical Assessment for building AI solutions to solve real-world problem statements. Our client, U.A Secondary School, requires a predictive model to identify students who may struggle with their O-level mathematics examination, enabling timely intervention and support.

## Problem Statement

U.A Secondary School needs to predict students' O-level mathematics examination scores to identify weaker students before the examination. This early identification will allow the school to provide additional support and ensure students are better prepared for the upcoming test.

## Project Objectives

1. **Exploratory Data Analysis (EDA)**: Conduct comprehensive analysis of student performance data
2. **Machine Learning Pipeline**: Build an end-to-end ML pipeline for score prediction

## Dataset Information

### Data Source
- **Database**: SQLite database (`score.db`)
- **Access URL**: `https://techassessment.blob.core.windows.net/aiap-preparatory-bootcamp/score.db`
- **Location**: Must be placed in `data/score.db` (relative path)

### Dataset Attributes

| Attribute | Description |
|-----------|-------------|
| `student_id` | Unique ID for each student |
| `number_of_siblings` | Number of siblings |
| `direct_admission` | Mode of entering the school |
| `CCA` | Enrolled Co-Curricular Activity |
| `learning_style` | Primary learning style |
| `tuition` | Indication of whether the student has tuition |
| `final_test` | **Target Variable**: Student's O-level mathematics examination score |
| `n_male` | Number of male classmates |
| `n_female` | Number of female classmates |
| `gender` | Gender type |
| `age` | Age of the student |
| `hours_per_week` | Number of hours student studies per week |
| `attendance_rate` | Attendance rate of the student (%) |
| `sleep_time` | Daily sleeping time (hour:minutes) |
| `wake_time` | Daily waking up time (hour:minutes) |
| `mode_of_transport` | Mode of transport to school |
| `bag_color` | Colour of student's bag |

## Task 1: Exploratory Data Analysis (EDA)

### Deliverable
- **File**: `eda.ipynb` (Jupyter Notebook)

### Requirements
- Interactive Python notebook suitable for presentation
- Appropriate visualizations and statistical techniques
- Clear explanations of:
  - Steps taken in the EDA process
  - Purpose of each step
  - Conclusions drawn from each step
  - Interpretation of statistics and their impact on analysis
- Clear, meaningful, and understandable visualizations
- Well-organized and easy-to-understand structure

### Key Areas to Explore
- Data quality assessment (missing values, outliers, data types)
- Target variable distribution and characteristics
- Feature distributions and relationships
- Correlation analysis between features and target
- Categorical variable analysis
- Time-based patterns (sleep/wake times)
- Class balance and potential bias identification

## Task 2: End-to-End Machine Learning Pipeline

### Deliverables
1. **Source Code**: `src/` folder containing Python modules (`.py` files)
2. **Dependencies**: `requirements.txt` file
3. **Documentation**: `README.md` file

### Pipeline Requirements

#### Technical Requirements
- **Data Access**: Must use SQLite or SQLAlchemy to fetch data from `data/score.db`
- **Configuration**: Easily configurable for experimentation (config files, environment variables, or CLI parameters)
- **Code Quality**: Reusable, readable, and self-explanatory Python scripts
- **Modularity**: Well-structured Python modules/classes

#### Machine Learning Requirements
- **Preprocessing**: Appropriate data preprocessing and feature engineering
- **Model Selection**: Implement and optimize at least 3 models from the approved list
- **Evaluation**: Appropriate evaluation metrics with clear explanations
- **Documentation**: Clear explanations for model and metric choices

### Approved Machine Learning Models

The following models are available for selection (minimum 3 required):

- **K-means Clustering**
- **Linear Regression**
- **Logistic Regression**
- **Decision Trees**
- **Random Forest**
- **Support Vector Machines (SVM)**
- **Neural Networks**
- **Gradient Boosting / XGBoost**
- **Ensemble Methods**
- **Naive Bayes**

### README.md Requirements

The README must include:

1. **Personal Information**
   - Full name (as in NRIC)
   - Email address (from application form)

2. **Project Structure**
   - Overview of submitted folder structure
   - File organization explanation

3. **Usage Instructions**
   - Pipeline execution instructions
   - Parameter modification guidelines

4. **Pipeline Architecture**
   - Logical steps/flow description
   - Flow charts or visualization aids (optional but recommended)

5. **EDA Summary**
   - Key findings from exploratory analysis
   - Feature engineering decisions based on EDA insights

6. **Feature Processing**
   - Summary table of how each feature is processed
   - Transformation and encoding methods

7. **Model Selection**
   - Explanation of chosen models
   - Rationale for model selection

8. **Evaluation Framework**
   - Model evaluation methodology
   - Explanation of chosen metrics
   - Performance comparison results

## Project Structure

```
project-root/
├── README.md
├── requirements.txt
├── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   └── [specific_model_implementations].py
│   ├── evaluation.py
│   ├── pipeline.py
│   └── config.py
└── data/
    └── score.db (not submitted)
```

## Success Criteria

1. **EDA Quality**: Comprehensive analysis with actionable insights
2. **Pipeline Functionality**: Working end-to-end ML pipeline
3. **Model Performance**: Well-performing models with proper evaluation
4. **Code Quality**: Clean, modular, and well-documented code
5. **Documentation**: Clear and comprehensive README
6. **Reproducibility**: Easy to execute and modify pipeline

## Important Notes

- **Database File**: DO NOT submit the `score.db` file in final submission
- **Code Format**: Use `.py` files for the ML pipeline, NOT Jupyter notebooks
- **Data Access**: Pipeline must fetch data using SQLite/SQLAlchemy from relative path `data/score.db`
- **Version**: This is Version 1.0, updated 09.08.23

## Next Steps

1. Download and explore the database
2. Set up development environment
3. Begin EDA in Jupyter notebook
4. Design pipeline architecture
5. Implement and test ML models
6. Document findings and methodology
7. Prepare final submission package