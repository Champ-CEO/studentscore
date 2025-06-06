# Enhanced Student ID Analysis Task (2.1.2a)

**Objective**: Comprehensive analysis and feature extraction from student_id before potential removal

## Sub-tasks:

### 2.1.2a.1 ID Structure Analysis
- **Tests Required**:
  - Test student_id uniqueness (expect 15,900 unique values)
  - Test ID format consistency and pattern recognition
  - Test identification of embedded information patterns
- **Implementation**:
  - Check for duplicate student_id values
  - Analyze ID structure (length, character patterns, prefixes/suffixes)
  - Identify embedded information (enrollment year, school code, cohort indicators)

### 2.1.2a.2 Feature Extraction from ID
- **Tests Required**:
  - Test extraction of meaningful features from ID structure
  - Test correlation of derived features with target variable
  - Test predictive value of ID-derived features
- **Implementation**:
  - Extract enrollment year/semester if embedded in ID
  - Create school/campus indicator if ID contains location codes
  - Generate student cohort features if discernible from ID pattern
  - Create "ID complexity" feature (character variety, length-based features)

### 2.1.2a.3 ID Retention Decision
- **Tests Required**:
  - Test decision framework for ID retention vs. removal
  - Test that extracted features capture ID information adequately
  - Test that original ID adds no additional predictive value after feature extraction
- **Implementation**:
  - Evaluate predictive value: original ID vs. derived features
  - **Decision Rule**: Drop original student_id if derived features capture all useful information
  - Document extraction process and retain mapping for inference pipeline


# Comprehensive Outlier Handling Task (2.1.7)

**Objective**: Systematic identification and handling of outliers across all numerical features

## Sub-tasks:

### 2.1.7.1 Outlier Detection
- **Tests Required**:
  - Test IQR method identifies outliers correctly
  - Test Z-score method (>3 standard deviations) identifies outliers
  - Test domain-specific outlier rules (e.g., age < 0, age > 100)
  - Test outlier detection across all numerical features
- **Implementation**:
  - Apply IQR method: values < Q1 - 1.5*IQR or > Q3 + 1.5*IQR
  - Apply Z-score method: |z-score| > 3
  - Domain validation: age, attendance_rate, hours_per_week, final_test
  - Generate outlier summary report

### 2.1.7.2 Outlier Analysis and Decision
- **Tests Required**:
  - Test outlier pattern analysis for systematic issues
  - Test impact assessment of outlier handling strategies
  - Test domain expert validation of outlier treatment decisions
- **Implementation**:
  - Analyze outlier patterns: are they errors or legitimate extreme values?
  - **Age outliers**: Negative ages = clear errors (remove/correct)
  - **Attendance outliers**: >100% may indicate data entry errors
  - **Score outliers**: Very high/low scores may be legitimate
  - Impact assessment: compare distributions before/after treatment

### 2.1.7.3 Outlier Treatment Application
- **Tests Required**:
  - Test removal strategy preserves data integrity
  - Test transformation reduces outlier impact appropriately
  - Test capping/winsorization maintains data distribution shape
  - Test chosen strategy improves model performance
- **Implementation**:
  - **Removal**: Clear data entry errors (negative ages, impossible values)
  - **Transformation**: Log transformation for right-skewed variables
  - **Capping**: 95th/5th percentile capping for extreme but potentially valid values
  - **Robust Models**: Document outliers for robust model consideration
  - Document all outlier treatment decisions with justification



# Systematic Imbalanced Data Analysis Task (2.1.8)

**Objective**: Identify and address imbalanced data issues across target and key features

## Sub-tasks:

### 2.1.8.1 Imbalance Detection
- **Tests Required**:
  - Test target variable (final_test) distribution analysis
  - Test categorical feature imbalance identification
  - Test imbalance severity measurement (ratios, entropy)
  - Test impact assessment on model performance
- **Implementation**:
  - Analyze final_test score distribution (continuous target)
  - Create score bins/categories to check for imbalanced ranges
  - Examine categorical features: gender, CCA, learning_style, transport mode
  - Calculate imbalance ratios and statistical measures

### 2.1.8.2 Imbalance Impact Assessment
- **Tests Required**:
  - Test baseline model performance on imbalanced data
  - Test evaluation metrics appropriate for imbalanced scenarios
  - Test identification of affected model performance areas
- **Implementation**:
  - Train simple baseline model on raw data
  - Use appropriate metrics: F1-score, precision-recall curves, AUC-PR
  - Identify if model exhibits bias toward majority classes
  - Document performance issues attributable to imbalance

### 2.1.8.3 Imbalance Treatment (If Required)
- **Tests Required**:
  - Test SMOTE/ADASYN oversampling effectiveness
  - Test undersampling impact on information loss
  - Test class weight adjustment in models
  - Test balanced accuracy vs. original accuracy
- **Implementation**:
  - **For Regression**: Consider stratified sampling by score ranges
  - **For Classification** (if binning scores): Apply SMOTE, ADASYN, or random oversampling
  - **Alternative**: Use class_weight='balanced' in model parameters
  - **Evaluation**: Focus on balanced accuracy, F1-score, AUC-PR rather than just accuracy
  - Document chosen approach and performance comparison