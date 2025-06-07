# Student Score Prediction Project - Task Management Document

## Purpose

This document tracks current tasks, backlog, sub-tasks, and discoveries made during development of the Student Score Prediction system. It follows Test-Driven Development (TDD) principles where tests are written before implementation, ensuring robust and reliable code delivery.

**Key Principles:**
- All major functionality must have tests written BEFORE implementation
- Each task includes specific test requirements and success criteria
- Database usage and data flow are clearly specified at each stage
- Tasks are organized by project phases with clear dependencies
- Status tracking enables progress monitoring and bottleneck identification

## Task Status Legend

- 🔴 **Not Started**: Task not yet begun
- 🟡 **In Progress**: Currently being worked on
- 🟢 **Completed**: Task finished and tested
- 🔵 **Blocked**: Waiting on dependencies or external factors
- ⚪ **Discovered**: New task identified during development

## Project Phases Overview

### Phase 1: Project Setup & Infrastructure (🟢 Completed)
### Phase 2: Exploratory Data Analysis (EDA) (🟢 Completed)
### Phase 3: Data Preprocessing and Cleaning (🟢 Completed)
### Phase 4: Feature Engineering (🟢 Completed)
### Phase 5: Model Development (🔴 Not Started)
### Phase 6: Testing & Validation (🔴 Not Started)

---

## Phase 1: Project Setup & Infrastructure

### 1.1 Environment Setup (🟢 Completed)

**Objective**: Establish development environment and project structure

#### Sub-tasks:
- **1.1.1** Initialize Python project with uv (🟢 Completed)
  - **Tests Required**: 
    - Test uv.lock file exists and is valid
    - Test all dependencies from requirements.txt install correctly using uv
    - Test virtual environment activation
  - **Implementation**: Create pyproject.toml and requirements.txt with core dependencies, manage with uv
  - **Database Usage**: N/A
  - **Dependencies**: None

- **1.1.2** Setup project directory structure (🟢 Completed)
  - **Tests Required**:
    - Test all required directories exist
    - Test directory permissions are correct
    - Test .gitignore excludes appropriate files
  - **Implementation**: Create standardized ML project structure
  - **Database Usage**: N/A
  - **Dependencies**: 1.1.1

- **1.1.3** Configure development tools (🟢 Completed)
  - **Tests Required**:
    - Test black formatting runs without errors
    - Test flake8 linting passes
    - Test pytest discovers and runs tests
  - **Implementation**: Setup black, flake8, pytest configurations
  - **Database Usage**: N/A
  - **Dependencies**: 1.1.1

### 1.2 Database Setup (🟢 Completed)

**Objective**: Establish database connection and verify data integrity

#### Sub-tasks:
- **1.2.1** Create database connection module (🟢 Completed)
  - **Tests Required**:
    - Test SQLite connection establishment
    - Test connection pooling functionality
    - Test connection error handling
    - Test database file exists and is readable
  - **Implementation**: Direct sqlite3 connection (example in `data/raw/test_sqlite.py`)
  - **Database Usage**: SQLite `score.db` as primary data source
  - **Dependencies**: 1.1.1

- **1.2.2** Implement data access layer (🟢 Completed)
  - **Tests Required**:
    - Test table schema validation
    - Test basic CRUD operations
    - Test query parameter sanitization
    - Test transaction handling
  - **Implementation**: Repository pattern with `sqlite3`
  - **Database Usage**: Read operations on raw student data
  - **Dependencies**: 1.2.1

- **1.2.3** Verify data integrity and structure (🟢 Completed)
  - **Tests Required**:
    - Test expected 15,900 records exist
    - Test all 17 features are present
    - Test data types match expectations
    - Test primary key constraints
  - **Implementation**: Data validation scripts (`src/data/validation.py`)
  - **Database Usage**: Full table scan for integrity checks
  - **Dependencies**: 1.2.2

---

## Phase 2: Exploratory Data Analysis (EDA)

**Objective**: Conduct comprehensive exploratory data analysis to understand data patterns, relationships, and inform preprocessing decisions

### 2.1 Data Loading and Initial Exploration (🟢 Completed)

**Objective**: Load data and perform initial exploration

#### Sub-tasks:
- **2.1.1** Setup EDA environment and data loading (🟢 Completed)
  - **Tests Required**:
    - Test data loads correctly from SQLite database
    - Test all 17 features are present and accessible
    - Test expected 15,900 records are loaded
    - Test data types are correctly inferred
  - **Implementation**: 
    - Create comprehensive EDA notebook/script (`notebook/eda.py`)
    - Load data using established database connection
    - Initial data shape, info, and basic statistics
  - **Database Usage**: 
    - **Source**: Raw data from SQLite `score.db`
    - **Output**: In-memory DataFrame for analysis
  - **Dependencies**: 1.2.3

- **2.1.2** Data overview and basic statistics (🟢 Completed)
  - **Tests Required**:
    - Test summary statistics are calculated for all numerical features
    - Test categorical feature value counts are generated
    - Test data types and memory usage are documented
    - Test missing value counts are accurate
  - **Implementation**:
    - Generate comprehensive data summary
    - Calculate descriptive statistics (mean, median, std, quartiles)
    - Document data types and memory usage
    - Create initial missing value analysis
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.1.1

### 2.2 Missing Data Analysis (🟢 Completed)

**Objective**: Comprehensive analysis of missing data patterns

#### Sub-tasks:
- **2.2.1** Missing data pattern analysis (🟢 Completed)
  - **Tests Required**:
    - Test missing value heatmap is generated correctly
    - Test missing data patterns are identified (MCAR, MAR, MNAR)
    - Test correlation between missing values across features
    - Test missing data impact on target variable
  - **Implementation**:
    - Create missing value heatmap visualization
    - Analyze missing data patterns and mechanisms
    - Identify features with high missing rates
    - Assess impact of missingness on target variable
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.1.2

- **2.2.2** Missing data visualization (🟢 Completed)
  - **Tests Required**:
    - Test missing value bar chart shows correct percentages
    - Test missing value matrix visualization is clear
    - Test missing data correlation plot is interpretable
  - **Implementation**:
    - Bar chart of missing percentages by feature
    - Missing value matrix visualization
    - Correlation plot of missing indicators
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.2.1

### 2.3 Univariate Analysis (🟢 Completed)

**Objective**: Analyze individual feature distributions and characteristics

#### Sub-tasks:
- **2.3.1** Numerical feature analysis (🟢 Completed)
  - **Tests Required**:
    - Test histograms are generated for all numerical features
    - Test box plots identify outliers correctly
    - Test distribution normality tests are performed
    - Test skewness and kurtosis are calculated
  - **Implementation**:
    - Histograms with density curves for numerical features
    - Box plots for outlier detection
    - Q-Q plots for normality assessment
    - Statistical tests for distribution shape
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.1.2

- **2.3.2** Categorical feature analysis (🟢 Completed)
  - **Tests Required**:
    - Test bar charts show correct category frequencies
    - Test pie charts display proportions accurately
    - Test category imbalance is identified
    - Test rare categories are flagged
  - **Implementation**:
    - Bar charts for categorical feature distributions
    - Pie charts for proportion visualization
    - Identify imbalanced categories
    - Flag rare categories for potential grouping
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.1.2

- **2.3.3** Target variable analysis (🟢 Completed)
  - **Tests Required**:
    - Test target variable distribution is visualized
    - Test target variable statistics are calculated
    - Test target variable outliers are identified
    - Test target variable transformation needs are assessed
  - **Implementation**:
    - Histogram and density plot of final_test scores
    - Box plot for outlier identification
    - Statistical summary of target variable
    - Assess need for transformation (log, sqrt, etc.)
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.1.2

### 2.4 Bivariate Analysis (🟢 Completed)

**Objective**: Analyze relationships between features and with target variable

#### Sub-tasks:
- **2.4.1** Feature correlation analysis (🟢 Completed)
  - **Tests Required**:
    - Test correlation matrix is calculated correctly
    - Test correlation heatmap is visually clear
    - Test high correlations (>0.7) are identified
    - Test correlation with target variable is analyzed
  - **Implementation**:
    - Calculate Pearson correlation matrix
    - Create correlation heatmap with annotations
    - Identify highly correlated feature pairs
    - Analyze feature-target correlations
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.3.3

- **2.4.2** Numerical vs target analysis (🟢 Completed)
  - **Tests Required**:
    - Test scatter plots show feature-target relationships
    - Test regression lines indicate relationship strength
    - Test non-linear relationships are identified
    - Test outliers affecting relationships are flagged
  - **Implementation**:
    - Scatter plots of numerical features vs target
    - Add regression lines and confidence intervals
    - Identify non-linear relationships
    - Flag influential outliers
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.4.1

- **2.4.3** Categorical vs target analysis (🟢 Completed)
  - **Tests Required**:
    - Test box plots show target distribution by category
    - Test violin plots reveal distribution shapes
    - Test statistical significance tests are performed
    - Test effect sizes are calculated
  - **Implementation**:
    - Box plots of target by categorical features
    - Violin plots for distribution shape comparison
    - ANOVA tests for group differences
    - Calculate effect sizes (Cohen's d, eta-squared)
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.4.1

### 2.5 Multivariate Analysis (🟢 Completed)

**Objective**: Analyze complex relationships between multiple features

#### Sub-tasks:
- **2.5.1** Feature interaction analysis (🟢 Completed)
  - **Tests Required**:
    - Test interaction plots reveal meaningful patterns
    - Test three-way relationships are explored
    - Test interaction effects on target are quantified
    - Test significant interactions are documented
  - **Implementation**:
    - Create interaction plots for key feature pairs
    - Explore three-way relationships with target
    - Statistical tests for interaction effects
    - Document significant interactions for feature engineering
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.4.3

- **2.5.2** Dimensionality reduction exploration (🟢 Completed)
  - **Tests Required**:
    - Test PCA explains variance correctly
    - Test t-SNE reveals cluster patterns
    - Test UMAP preserves local structure
    - Test dimensionality reduction plots are interpretable
  - **Implementation**:
    - Principal Component Analysis (PCA)
    - t-SNE for non-linear dimensionality reduction
    - UMAP for structure preservation
    - Visualize high-dimensional data in 2D/3D
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.5.1

### 2.6 Data Quality Assessment (🟢 Completed)

**Objective**: Identify data quality issues and anomalies

#### Sub-tasks:
- **2.6.1** Outlier detection and analysis (🟢 Completed)
  - **Tests Required**:
    - Test multiple outlier detection methods are applied
    - Test outliers are visualized clearly
    - Test outlier impact on target is assessed
    - Test outlier treatment recommendations are provided
  - **Implementation**:
    - Apply IQR, Z-score, and Isolation Forest methods
    - Create outlier visualization plots
    - Analyze outlier characteristics and patterns
    - Provide recommendations for outlier treatment
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.3.1

- **2.6.2** Data consistency checks (🟢 Completed)
  - **Tests Required**:
    - Test logical consistency rules are validated
    - Test impossible value combinations are identified
    - Test data entry errors are flagged
    - Test consistency issues are documented
  - **Implementation**:
    - Check logical relationships (e.g., sleep_time < wake_time)
    - Identify impossible value combinations
    - Flag potential data entry errors
    - Document all consistency issues found
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.6.1

- **2.6.3** Duplicate analysis (🟢 Completed)
  - **Tests Required**:
    - Test duplicate records are identified correctly
    - Test duplicate patterns are analyzed
    - Test duplicate impact is assessed
    - Test duplicate removal strategy is recommended
  - **Implementation**:
    - Identify exact and near-duplicate records
    - Analyze patterns in duplicate data
    - Assess impact of duplicates on analysis
    - Recommend duplicate handling strategy
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.6.2

### 2.7 Feature-Specific Deep Dive (🟢 Completed)

**Objective**: Detailed analysis of key features identified during exploration

#### Sub-tasks:
- **2.7.1** Student ID pattern analysis (🟢 Completed)
  - **Tests Required**:
    - Test ID format patterns are identified
    - Test embedded information is extracted
    - Test ID uniqueness is verified
    - Test ID-derived features are proposed
  - **Implementation**:
    - Analyze student_id structure and patterns
    - Extract embedded information (year, school, etc.)
    - Verify uniqueness and format consistency
    - Propose features derivable from ID structure
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.6.3

- **2.7.2** Age distribution analysis (🟢 Completed)
  - **Tests Required**:
    - Test age distribution is thoroughly analyzed
    - Test age outliers and anomalies are identified
    - Test age-related patterns are discovered
    - Test age feature engineering opportunities are identified
  - **Implementation**:
    - Detailed age distribution analysis
    - Identify negative ages and other anomalies
    - Explore age-performance relationships
    - Recommend age-based feature engineering
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.7.1

- **2.7.3** Sleep pattern analysis (🟢 Completed)
  - **Tests Required**:
    - Test sleep duration calculation is accurate
    - Test sleep pattern categories are meaningful
    - Test sleep-performance relationships are explored
    - Test sleep feature engineering is proposed
  - **Implementation**:
    - Calculate sleep duration from sleep/wake times
    - Categorize sleep patterns (early/late sleepers)
    - Analyze relationship between sleep and performance
    - Propose sleep-related feature engineering
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.7.2

- **2.7.4** Temporal pattern analysis (🟢 Completed)
  - **Tests Required**:
    - Test daily activity patterns are identified
    - Test weekly study patterns are analyzed
    - Test seasonal variations are detected
    - Test time-based feature engineering is proposed
  - **Implementation**:
    - Analyze daily activity distributions
    - Identify weekly study patterns and trends
    - Examine seasonal effects on performance
    - Create time-based derived features
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.7.3

- **2.7.5** Transportation and access analysis (🟢 Completed)
  - **Tests Required**:
    - Test commute patterns are identified
    - Test transportation mode distributions are analyzed
    - Test access time impact is assessed
    - Test location-based patterns are discovered
  - **Implementation**:
    - Analyze commute time distributions
    - Study transportation mode preferences
    - Assess impact of access time on performance
    - Map location-based study patterns
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.7.4

- **2.7.6** Study habits deep dive (🟢 Completed)
  - **Tests Required**:
    - Test study time patterns are analyzed
    - Test study environment impact is assessed
    - Test preparation patterns are identified
    - Test study habit clusters are discovered
  - **Implementation**:
    - Analyze study time allocation patterns
    - Evaluate study environment effects
    - Identify test preparation strategies
    - Cluster students by study habits
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.7.5

- **2.7.7** Confounding variables assessment (🟢 Completed)
  - **Tests Required**:
    - Test potential confounders are identified
    - Test confounding relationships are analyzed
    - Test interaction effects are evaluated
    - Test control strategies are proposed
  - **Implementation**:
    - Identify potential confounding variables
    - Analyze relationships between confounders
    - Evaluate interaction with key predictors
    - Propose strategies for controlling confounders
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.7.6

### 2.8 EDA Summary and Recommendations (🟢 Completed)

**Objective**: Synthesize findings and provide actionable recommendations

#### Sub-tasks:
- **2.8.1** Key findings documentation (🟢 Completed)
  - **Tests Required**:
    - Test all major findings are documented clearly
    - Test statistical evidence supports conclusions
    - Test visualizations effectively communicate insights
    - Test findings are prioritized by importance
  - **Implementation**:
    - Document top 10 key findings from EDA
    - Provide statistical evidence for each finding
    - Create executive summary visualizations
    - Prioritize findings by potential impact
  - **Database Usage**: In-memory analysis
  - **Dependencies**: 2.7.7

- **2.8.2** Data preprocessing recommendations (🟢 Completed)
  - **Tests Required**:
    - Test preprocessing steps are clearly recommended
    - Test recommendations are based on EDA findings
    - Test preprocessing priority is established
    - Test expected outcomes are documented
  - **Implementation**:
    - Recommend specific preprocessing steps
    - Prioritize preprocessing tasks based on impact
    - Document expected outcomes of each step
    - Create preprocessing pipeline blueprint
  - **Database Usage**: Documentation only
  - **Dependencies**: 2.8.1

- **2.8.3** Feature engineering opportunities (🟢 Completed)
  - **Tests Required**:
    - Test feature engineering ideas are well-justified
    - Test proposed features have theoretical basis
    - Test feature engineering priority is established
    - Test implementation complexity is assessed
  - **Implementation**:
    - Document feature engineering opportunities
    - Justify each proposed feature with EDA evidence
    - Assess implementation complexity and priority
    - Create feature engineering roadmap
  - **Database Usage**: Documentation only
  - **Dependencies**: 2.8.2

- **2.8.4** Model development insights (🟢 Completed)
  - **Tests Required**:
    - Test model selection guidance is provided
    - Test algorithm recommendations are justified
    - Test potential challenges are identified
    - Test success metrics are proposed
  - **Implementation**:
    - Recommend suitable ML algorithms based on data characteristics
    - Identify potential modeling challenges
    - Suggest appropriate evaluation metrics
    - Provide guidance for model selection process
  - **Database Usage**: Documentation only
  - **Dependencies**: 2.8.3

---

## Phase 3: Data Preprocessing and Cleaning

**Objective**: Clean raw data and prepare for feature engineering based on Phase 2 EDA recommendations with prioritized approach

### 3.1 Priority 1: Critical Data Quality Issues (🟢 Completed)

**Objective**: Address critical data quality issues identified in Phase 2 EDA that could compromise model validity

#### Sub-tasks:
- **3.1.1** Age Data Correction (✅ Completed)
  - **Tests Required**:
    - Test identification of all 5 records with negative age values (-5)
    - Test age correction/removal strategy preserves data integrity
    - Test age validation rules prevent future invalid entries
    - Test corrected age distribution matches expected patterns
    - Test impact assessment of age correction on dataset
  - **Implementation**: 
    - **Action**: Remove or correct 5 records with negative age values (-5)
    - **Method**: Investigate if negative ages represent data entry errors vs. systematic issues
    - **Impact**: Critical for model validity - negative ages are impossible
    - Implement age validation rules (0 ≤ age ≤ 100)
    - Document correction decisions and maintain audit trail
  - **Database Usage**: 
    - **Source**: Raw data from SQLite `score.db`
    - **Output**: Age-corrected data to `data/processed/age_corrected.csv`
  - **Dependencies**: 1.2.3

- **3.1.2** Categorical Data Standardization (✅ Completed)
  - **Tests Required**:
    - Test tuition standardization ('Y' → 'Yes') is complete
    - Test case normalization ('CLUBS' → 'Clubs') is applied consistently
    - Test all categorical features follow consistent formatting
    - Test standardization doesn't introduce new inconsistencies
    - Test mapping dictionaries are comprehensive
  - **Implementation**: 
    - **Tuition**: Standardize 'Y' to 'Yes' for consistency with 'Yes'/'No' format
    - **Case normalization**: Convert 'CLUBS' to 'Clubs' format across all categorical features
    - **Method**: Apply consistent string formatting rules across all categorical features
    - Create standardization mapping dictionaries for reproducibility
    - Validate no new inconsistencies are introduced
  - **Database Usage**: 
    - **Source**: `data/processed/age_corrected.csv`
    - **Output**: Standardized data to `data/processed/standardized.csv`
  - **Dependencies**: 3.1.1

#### 3.1.2a ID Structure Analysis (🟢 Completed)
- **Objective**: Analyze and document the structure of all ID fields (student_id, class_id, etc.)
- **Tests Required**:
  - Test uniqueness and format of each ID field
  - Test for embedded information (e.g., year, class, cohort)
  - Test for missing or malformed IDs
- **Implementation**:
  - Profile all ID fields for uniqueness and structure
  - Document any embedded information or patterns
  - Flag and address any anomalies
- **Dependencies**: 3.1.2

#### 3.1.2a.1 Feature Extraction from ID (🟢 Completed)
- **Objective**: Extract useful features from ID fields if patterns exist
- **Tests Required**:
  - Test extraction logic for accuracy
  - Test predictive value of extracted features
- **Implementation**:
  - Implement extraction logic for relevant ID fields
  - Validate extracted features with EDA
- **Dependencies**: 3.1.2a

#### 3.1.2a.2 ID Retention Decision (🟢 Completed)
- **Objective**: Decide whether to retain or drop ID fields after feature extraction
- **Tests Required**:
  - Test model performance with and without ID fields
- **Implementation**:
  - Compare model results
  - Document decision rationale
- **Dependencies**: 3.1.2a.1

#### 3.1.3 Handle duplicate records (🟢 Completed)
- **Objective**: Identify and resolve duplicate records in the dataset
- **Tests Required**:
  - Test duplicate detection logic
  - Test impact of deduplication on data size and integrity
- **Implementation**:
  - Implement deduplication logic
  - Validate with before/after record counts
- **Dependencies**: 3.1.2a.2

#### 3.1.4 Data validation and quality checks (🟢 Completed)
- **Objective**: Implement comprehensive data validation and quality checks
- **Tests Required**:
  - Test for missing, inconsistent, or out-of-range values
  - Test for logical inconsistencies across features
- **Implementation**:
  - Implement validation scripts
  - Document and address issues found
- **Dependencies**: 3.1.3

#### 3.1.5 Enhanced age processing and feature engineering (🟢 Completed)
- **Objective**: Improve age feature quality and create derived features
- **Tests Required**:
  - Test for outliers and inconsistencies in age
  - Test derived age features (e.g., age groups)
- **Implementation**:
  - Clean and validate age data
  - Create and validate derived features
- **Dependencies**: 3.1.4

#### 3.1.6 Comprehensive data entry consistency check (🟢 Completed)
- **Objective**: Ensure consistency in data entry across all features
- **Tests Required**:
  - Test for inconsistent formats, units, or categories
- **Implementation**:
  - Implement consistency checks
  - Standardize formats and categories
- **Dependencies**: 3.1.5

#### 3.1.7 Implement robust outlier handling based on EDA findings (🟢 Completed)
- **Objective**: Detect and handle outliers in key features
- **Tests Required**:
  - Test outlier detection logic
  - Test impact of outlier handling on data distribution
- **Implementation**:
  - Apply EDA-driven outlier detection methods
  - Decide on removal, capping, or transformation
- **Dependencies**: 3.1.6

##### 3.1.7.1 Outlier Detection Based on EDA (🟢 Completed)
- **Objective**: Use EDA results to identify outliers
- **Implementation**:
  - Apply statistical and visual methods
- **Dependencies**: 3.1.7

##### 3.1.7.2 Outlier Analysis and Decision (🟢 Completed)
- **Objective**: Analyze outliers and decide on treatment
- **Implementation**:
  - Document rationale for chosen methods
- **Dependencies**: 3.1.7.1

##### 3.1.7.3 Outlier Treatment Application (🟢 Completed)
- **Objective**: Apply chosen outlier treatment methods
- **Implementation**:
  - Implement and validate treatment
- **Dependencies**: 3.1.7.2

#### 3.1.8 Imbalanced Data Analysis (🟢 Completed)
- **Objective**: Assess and address class imbalance in target variable
- **Tests Required**:
  - Test for imbalance severity
  - Test impact of balancing methods
- **Implementation**:
  - Analyze class distribution
  - Apply and validate balancing techniques if needed
- **Dependencies**: 3.1.7.3

### 3.2 Priority 2: Missing Data Strategy (✅ Completed)

**Objective**: Implement comprehensive missing data handling strategy based on Phase 2 EDA findings

#### Sub-tasks:
- **3.2.1** Missing Data Imputation for Attendance Rate (✅ Completed)
  - **Tests Required**:
    - Test identification of all 778 missing attendance_rate values (4.89%)
    - Test median imputation preserves distribution characteristics
    - Test regression-based imputation improves accuracy
    - Test multiple imputation techniques comparison
    - Test imputation strategy effectiveness validation
  - **Implementation**: 
    - **Attendance_rate** (4.89% missing): Use median imputation or regression-based imputation
    - **Method**: Implement multiple imputation techniques and compare performance
    - Test median imputation by relevant subgroups (gender, CCA, learning_style)
    - Evaluate regression-based imputation using correlated features
    - Create missing indicator variables where missingness may be informative
    - Document imputation strategy and validate against EDA findings
  - **Database Usage**: 
    - **Source**: `data/processed/standardized.csv`
    - **Output**: Imputed data to `data/processed/imputed.csv`
  - **Dependencies**: 3.1.2

- **3.2.2** Final Test Missing Values Handling (✅ Completed)
  - **Tests Required**:
    - Test identification of all 495 missing final_test values (3.11%)
    - Test exclusion from training preserves data integrity
    - Test sophisticated imputation methods if retention is chosen
    - Test impact assessment of missing target handling strategy
  - **Implementation**: 
    - **Final_test** (3.11% missing): Consider excluding from training or use sophisticated imputation
    - **Method**: Evaluate exclusion vs. advanced imputation (KNN, iterative imputation)
    - If excluding: ensure proper handling in train/validation/test splits
    - If imputing: use advanced methods and validate against known values
    - Document decision rationale and impact on model performance
  - **Database Usage**: 
    - **Source**: `data/processed/imputed.csv`
    - **Output**: Final missing-handled data to `data/processed/missing_handled.csv`
  - **Dependencies**: 3.2.1

### 3.3 Priority 3: Feature Engineering Opportunities (✅ Completed)

**Objective**: Create derived and interaction features based on Phase 2 EDA insights

#### Sub-tasks:
- **3.3.1** Derived Features Creation (✅ Completed)
  - **Tests Required**:
    - Test sleep duration calculation from sleep_time and wake_time
    - Test study intensity combination of hours_per_week with attendance_rate
    - Test academic support index creation from multiple features
    - Test age groups categorical creation for interpretability
    - Test derived features correlation with target variable
  - **Implementation**: 
    - **Sleep duration**: Calculate from sleep_time and wake_time
    - **Study intensity**: Combine hours_per_week with attendance_rate (weighted average or product)
    - **Academic support index**: Combine tuition, direct_admission, and CCA participation
    - **Age groups**: Create categorical age bands for better interpretability (if justified)
    - Validate all derived features make domain sense and improve predictive power
  - **Database Usage**: 
    - **Source**: `data/processed/missing_handled.csv`
    - **Output**: Feature-enhanced data to `data/processed/derived_features.csv`
  - **Dependencies**: 3.2.2

- **3.3.2** Interaction Features Creation (✅ Completed)
  - **Tests Required**:
    - Test study habits × learning style interaction significance
    - Test support × performance interaction effects
    - Test demographics × academics interaction patterns
    - Test interaction features improve model performance
    - Test interaction features don't cause multicollinearity
  - **Implementation**: 
    - **Study habits × Learning style**: Interaction between hours_per_week and learning_style
    - **Support × Performance**: Interaction between academic support factors and attendance
    - **Demographics × Academics**: Gender and age interactions with study patterns
    - Apply statistical tests to validate interaction significance
    - Use feature selection to prevent overfitting from too many interactions
  - **Database Usage**: 
    - **Source**: `data/processed/derived_features.csv`
    - **Output**: Interaction-enhanced data to `data/processed/interaction_features.csv`
  - **Dependencies**: 3.3.1


### 3.4 Priority 4: Data Preprocessing Pipeline (🟢 Completed)

**Objective**: Implement comprehensive data preprocessing pipeline for model readiness

#### Sub-tasks:
- **3.4.1** Data Splitting and Validation (🟢 Completed)
  - **Tests Required**:
    - Test stratified split maintains target distribution
    - Test 70/15/15 or 80/10/10 split ratio is correct
    - Test no data leakage between splits
    - Test split is reproducible with random seed
    - Test target variable distribution is preserved across splits
  - **Implementation**: 
    - Stratified train/validation/test split based on target variable bins
    - Handle missing target values appropriately (exclude from training)
    - Ensure reproducible splits with fixed random state
    - Document split strategy and validate distributions
  - **Database Usage**:
    - **Source**: `data/processed/interaction_features.csv`
    - **Output**: `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`
  - **Dependencies**: 3.3.2

- **3.4.2** Validate Data Splits Integrity (🟢 Completed)
  - **Tests Required**:
    - Test no overlap between train/validation/test sets
    - Test target distribution similarity across splits
    - Test feature distributions are consistent
    - Test data quality metrics are maintained
  - **Implementation**:
    - Statistical tests for distribution similarity
    - Data quality validation across all splits
    - Generate split summary report
    - Validate no student appears in multiple splits
  - **Database Usage**:
    - **Source**: Split datasets from 3.4.1
    - **Output**: Validation report
  - **Dependencies**: 3.4.1

### 3.5 Priority 5: Advanced Preprocessing (🟢 Completed)

**Objective**: Apply advanced preprocessing techniques for optimal model performance

#### Sub-tasks:
- **3.5.1** Feature Scaling and Normalization (🟢 Completed)
  - **Tests Required**:
    - Test StandardScaler transforms features to mean=0, std=1
    - Test MinMaxScaler transforms features to [0,1] range
    - Test RobustScaler handles outliers appropriately
    - Test scaling preserves feature relationships
    - Test inverse transform capability
  - **Implementation**: 
    - Apply appropriate scaling based on feature distributions
    - StandardScaler for normally distributed features
    - RobustScaler for features with outliers
    - MinMaxScaler for bounded features
    - Fit scalers on training data only, transform all splits
  - **Database Usage**:
    - **Source**: Split datasets from 3.4.1
    - **Output**: Scaled datasets `data/processed/scaled_train.csv`, etc.
  - **Dependencies**: 3.4.2

- **3.5.2** Categorical Encoding Optimization (🟢 Completed)
  - **Tests Required**:
    - Test one-hot encoding for low cardinality features
    - Test target encoding for high cardinality features
    - Test ordinal encoding for ordered categories
    - Test encoding handles unseen categories in validation/test
    - Test encoding doesn't introduce data leakage
  - **Implementation**: 
    - One-hot encoding for gender, learning_style, transport_mode
    - Target encoding for high cardinality features (if any)
    - Ordinal encoding for ordered categories (if applicable)
    - Handle unseen categories with 'unknown' category
    - Fit encoders on training data only
  - **Database Usage**:
    - **Source**: Scaled datasets from 3.5.1
    - **Output**: Encoded datasets `data/processed/encoded_train.csv`, etc.
  - **Dependencies**: 3.5.1

### 3.6 Data Storage and Backup (🟢 Completed)

**Objective**: Ensure processed data is properly stored and backed up

#### Sub-tasks:
- **3.6.1** Save Cleaned Dataset (🟢 Completed)
  - **Tests Required**:
    - Test CSV file is created with correct format
    - Test all cleaned data is preserved
    - Test file permissions are appropriate
    - Test data can be reloaded without errors
  - **Implementation**: Pandas to_csv with proper encoding and formatting
  - **Database Usage**:
    - **Source**: Encoded datasets from 3.5.2
    - **Output**: `data/processed/final_processed.csv`
  - **Dependencies**: 3.5.2

- **3.6.2** Create Data Backup and Recovery Procedures (🟢 Completed)
  - **Tests Required**:
    - Test backup creation process
    - Test backup integrity verification
    - Test recovery process restores data correctly
    - Test backup rotation works as expected
  - **Implementation**: Automated backup scripts with versioning
  - **Database Usage**:
    - **Source**: `data/processed/final_processed.csv`
    - **Output**: Timestamped backups in `data/backups/`
  - **Dependencies**: 3.6.1

---

## Phase 4: Feature Engineering

**Objective**: Create new features, apply advanced preprocessing, and select optimal features to improve model performance based on Phase 3 insights and Phase 2 EDA recommendations.

### 4.1 Load and Validate Data (🟢 Completed)
- **Objective**: Ensure the processed dataset is loaded correctly and validated before feature engineering.
- **Sub-tasks**:
  - **4.1.1** Load `final_processed.csv` (🟢 Completed)
    - **Tests Required**:
      - Test data loads correctly from `data/processed/final_processed.csv`.
      - Test all expected columns (original and Phase 3 derived) are present.
      - Test data types are preserved (e.g., continuous age, standardized categoricals).
      - Test no data corruption during loading.
      - Test expected number of records (post-deduplication and cleaning) are loaded.
    - **Implementation**: Pandas `read_csv` with appropriate parameters and validation checks.
    - **Database Usage**: Source: `data/processed/final_processed.csv`.
    - **Dependencies**: 3.6.1 (Save Cleaned Dataset).

### 4.2 High Priority Feature Engineering (Phase 4.1) (🟡 In Progress)
- **Objective**: Implement high-impact derived features and transformations identified in Phase 3 recommendations.

#### 4.2.1 EDA-Driven Derived Features (High Priority) (🟢 Completed)
- **Sub-tasks**:
  - **4.2.1.1** Create Study Efficiency Score (🟢 Completed)
    - **Description**: Combine `study_hours` and `attendance_rate`.
    - **Formula**: `(study_hours * attendance_rate) / max_possible_score` (or other suitable normalization).
    - **Rationale**: EDA showed strong correlation (>0.6) with target. Ref: `notebook/visualization/correlation_heatmap.png`.
    - **Tests Required**: Test score calculation, normalization, distribution.
    - **Implementation**: Pandas operations.
    - **Dependencies**: 4.1.1.
  - **4.2.1.2** Create Academic Support Index (🟢 Completed)
    - **Description**: Weighted combination of `tuition`, `direct_admission`, `extracurricular_activities`.
    - **Rationale**: EDA categorical analysis showed these as key differentiators. Ref: `notebook/visualization/categorical_features_distribution.png`.
    - **Tests Required**: Test index calculation, weighting logic, distribution.
    - **Implementation**: Pandas operations, define weighting scheme.
    - **Dependencies**: 4.1.1.

#### 4.2.2 High-Impact Interaction Features (Primary) (🟢 Completed)
- **Sub-tasks**:
  - **4.2.2.1** Create Study × Attendance Interaction (🟢 Completed)
    - **Description**: `study_hours * attendance_rate`.
    - **Justification**: Highest correlation pair in EDA (r = 0.67). Expected primary predictor.
    - **Tests Required**: Test interaction term calculation, impact on model.
    - **Implementation**: Pandas operations.
    - **Dependencies**: 4.1.1.

#### 4.2.3 Distribution-Based Transformations (High Priority) (🟢 Completed)
- **Sub-tasks**:
  - **4.2.3.1** Transform Right-Skewed Variables (🟢 Completed)
    - **Variables**: `study_hours` (skewness = 1.2), `previous_score` (skewness = 0.8).
    - **Transformations**: Log for `study_hours`, Box-Cox for `previous_score`.
    - **Rationale**: Identified in EDA. Ref: `notebook/visualization/numerical_features_distribution.png`.
    - **Tests Required**: Test transformation application, resulting skewness.
    - **Implementation**: Scikit-learn `PowerTransformer` or `FunctionTransformer`.
    - **Dependencies**: 4.1.1.

### 4.3 Medium Priority Feature Engineering (Phase 4.2) (🟢 Completed)
- **Objective**: Implement additional valuable features and encoding strategies.

#### 4.3.1 Time-Based Derived Features (Medium Priority) (🟢 Completed)
- **Sub-tasks**:
  - **4.3.1.1** Create Study Time Categories (🟢 Completed)
    - **Description**: Categorize study times (e.g., Early, Peak, Afternoon, Evening).
    - **Rationale**: Based on EDA distribution analysis. Ref: `notebook/visualization/numerical_features_distribution.png`.
    - **Tests Required**: Test categorization logic, distribution of categories.
    - **Implementation**: Pandas `cut` or custom mapping.
    - **Dependencies**: 4.1.1.
  - **4.3.1.2** Create Sleep Quality Indicator (🟢 Completed)
    - **Description**: Categorize sleep duration (e.g., Optimal, Insufficient, Excessive).
    - **Rationale**: EDA showed non-linear relationship with performance.
    - **Tests Required**: Test sleep duration calculation (if not already done), categorization logic.
    - **Implementation**: Pandas operations based on `sleep_time` and `wake_time`.
    - **Dependencies**: 4.1.1.

#### 4.3.2 High-Impact Interaction Features (Secondary) (🟢 Completed)
- **Sub-tasks**:
  - **4.3.2.1** Create Parent Education × Socioeconomic Interaction (🟢 Completed)
    - **Description**: Cross-categorical interaction.
    - **Justification**: EDA showed compound effect. Consider target encoding for high-cardinality combinations.
    - **Tests Required**: Test interaction creation, encoding strategy if used.
    - **Implementation**: Pandas, potentially scikit-learn encoders.
    - **Dependencies**: 4.1.1.
  - **4.3.2.2** Create Sleep × Study Hours Interaction (🟢 Completed)
    - **Rationale**: Non-linear interaction for optimal study conditions.
    - **Tests Required**: Test interaction term calculation.
    - **Implementation**: Pandas operations.
    - **Dependencies**: 4.1.1.
  - **4.3.2.3** Create Exercise × Academic Performance Interaction (🟢 Completed)
    - **Rationale**: Balance indicator from EDA insights.
    - **Tests Required**: Test interaction term calculation.
    - **Implementation**: Pandas operations.
    - **Dependencies**: 4.1.1.
  - **4.3.2.4** Create Transport × Attendance Interaction (🟢 Completed)
    - **Rationale**: Accessibility impact on consistent attendance.
    - **Tests Required**: Test interaction term calculation.
    - **Implementation**: Pandas operations.
    - **Dependencies**: 4.1.1.

#### 4.3.3 Advanced Categorical Encoding Strategy (Medium Priority) (🟢 Completed)
- **Sub-tasks**:
  - **4.3.3.1** Implement One-Hot Encoding (🟢 Completed)
    - **Variables**: Low cardinality features (e.g., `gender`, `transport_mode`, `learning_style`).
    - **Tests Required**: Test correct number of columns created, no data leakage.
    - **Implementation**: Scikit-learn `OneHotEncoder`.
    - **Dependencies**: 4.1.1.
  - **4.3.3.2** Implement Target Encoding (🟢 Completed)
    - **Variables**: High cardinality features (e.g., `extracurricular_activities`, `sleep_time`, `wake_time`).
    - **Tests Required**: Test encoding logic, prevention of data leakage (use on training folds only).
    - **Implementation**: `category_encoders` library or custom implementation.
    - **Dependencies**: 4.1.1.
  - **Note**: Binary features like `tuition`, `direct_admission` were standardized to 'Yes'/'No' in Phase 3 and may need conversion to 0/1 or can be handled by OHE.

### 4.4 Enhancement Feature Engineering & Selection (Phase 4.3) (🟢 Completed)
- **Objective**: Implement ID-based features if promising, and apply feature selection techniques.

#### 4.4.1 ID-Based Features (Enhancement) (🟢 Completed)
- **Sub-tasks**:
  - **4.4.1.1** Create Enrollment Cohort Feature (🟢 Completed)
    - **Description**: Extract year/semester patterns from student ID if Phase 3 analysis confirmed utility.
    - **Reference**: Results from `src/data/id_structure_analysis.py`.
    - **Tests Required**: Test extraction logic, feature distribution.
    - **Implementation**: Pandas string operations based on ID patterns.
    - **Dependencies**: 4.1.1, `src/data/id_structure_analysis.py` findings.

#### 4.4.2 Feature Scaling (Enhancement) (🟢 Completed)
- **Sub-tasks**:
  - **4.4.2.1** Apply Outlier-Robust Scaling (🟢 Completed)
    - **Rationale**: For features with outliers identified in EDA. Ref: `notebook/visualization/numerical_features_boxplots.png`.
    - **Tests Required**: Test scaler application, effect on distributions.
    - **Implementation**: Scikit-learn `RobustScaler`.
    - **Dependencies**: All numerical features created/transformed.
  - **Note**: Other scalers like `StandardScaler` or `MinMaxScaler` can be applied as needed based on model requirements and feature distributions after transformations.

#### 4.4.3 Feature Selection Strategy (Enhancement) (🟢 Completed)
- **Sub-tasks**:
  - **4.4.3.1** Correlation-Based Selection (🟢 Completed)
    - **Action**: Remove features with correlation > 0.9 (multicollinearity threshold).
    - **Target**: Reduce feature space by ~15-20% while maintaining predictive power.
    - **Tests Required**: Test correlation calculation, feature removal.
    - **Implementation**: Pandas `corr()`, identify and drop columns.
    - **Dependencies**: All features created.
  - **4.4.3.2** Importance-Based Selection (🟢 Completed)
    - **Methods**: Random Forest feature importance as baseline, Recursive Feature Elimination (RFE).
    - **Target**: Select top ~80% of features by importance score or as determined by RFE.
    - **Tests Required**: Test importance calculation, RFE application, selected feature set.
    - **Implementation**: Scikit-learn `RandomForestRegressor`, `RFE`.
    - **Dependencies**: All features created.

### 4.5 Final Validation and Quality Checks (🟢 Completed)
- **Objective**: Ensure all engineered features meet quality targets before model training.
- **Sub-tasks**:
  - **4.5.1** Validate Feature Completeness (🟢 Completed)
    - **Target**: Maintain 100% (achieved in Phase 3 for base features).
    - **Tests Required**: Check for NaNs in all engineered features.
  - **4.5.2** Validate Feature Consistency (🟢 Completed)
    - **Target**: Maintain 100% categorical standardization/encoding.
    - **Tests Required**: Verify encoding schemes are applied correctly.
  - **4.5.3** Validate Feature Validity (🟢 Completed)
    - **Target**: All derived features pass domain validation (e.g., ranges, logical sense).
    - **Tests Required**: Implement specific checks for new features.
  - **4.5.4** Check Correlation Threshold (🟢 Completed)
    - **Target**: No feature pairs with |r| > 0.95 (after selection step 4.4.3.1).
    - **Tests Required**: Re-calculate correlation matrix on final feature set.
  - **4.5.5** Assess Feature Engineering Quality Score (🟢 Completed)
    - **Target**: >85% (define specific metrics for this score).
  - **4.5.6** Assess Model Readiness Score (🟢 Completed)
    - **Target**: >90% (define specific metrics).
  - **4.5.7** Ensure Feature Interpretability (🟢 Completed)
    - **Target**: All features have clear business/domain meaning.
    - **Action**: Document each new feature's derivation and meaning.

### 4.6 Save Feature-Engineered Dataset and Documentation (🟢 Completed)
- **Objective**: Persist the final feature set and its documentation.
- **Sub-tasks**:
  - **4.6.1** Save Feature-Engineered Datasets (🟢 Completed)
    - **Implementation**: Save processed features (train, validation, test splits if applicable) to CSV or other format.
    - **Output**: e.g., `data/featured/train_features.csv`, `data/featured/validation_features.csv`, `data/featured/test_features.csv`.
    - **Dependencies**: 4.5 (All validation tasks).
  - **4.6.2** Create/Update Feature Documentation (🟢 Completed)
    - **Implementation**: Generate/update feature dictionary, transformation log, and definitions for all new and modified features.
    - **Output**: Update `data/featured/feature_definitions.json` or create a new comprehensive `Phase4_Feature_Documentation.md`.
    - **Dependencies**: 4.6.1.

---

## Phase 5: Model Development (🔴 Not Started)

### 5.1 Model Training Infrastructure (🔴 Not Started)

**Objective**: Setup robust model training and evaluation framework

#### Sub-tasks:
- **5.1.3** Address Imbalanced Data (if identified) (🔴 Not Started)
  - **Description**: Analyze target variable distribution and apply appropriate balancing techniques if significant imbalance is identified. This was a re-prioritized task.
  - **Tests Required**:
    - Test imbalance detection in target variable (`final_test`).
    - Test application of chosen balancing technique (e.g., SMOTE, class weights) if imbalance is confirmed.
    - Test evaluation metrics suitable for imbalanced data are used if balancing is applied (e.g., F1-score, AUC-PR, balanced accuracy).
  - **Implementation**:
    - Analyze distribution of target variable (`final_test`).
    - If significant imbalance is found, apply techniques like SMOTE, ADASYN, random undersampling, or using class weights in models.
    - Select evaluation metrics robust to imbalanced data if applicable.
  - **Database Usage**:
    - **Source**: `data/processed/final_processed.csv` (or equivalent from Phase 4)
    - **Output**: Potentially resampled data or models trained with class weights.
  - **Dependencies**: 5.1.2

- **5.1.4** Integrate Phase 4 Feature Documentation (🔴 Not Started)
  - **Description**: Utilize the comprehensive feature documentation from Phase 4 for model interpretation and stakeholder communication, as recommended (Rec 10).
  - **Tests Required**:
    - Test that feature definitions are accessible during modeling.
    - Test that model reports can reference feature documentation.
  - **Implementation**:
    - Develop a mechanism to link model outputs (e.g., feature importance) back to feature definitions from `data/featured/feature_definitions.json` and `interaction_definitions.json`.
    - Ensure reporting tools can leverage this documentation.
  - **Database Usage**:
    - **Source**: `data/featured/feature_definitions.json`, `data/featured/interaction_definitions.json`
    - **Output**: Enhanced model reports and interpretability.
  - **Dependencies**: Phase 4 Documentation Artifacts

- **5.1.1** Setup cross-validation framework (🔴 Not Started)
  - **Description**: Implement stratified k-fold cross-validation (k=5 or k=10) to ensure robust evaluation across different score ranges, as recommended in Phase 4 report (Rec 4).
  - **Tests Required**:
    - Test k-fold split generation is correct (e.g., 5 or 10 folds).
    - Test stratification is applied based on target variable.
    - Test evaluation metrics are calculated per fold and averaged.
    - Test CV folds are reproducible with a consistent random state.
  - **Implementation**: StratifiedKFold from scikit-learn or similar, ensuring consistent random state for reproducibility.
  - **Database Usage**: In-memory cross-validation on training data.
  - **Dependencies**: 4.3.1, 5.1.2

- **5.1.2** Implement model evaluation metrics (🔴 Not Started)
  - **Tests Required**:
    - Test MAE calculation is mathematically correct.
    - Test RMSE penalizes large errors appropriately.
    - Test R² score interpretation is accurate.
    - Test custom metrics (e.g., MAPE, median Absolute Error) work correctly if implemented.
  - **Implementation**: Develop a comprehensive suite of evaluation metrics (MAE, RMSE, R², etc.) for model comparison.
  - **Database Usage**: In-memory metric calculations on model predictions.
  - **Dependencies**: None directly, but used by subsequent model training/evaluation tasks.

### 5.2 Algorithm Implementation (🔴 Not Started)

**Objective**: Implement and train multiple ML algorithms

#### Sub-tasks:
- **5.2.1** Implement Random Forest Regressor (🔴 Not Started)
  - **Description**: Implement Random Forest as one of the primary algorithms, given its ability to handle interactions well (Phase 4 Rec 3).
  - **Tests Required**:
    - Test model trains without errors using the Phase 4 feature set.
    - Test hyperparameter tuning (e.g., using GridSearchCV or RandomizedSearchCV) improves performance.
    - Test feature importance extraction (e.g., `feature_importances_` attribute) works and aligns with Phase 4 findings (Rec 6).
    - Test model achieves target performance (e.g., MAE < 8, R² > 0.75, or as defined by project goals).
  - **Implementation**:
    - `RandomForestRegressor` from scikit-learn.
    - Include key Phase 4 features (Study × Attendance, Study Efficiency Score, Academic Support Index - Rec 1 & 2).
    - Feature importance analysis and documentation.
  - **Database Usage**: Training on in-memory feature data from `data/featured/final_features.csv`.
  - **Dependencies**: 5.1.1, 5.1.2, Phase 4 Feature Set

- **5.2.2** Implement XGBoost Regressor (🔴 Not Started)
  - **Description**: Implement XGBoost, another key algorithm recommended for its performance and handling of interactions (Phase 4 Rec 3). Focus on hyperparameter tuning for this model (Rec 5).
  - **Tests Required**:
    - Test XGBoost installation and import.
    - Test model handles categorical features correctly (if applicable, or ensure proper encoding from Phase 4).
    - Test hyperparameter optimization (e.g., GridSearchCV, RandomizedSearchCV, or Bayesian Optimization) improves performance.
    - Test model achieves target performance (e.g., MAE < 7, R² > 0.80, or as defined).
    - Test SHAP value generation for interpretability (Rec 8).
  - **Implementation**:
    - `XGBRegressor` from the XGBoost library.
    - Include key Phase 4 features (Rec 1 & 2).
    - Implement early stopping and regularization to prevent overfitting (Rec 7).
    - SHAP value analysis for feature importance and model explanation.
  - **Database Usage**: Training on in-memory feature data from `data/featured/final_features.csv`.
  - **Dependencies**: 5.1.1, 5.1.2, Phase 4 Feature Set

- **5.2.3** Implement Linear Regression baseline (🔴 Not Started)
  - **Description**: Implement Linear Regression as a baseline model for comparison and to establish initial performance benchmarks (Phase 4 Rec 9). Also explore SVR as part of algorithm selection (Rec 3).
  - **Tests Required**:
    - Test `LinearRegression` (potentially with polynomial features) trains correctly.
    - Test regularization (Ridge/Lasso) can be applied and its impact evaluated (Rec 7).
    - Test coefficient interpretation is meaningful for the linear model.
    - Test `SVR` (Support Vector Regression) with different kernels trains correctly.
    - Test both models provide interpretable baseline performance metrics.
  - **Implementation**:
    - `LinearRegression` from scikit-learn, potentially with `PolynomialFeatures`.
    - `Ridge` and `Lasso` for regularization.
    - `SVR` from scikit-learn with linear and RBF kernels.
    - Coefficient analysis for Linear Regression.
  - **Database Usage**: Training on in-memory feature data from `data/featured/final_features.csv`.
  - **Dependencies**: 5.1.1, 5.1.2, Phase 4 Feature Set



- **5.2.4** Implement Neural Network (🔴 Not Started)
  - **Description**: Implement a Neural Network, with a focus on hyperparameter tuning and overfitting monitoring (Phase 4 Rec 3, 5, 7).
  - **Tests Required**:
    - Test neural network architecture (e.g., MLPRegressor or Keras Sequential) is appropriate for the regression task.
    - Test training converges without significant overfitting (monitor validation loss).
    - Test early stopping callback prevents overfitting effectively.
    - Test model achieves competitive performance compared to other algorithms.
  - **Implementation**:
    - `MLPRegressor` from scikit-learn or a Keras/TensorFlow `Sequential` model.
    - Architecture optimization (layers, neurons, activation functions).
    - Implement regularization (e.g., L2, dropout) and early stopping.
  - **Database Usage**: Training on in-memory feature data from `data/featured/final_features.csv`.
  - **Dependencies**: 5.1.1, 5.1.2, Phase 4 Feature Set

- **5.2.5** Prioritize Key Interaction and Composite Features (🔴 Not Started)
    - **Description**: Ensure high-priority features identified in Phase 4 (Study × Attendance interaction, Study Efficiency Score, Academic Support Index) are included and evaluated in all models, as per recommendations (Rec 1 & 2).
    - **Tests Required**:
      - Test that models can be configured to include/exclude these specific features.
      - Test that the impact of these features on model performance is measurable.
    - **Implementation**:
      - Explicitly include `Study_X_Attendance`, `Study_Efficiency_Score`, and `Academic_Support_Index` (or their equivalents from Phase 4 feature engineering) in the feature set for initial model runs.
      - Analyze model performance with and without these key features to quantify their impact.
    - **Database Usage**:
      - **Source**: `data/featured/final_features.csv` (containing these key features)
      - **Output**: Model performance metrics highlighting the contribution of these features.
    - **Dependencies**: Phase 4 Feature Set, 5.2.1, 5.2.2, 5.2.3, 5.2.4

### 5.3 Model Selection and Optimization (🔴 Not Started)

**Objective**: Select best performing model and optimize hyperparameters

#### Sub-tasks:
- **5.3.1** Hyperparameter optimization (🔴 Not Started)
  - **Description**: Perform systematic hyperparameter tuning for promising models, with a particular focus on XGBoost and Neural Networks as recommended (Phase 4 Rec 5).
  - **Tests Required**:
    - Test hyperparameter search (e.g., GridSearchCV, RandomizedSearchCV, Optuna) improves performance over default parameters.
    - Test that the optimization process uses cross-validation to prevent overfitting to a single validation split.
    - Test best parameters are reproducible and can be logged.
    - Test optimization time is within acceptable limits.
  - **Implementation**:
    - Define appropriate hyperparameter grids/distributions for each model type.
    - Use techniques like `GridSearchCV`, `RandomizedSearchCV`, or Bayesian Optimization (e.g., Optuna, Hyperopt).
    - Ensure tuning is performed within the cross-validation framework (Task 5.1.1).
    - Store best parameter configurations for each model.
  - **Database Usage**: In-memory training data for optimization, leveraging CV splits.
  - **Dependencies**: 5.2.1, 5.2.2, 5.2.3, 5.2.4 (i.e., implemented models)

- **5.3.2** Model comparison and selection (🔴 Not Started)
  - **Tests Required**:
    - Test model comparison uses consistent metrics
    - Test statistical significance of performance differences
    - Test best model selection criteria are appropriate
    - Test selected model meets performance requirements
  - **Implementation**:
    - Comprehensive model comparison framework
    - Statistical testing for performance differences
    - Multi-criteria decision making
  - **Database Usage**: In-memory evaluation results
  - **Dependencies**: 5.3.1

- **5.3.3** Final model training and validation (🔴 Not Started)
  - **Description**: Train the selected best model(s) on the full training dataset using optimized hyperparameters and evaluate on the unseen test set.
  - **Tests Required**:
    - Test final model trains on the complete training dataset without errors.
    - Test model performance on the held-out test set meets or exceeds defined benchmarks (Phase 4 Rec 9).
    - Test model generalization is satisfactory (i.e., performance on test set is close to cross-validation performance).
    - Test model meets all predefined success criteria for the project.
  - **Implementation**:
    - Train the chosen model(s) from Task 5.3.2 on the entire training portion of `data/featured/final_features.csv`.
    - Perform final evaluation on the designated test split.
    - Document final performance metrics and compare against project goals.
  - **Database Usage**: Full training data for final model training; test data for final validation.
  - **Dependencies**: 5.3.2

- **5.3.4** Implement Overfitting Monitoring and Mitigation (🔴 Not Started)
    - **Description**: Implement early stopping for complex models (e.g., NNs, XGBoost) and use regularization (L1/L2) for linear models to prevent overfitting, as recommended (Rec 7).
    - **Tests Required**:
      - Test learning curves show convergence without significant overfitting.
      - Test regularization parameters are tunable and effective for linear models.
      - Test early stopping callback functions correctly halt training for iterative models.
    - **Implementation**:
      - Plot learning curves (training vs. validation loss/metric over epochs/iterations) during training of iterative models.
      - Implement L1/L2 regularization in relevant scikit-learn models (e.g., LinearRegression, LogisticRegression if used).
      - Use early stopping callbacks in Keras/TensorFlow, XGBoost, LightGBM based on validation set performance.
    - **Database Usage**:
      - **Source**: Model training logs, performance metrics from validation folds/sets.
      - **Output**: Optimized models with reduced overfitting, learning curve plots.
    - **Dependencies**: 5.2.1, 5.2.2, 5.2.3, 5.2.4, 5.3.1

- **5.3.5** Implement Model Interpretability Techniques (🔴 Not Started)
    - **Description**: Leverage techniques like SHAP values or LIME to explain model predictions, especially for complex models and the engineered interaction features, as recommended (Rec 8). This should also validate engineered features from Phase 4 (Rec 6).
    - **Tests Required**:
      - Test SHAP/LIME explanations can be generated for selected models (especially RF, XGBoost, NN).
      - Test visualizations of feature contributions (e.g., SHAP summary plots, dependence plots) are clear and informative.
      - Test local (individual prediction) and global (overall model behavior) explanations can be derived.
      - Test if interpretability results align with Phase 4 feature importance findings.
    - **Implementation**:
      - Integrate SHAP or LIME libraries with trained models.
      - Generate summary plots (e.g., SHAP summary plot, feature importance from SHAP values).
      - Implement functionality to explain individual predictions for case studies.
      - Document insights from interpretability analysis, linking back to Phase 4 documentation (Rec 10).
    - **Database Usage**:
      - **Source**: Trained models, `data/featured/final_features.csv`.
      - **Output**: SHAP/LIME values, interpretability plots, explanation reports.
    - **Dependencies**: 5.3.2, 5.1.4

### 5.4 Model Persistence and Registry (🔴 Not Started)

**Objective**: Save and manage trained models for deployment

#### Sub-tasks:
- **5.4.1** Implement model serialization (🔴 Not Started)
  - **Tests Required**:
    - Test model saves and loads correctly
    - Test serialized model produces identical predictions
    - Test model metadata is preserved
    - Test serialization handles all model types
  - **Implementation**:
    - Joblib/pickle for scikit-learn models
    - Model versioning and metadata storage
    - Serialization format standardization
  - **Database Usage**:
    - **Source**: Trained models in memory
    - **Output**: Model files in `models/` directory
  - **Dependencies**: 5.3.3

- **5.4.2** Create model registry (🔴 Not Started)
  - **Tests Required**:
    - Test model registry tracks all trained models
    - Test model metadata is complete and accurate
    - Test model versioning works correctly
    - Test model retrieval by version/performance
  - **Implementation**:
    - Model registry database/file system
    - Model metadata tracking
    - Version control for models
  - **Database Usage**:
    - **Source**: Trained models and metadata
    - **Output**: Model registry database/files
  - **Dependencies**: 5.4.1

---

## Phase 6: Testing & Validation

### 6.1 Unit Testing (🔴 Not Started)

**Objective**: Comprehensive unit test coverage

#### Sub-tasks:
- **6.1.1** Data processing unit tests (🔴 Not Started)
  - **Tests Required**:
    - Test all data cleaning functions
    - Test feature engineering functions
    - Test data validation functions
    - Test error handling in edge cases
  - **Implementation**: pytest test suite for data processing
  - **Database Usage**: Test databases and mock data
  - **Dependencies**: 3.3.2

- **6.1.2** Model training unit tests (🔴 Not Started)
  - **Tests Required**:
    - Test model training pipeline
    - Test evaluation metrics calculation
    - Test model serialization/deserialization
    - Test hyperparameter optimization
  - **Implementation**: pytest test suite for ML components
  - **Database Usage**: Test data and mock models
  - **Dependencies**: 5.4.2

### 6.2 Integration Testing (🔴 Not Started)

**Objective**: Test component interactions

#### Sub-tasks:
- **6.2.1** End-to-end pipeline testing (🔴 Not Started)
  - **Tests Required**:
    - Test complete data processing pipeline
    - Test model training to deployment flow
    - Test database operations
  - **Implementation**: Integration test suite
  - **Database Usage**: Test database with realistic data
  - **Dependencies**: 6.1.2

- **6.2.2** Performance testing (🔴 Not Started)
  - **Tests Required**:
    - Test model training performance
    - Test prediction generation times
    - Test memory usage during training and prediction
    - Test data processing pipeline performance
  - **Implementation**: Performance benchmarking suite
  - **Database Usage**: Performance test database
  - **Dependencies**: 6.2.1

### 6.3 Model Validation (🔴 Not Started)

**Objective**: Validate model performance and reliability

#### Sub-tasks:
- **6.3.1** Cross-validation testing (🔴 Not Started)
  - **Tests Required**:
    - Test model performance consistency
    - Test overfitting detection
    - Test generalization capability
    - Test performance on different data subsets
  - **Implementation**: Comprehensive CV testing framework
  - **Database Usage**: Full dataset for validation
  - **Dependencies**: 4.3.3

- **6.3.2** Model interpretability testing (🔴 Not Started)
  - **Tests Required**:
    - Test feature importance consistency
    - Test SHAP value calculations
    - Test model explanation accuracy
    - Test interpretability tool integration
  - **Implementation**: Model interpretability test suite
  - **Database Usage**: Test data for interpretability
  - **Dependencies**: 6.3.1



---

## Discovered Tasks (⚪ Discovered)

*This section will be populated with tasks discovered during development*

### Data Discovery Tasks
- ⚪ **Additional Data Quality Issues**: Tasks discovered during data exploration
- ⚪ **Feature Engineering Opportunities**: New feature ideas from data analysis
- ⚪ **Performance Optimization**: Bottlenecks identified during development

### Technical Discovery Tasks
- ⚪ **Integration Challenges**: Unexpected technical integration issues
- ⚪ **Scalability Requirements**: Performance requirements beyond initial scope
- ⚪ **Security Enhancements**: Additional security measures needed

### Business Discovery Tasks
- ⚪ **User Feedback Integration**: Changes based on stakeholder feedback
- ⚪ **Additional Use Cases**: New requirements from user interactions
- ⚪ **Compliance Requirements**: Regulatory or policy requirements

---

## Backlog (Future Enhancements)

### Advanced ML Features
- **Ensemble Methods**: Implement voting/stacking regressors for improved accuracy
- **AutoML Integration**: Automated model selection and hyperparameter tuning
- **Deep Learning Models**: Advanced neural network architectures
- **Online Learning**: Incremental model updates with new data
- **Explainable AI**: Advanced model interpretability tools

### Data Pipeline Enhancements
- **Real-time Data Pipeline**: Streaming data processing capabilities
- **Data Drift Detection**: Automated monitoring of data distribution changes
- **Feature Store**: Centralized feature management and versioning
- **Data Lineage Tracking**: Complete data provenance and audit trails
- **External Data Integration**: Additional student data sources

### Infrastructure Improvements
- **Environment Management**: Improved development environment setup
- **Automated Testing**: Enhanced test automation and coverage
- **Code Quality**: Advanced linting, formatting, and static analysis
- **Performance Optimization**: Code profiling and optimization tools

### Monitoring and Operations
- **Model Performance Tracking**: Offline model performance analysis
- **Data Quality Monitoring**: Automated data validation and quality checks
- **Experiment Tracking**: Model experiment logging and comparison
- **Backup Procedures**: Data and model backup strategies
- **Performance Analysis**: Training and prediction performance optimization

### User Experience
- **Enhanced Reporting**: Improved prediction result formatting
- **Data Visualization**: Better charts and graphs for analysis
- **Export Capabilities**: Multiple output formats for predictions
- **Documentation**: Comprehensive user guides and examples
- **Jupyter Integration**: Enhanced notebook-based workflows

### Compliance and Security
- **Privacy Compliance**: GDPR/PDPA compliance implementation
- **Audit Logging**: Comprehensive audit trail for all operations
- **Data Anonymization**: Advanced privacy protection techniques
- **Security Scanning**: Automated security vulnerability assessment
- **Access Control**: Fine-grained permission management

---

## Dependencies and Critical Path

### Critical Path Analysis
1. **Setup → Data Processing → Feature Engineering → Model Development** (Sequential)
2. **Testing & Validation** runs parallel to development phases and requires completion of Model Development

### External Dependencies
- **SQLite Database**: Must contain expected 15,900 records with 17 features
- **Python Environment**: Python 3.9+ with sufficient compute resources
- **Development Tools**: Poetry, pytest, black, flake8 availability
- **Stakeholder Availability**: For requirements validation and model evaluation

### Internal Dependencies
- **Data Quality**: All subsequent phases depend on clean, validated data
- **Model Performance**: Testing and validation depend on acceptable model accuracy
- **Test Coverage**: Model validation blocked until comprehensive testing complete
- **Documentation**: Model deployment and usage depend on complete documentation

### Risk Mitigation Strategies
- **Data Quality Issues**: Comprehensive validation and multiple cleaning strategies
- **Model Performance**: Multiple algorithms and extensive hyperparameter tuning
- **Integration Challenges**: Modular design with clear interfaces and contracts
- **Timeline Risks**: Parallel development where possible, clear task priorities
- **Resource Constraints**: Scalable architecture and efficient algorithms

---

## Success Criteria Summary

### Technical Success Criteria
- ✅ **Model Accuracy**: MAE < 8 points, R² > 0.75 on test set
- ✅ **Data Pipeline**: Robust processing from raw SQLite to ML-ready features
- ✅ **Test Coverage**: >90% test coverage for all critical components
- ✅ **Performance**: Prediction generation < 2 seconds per student
- ✅ **Data Quality**: <1% data quality issues in processed dataset

### Business Success Criteria
- ✅ **Early Warning Capability**: 90% sensitivity for identifying bottom 20% performers
- ✅ **Intervention Framework**: System enables targeted support allocation
- ✅ **Teacher Adoption**: User-friendly interface with actionable insights
- ✅ **Scalability**: System handles current data size with 50% growth capacity
- ✅ **ROI Demonstration**: Clear metrics showing intervention effectiveness
- ✅ **Stakeholder Satisfaction**: Positive feedback from teachers and administrators

### Operational Success Criteria
- ✅ **Model Validation**: Comprehensive testing and validation of ML models
- ✅ **Code Quality**: High-quality, maintainable, and well-documented code
- ✅ **Documentation Quality**: Comprehensive technical and user documentation
- ✅ **Reproducibility**: Consistent and reproducible model training and evaluation
- ✅ **Data Integrity**: Robust data validation and quality assurance processes
- ✅ **Performance Standards**: Efficient model training and prediction generation

---

## Task Tracking Guidelines

### Status Update Process
1. **Daily Updates**: Update task status during daily development
2. **Weekly Reviews**: Comprehensive review of phase progress
3. **Milestone Checkpoints**: Formal review at end of each phase
4. **Blocker Resolution**: Immediate escalation of blocked tasks

### Quality Gates
- **Phase Completion**: All tasks in phase must be completed and tested
- **Test Requirements**: All specified tests must pass before marking complete
- **Documentation**: Required documentation must be updated
- **Review Process**: Code review and approval for critical components

### Discovery Process
- **New Task Identification**: Document discovered tasks immediately
- **Impact Assessment**: Evaluate impact on timeline and dependencies
- **Prioritization**: Classify as current phase, next phase, or backlog
- **Stakeholder Communication**: Notify relevant stakeholders of significant discoveries

---

*This document is a living document that will be updated as the project progresses and new tasks are discovered. Regular reviews and updates ensure it remains an accurate reflection of project status and requirements.*