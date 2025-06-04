# Database Structure Analysis: score.db

## Overview
- **Database Type**: SQLite
- **File Size**: 1,601,536 bytes (1.53 MB)
- **Total Tables**: 1
- **Total Records**: 15,900

## Database Schema Analysis

### Table: `score`
**Purpose**: Student academic and demographic data for educational analysis

#### Column Structure
| Column Name | Data Type | Nullable | Primary Key | Description |
|-------------|-----------|----------|-------------|-------------|
| `index` | INTEGER | Yes | No | Row identifier/index |
| `number_of_siblings` | INTEGER | Yes | No | Number of siblings student has |
| `direct_admission` | TEXT | Yes | No | Whether student got direct admission |
| `CCA` | TEXT | Yes | No | Co-curricular activity participation |
| `learning_style` | TEXT | Yes | No | Student's learning style preference |
| `student_id` | TEXT | Yes | No | Unique student identifier |
| `gender` | TEXT | Yes | No | Student gender |
| `tuition` | TEXT | Yes | No | Whether student receives tuition |
| `final_test` | REAL | Yes | No | Final test score (target variable) |
| `n_male` | REAL | Yes | No | Number of male students in class/group |
| `n_female` | REAL | Yes | No | Number of female students in class/group |
| `age` | REAL | Yes | No | Student age |
| `hours_per_week` | REAL | Yes | No | Study hours per week |
| `attendance_rate` | REAL | Yes | No | Attendance percentage |
| `sleep_time` | TEXT | Yes | No | Student's sleep time |
| `wake_time` | TEXT | Yes | No | Student's wake time |
| `mode_of_transport` | TEXT | Yes | No | Transportation method to school |
| `bag_color` | TEXT | Yes | No | Student's bag color |

#### Indexes
- `ix_score_index`: Non-unique index on the `index` column

#### Constraints
- No primary keys defined
- No foreign key relationships
- No NOT NULL constraints
- No default values

## Data Content Overview

### Sample Data (First 5 Rows)
```
index | siblings | admission | CCA    | style   | student_id | gender | tuition | final_test | n_male | n_female | age | hours | attendance | sleep | wake | transport | bag_color
0     | 0        | Yes       | Sports | Visual  | ACN2BE     | Female | No      | 69.0       | 14.0   | 2.0      | 16.0| 10.0  | 91.0       | 22:00 | 6:00 | private   | yellow
1     | 2        | No        | Sports | Auditory| FGXIIZ     | Female | No      | 47.0       | 4.0    | 19.0     | 16.0| 7.0   | 94.0       | 22:30 | 6:30 | private   | green
2     | 0        | Yes       | None   | Visual  | B9AI9F     | Male   | No      | 85.0       | 14.0   | 2.0      | 15.0| 8.0   | 92.0       | 22:30 | 6:30 | private   | white
```

### Data Quality Issues

#### Missing Values
| Column | Missing Count | Percentage |
|--------|---------------|------------|
| `final_test` | 495 | 3.1% |
| `attendance_rate` | 778 | 4.9% |
| All other columns | 0 | 0.0% |

#### Data Inconsistencies
1. **Tuition field**: Mixed formats ('Yes'/'No' vs 'Y'/'N')
2. **CCA field**: Case inconsistency ('Clubs' vs 'CLUBS')
3. **Age field**: Contains negative values (min: -5.0) - data quality issue
4. **Duplicate records**: 139 duplicate rows (excluding index column)

### Statistical Summary

#### Numeric Columns
| Column | Min | Max | Average | Distinct Values |
|--------|-----|-----|---------|-----------------|
| `number_of_siblings` | 0 | 2 | 0.89 | 3 |
| `final_test` | 32.0 | 100.0 | 67.17 | 68 |
| `n_male` | 0.0 | 31.0 | 13.88 | 32 |
| `n_female` | 0.0 | 31.0 | 8.91 | 32 |
| `age` | -5.0 | 16.0 | 15.21 | 6 |
| `hours_per_week` | 0.0 | 20.0 | 10.31 | 21 |
| `attendance_rate` | 40.0 | 100.0 | 93.27 | 61 |

#### Categorical Columns
| Column | Unique Values | Top Categories |
|--------|---------------|----------------|
| `direct_admission` | 2 | No (11,195), Yes (4,705) |
| `CCA` | 8 | Clubs (3,912), Sports (3,865), None (3,829), Arts (3,785) |
| `learning_style` | 2 | Auditory (9,132), Visual (6,768) |
| `student_id` | 15,000 | Mostly unique (some duplicates) |
| `gender` | 2 | Male (7,984), Female (7,916) |
| `tuition` | 4 | Yes (8,669), No (6,643), Y (327), N (261) |
| `sleep_time` | 13 | 23:00 (3,131), 22:00 (3,067), 22:30 (3,034) |
| `wake_time` | 5 | 5:00 (3,246), 7:00 (3,206), 6:00 (3,165) |
| `mode_of_transport` | 3 | public (6,371), private (6,323), walk (3,206) |
| `bag_color` | 6 | yellow (2,731), green (2,653), black (2,650) |

## Recommendations for Next Steps

### 1. Exploratory Data Analysis (EDA)

#### Distribution Analysis
- **Final Test Scores**: Analyze distribution, identify outliers, check for normality
- **Age Distribution**: Investigate negative age values and overall age patterns
- **Study Hours vs Performance**: Correlation analysis between hours_per_week and final_test
- **Attendance Impact**: Relationship between attendance_rate and academic performance

#### Correlation Analysis
- Numeric variables correlation matrix (final_test, age, hours_per_week, attendance_rate, n_male, n_female)
- Gender balance impact (n_male vs n_female) on performance
- Family size effect (number_of_siblings) on academic outcomes

#### Categorical Analysis
- Performance differences by gender, learning_style, CCA participation
- Direct admission impact on final test scores
- Transportation mode and academic performance relationship
- Sleep/wake time patterns and their correlation with performance

### 2. Data Cleaning

#### Critical Issues to Address
1. **Age Data**: Fix negative age values (-5.0 minimum) - likely data entry errors
2. **Tuition Standardization**: Convert 'Y'/'N' to 'Yes'/'No' format for consistency
3. **CCA Case Normalization**: Standardize 'CLUBS' to 'Clubs' format
4. **Duplicate Removal**: Handle 139 duplicate records appropriately
5. **Missing Value Treatment**: 
   - final_test: 495 missing (3.1%) - consider imputation or exclusion
   - attendance_rate: 778 missing (4.9%) - investigate pattern and impute

#### Data Validation
- Verify student_id uniqueness (currently 15,000 unique IDs for 15,900 records)
- Check logical consistency between sleep_time and wake_time
- Validate age ranges (should be positive, reasonable for students)

### 3. Data Preprocessing

#### Feature Engineering Opportunities
1. **Sleep Duration**: Calculate from sleep_time and wake_time
2. **Class Gender Ratio**: Create ratio of n_male/(n_male + n_female)
3. **Time Categories**: Group sleep/wake times into categories (early/late sleepers)
4. **Performance Categories**: Bin final_test scores into performance levels
5. **Study Intensity**: Combine hours_per_week and attendance_rate

#### Encoding Requirements
- One-hot encoding for categorical variables (gender, CCA, learning_style, etc.)
- Ordinal encoding for ordered categories if applicable
- Time feature extraction from sleep_time/wake_time

#### Normalization Needs
- Scale numeric features (final_test, hours_per_week, attendance_rate)
- Handle class imbalance in categorical variables if needed

### 4. Technical Considerations

#### Performance Notes
- Database size (1.53 MB) is manageable for most analysis tools
- 15,900 records suitable for machine learning without sampling
- Single table structure simplifies analysis pipeline

#### Recommended Tools
- **Python**: pandas, numpy, scikit-learn for analysis and modeling
- **Visualization**: matplotlib, seaborn, plotly for EDA
- **Statistical Analysis**: scipy.stats for hypothesis testing
- **Machine Learning**: scikit-learn, xgboost for predictive modeling

#### Potential Challenges
1. **Missing Data**: Strategy needed for final_test and attendance_rate nulls
2. **Data Quality**: Age validation and duplicate handling required
3. **Feature Selection**: Many categorical variables may need dimensionality reduction
4. **Target Variable**: final_test appears to be the main outcome variable for prediction

#### Next Analysis Steps
1. Load data into pandas DataFrame for detailed EDA
2. Create comprehensive visualization dashboard
3. Perform statistical tests for significant relationships
4. Build baseline predictive models for final_test scores
5. Feature importance analysis to identify key predictors
