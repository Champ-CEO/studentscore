import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Phase5ModelTraining:
    def __init__(self, processed_data_path='data/featured/', output_path='data/modeling_outputs/', random_state=42):
        self.processed_data_path = processed_data_path
        self.output_path = output_path
        self.random_state = random_state
        self.feature_definitions = {}
        self.X = None
        self.y = None

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            logging.info(f"Created directory: {self.output_path}")

    def load_data(self, train_file='train_features.csv', target_column='final_grade'):
        """Loads the feature-engineered training data."""
        try:
            train_df_path = os.path.join(self.processed_data_path, train_file)
            if not os.path.exists(train_df_path):
                logging.error(f"Training data file not found: {train_df_path}")
                raise FileNotFoundError(f"Training data file not found: {train_df_path}")
            
            train_df = pd.read_csv(train_df_path)
            logging.info(f"Successfully loaded training data from {train_df_path} with shape {train_df.shape}")
            
            if target_column not in train_df.columns:
                logging.error(f"Target column '{target_column}' not found in the training data.")
                raise ValueError(f"Target column '{target_column}' not found.")

            self.X = train_df.drop(columns=[target_column])
            self.y = train_df[target_column]
            logging.info(f"Features (X) shape: {self.X.shape}, Target (y) shape: {self.y.shape}")
            return True
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return False

    def setup_cross_validation(self, n_splits=5):
        """Sets up and demonstrates Stratified K-Fold cross-validation."""
        if self.X is None or self.y is None:
            logging.error("Data not loaded. Call load_data() first.")
            return None

        # StratifiedKFold requires discrete bins for continuous target for stratification
        # We'll create bins for 'final_grade' if it's continuous
        # This is a common approach, though for regression, regular KFold is often used.
        # If the task implies classification, y should already be discrete.
        # For regression, if stratification is strictly needed by target value ranges:
        if pd.api.types.is_numeric_dtype(self.y) and self.y.nunique() > n_splits * 2: # Heuristic for continuous
            y_binned = pd.cut(self.y, bins=n_splits, labels=False, include_lowest=True)
            logging.info(f"Target variable '{self.y.name}' is continuous. Binned for StratifiedKFold.")
        else:
            y_binned = self.y # Assume discrete or already binned
            logging.info(f"Target variable '{self.y.name}' treated as discrete for StratifiedKFold.")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        logging.info(f"Initialized StratifiedKFold with n_splits={n_splits}, random_state={self.random_state}")

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, y_binned)):
            logging.info(f"--- Fold {fold+1}/{n_splits} ---")
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            y_train_binned, y_val_binned = y_binned.iloc[train_idx], y_binned.iloc[val_idx]

            logging.info(f"Train set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")
            logging.info(f"Train target distribution (binned):\n{y_train_binned.value_counts(normalize=True).sort_index()}")
            logging.info(f"Validation target distribution (binned):\n{y_val_binned.value_counts(normalize=True).sort_index()}")

            # Placeholder: Train a dummy model
            dummy_model = DummyRegressor(strategy="mean")
            dummy_model.fit(X_train, y_train)
            y_pred_val = dummy_model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred_val)
            logging.info(f"Fold {fold+1} DummyRegressor MAE: {mae:.4f}")
            fold_metrics.append({'fold': fold+1, 'mae': mae})
        
        avg_mae = np.mean([m['mae'] for m in fold_metrics])
        logging.info(f"Average MAE across {n_splits} folds (DummyRegressor): {avg_mae:.4f}")
        
        # Save fold metrics
        metrics_df = pd.DataFrame(fold_metrics)
        metrics_path = os.path.join(self.output_path, 'cv_fold_metrics_dummy.csv')
        metrics_df.to_csv(metrics_path, index=False)
        logging.info(f"Saved CV fold metrics to {metrics_path}")

        return skf # Return the SKF object for potential reuse

    def implement_evaluation_metrics(self, y_true, y_pred, model_name="Dummy"):
        """Calculates and logs standard regression evaluation metrics."""
        if len(y_true) != len(y_pred):
            logging.error("y_true and y_pred must have the same length.")
            return None

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        logging.info(f"--- Evaluation Metrics for {model_name} ---")
        logging.info(f"Mean Absolute Error (MAE): {mae:.4f}")
        logging.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        logging.info(f"R-squared (R²): {r2:.4f}")

        metrics = {'model': model_name, 'mae': mae, 'rmse': rmse, 'r2': r2}
        
        # Save metrics
        # This could be appended to a general metrics file or saved per model
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(self.output_path, f'{model_name.lower().replace(" ", "_")}_evaluation_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        logging.info(f"Saved evaluation metrics for {model_name} to {metrics_path}")
        
        return metrics

    def train_random_forest(self, X_train, y_train, X_val, y_val, cv_folds=3):
        """Trains a RandomForestRegressor, tunes hyperparameters, and evaluates."""
        if X_train is None or y_train is None or X_val is None or y_val is None:
            logging.error("Training/validation data missing for Random Forest.")
            return None, None

        logging.info("--- Training Random Forest Regressor ---")
        rf_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)

        # Simplified GridSearchCV for demonstration
        param_grid = {
            'n_estimators': [50, 100], # Reduced for speed in demo
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }

        # Using the passed CV object if available, otherwise default KFold for GridSearchCV
        # For simplicity here, we'll use a basic CV for GridSearchCV internal folds.
        # The outer CV (from setup_cross_validation) is for overall model performance assessment.
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                                   cv=cv_folds, scoring='neg_mean_absolute_error', 
                                   verbose=1, n_jobs=-1)
        
        logging.info(f"Starting GridSearchCV for Random Forest with {cv_folds} folds...")
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_
        logging.info(f"Best RandomForest parameters: {grid_search.best_params_}")
        logging.info(f"Best CV MAE (negative): {grid_search.best_score_:.4f}")

        # Evaluate on the provided validation set (from the outer CV fold)
        y_pred_val = best_rf.predict(X_val)
        metrics = self.implement_evaluation_metrics(y_val, y_pred_val, model_name="RandomForest_Optimized")

        # Feature Importances
        if hasattr(best_rf, 'feature_importances_'):
            importances = best_rf.feature_importances_
            feature_names = X_train.columns
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
            logging.info(f"Top 10 Feature Importances (Random Forest):\n{feature_importance_df.head(10)}")
            
            # Save feature importances
            fi_path = os.path.join(self.output_path, 'random_forest_feature_importances.csv')
            feature_importance_df.to_csv(fi_path, index=False)
            logging.info(f"Saved Random Forest feature importances to {fi_path}")

        self.save_model(best_rf, "RandomForest_Optimized_Model")
        return best_rf, metrics

    def train_xgboost(self, X_train, y_train, X_val, y_val, cv_folds=3):
        """Trains an XGBoost Regressor, tunes hyperparameters, and evaluates."""
        if X_train is None or y_train is None or X_val is None or y_val is None:
            logging.error("Training/validation data missing for XGBoost.")
            return None, None
        
        # Ensure feature names are strings and do not contain special JSON characters
        # XGBoost can be sensitive to this, especially with certain versions or when saving models.
        X_train_xgb = X_train.copy()
        X_val_xgb = X_val.copy()
        X_train_xgb.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_train_xgb.columns]
        X_val_xgb.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_val_xgb.columns]

        logging.info("--- Training XGBoost Regressor ---")
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', 
                                   random_state=self.random_state, 
                                   n_jobs=-1,
                                   early_stopping_rounds=10) # Added early stopping

        # Simplified GridSearchCV for demonstration
        # For a real scenario, Bayesian Optimization (e.g., with Hyperopt or Optuna) would be better.
        param_grid = {
            'n_estimators': [50, 100], # Reduced for speed
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 1.0] # Added subsample
        }

        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                                   cv=cv_folds, scoring='neg_mean_absolute_error', 
                                   verbose=1, n_jobs=-1)
        
        logging.info(f"Starting GridSearchCV for XGBoost with {cv_folds} folds...")
        # XGBoost needs eval_set for early stopping within GridSearchCV if not handled by CV object itself
        # For simplicity, we'll rely on early_stopping_rounds in XGBRegressor constructor
        # and fit on X_train, y_train. A more robust setup would pass eval_set to fit.
        grid_search.fit(X_train_xgb, y_train) # Using X_train_xgb

        best_xgb = grid_search.best_estimator_
        logging.info(f"Best XGBoost parameters: {grid_search.best_params_}")
        logging.info(f"Best CV MAE (negative): {grid_search.best_score_:.4f}")

        # Evaluate on the provided validation set
        y_pred_val = best_xgb.predict(X_val_xgb) # Using X_val_xgb
        metrics = self.implement_evaluation_metrics(y_val, y_pred_val, model_name="XGBoost_Optimized")

        # Feature Importances (XGBoost has built-in)
        if hasattr(best_xgb, 'feature_importances_'):
            importances = best_xgb.feature_importances_
            feature_names = X_train_xgb.columns # Use sanitized names
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
            logging.info(f"Top 10 Feature Importances (XGBoost):\n{feature_importance_df.head(10)}")
            
            fi_path = os.path.join(self.output_path, 'xgboost_feature_importances.csv')
            feature_importance_df.to_csv(fi_path, index=False)
            logging.info(f"Saved XGBoost feature importances to {fi_path}")

        self.save_model(best_xgb, "XGBoost_Optimized_Model")
        return best_xgb, metrics

    def train_linear_regression(self, X_train, y_train, X_val, y_val):
        """Trains a Linear Regression model and evaluates it."""
        if X_train is None or y_train is None or X_val is None or y_val is None:
            logging.error("Training/validation data missing for Linear Regression.")
            return None, None

        logging.info("--- Training Linear Regression Baseline ---")
        
        # For a more robust baseline, consider:
        # 1. Scaling features (e.g., StandardScaler)
        # 2. PolynomialFeatures
        # 3. Regularization (Ridge, Lasso)
        # For this demo, we'll use a simple LinearRegression.
        
        # Preprocessing: Fill NaN values for Linear Regression if any exist
        # This is a simple strategy; more sophisticated imputation might be needed.
        X_train_lin = X_train.copy()
        X_val_lin = X_val.copy()

        # Identify numeric columns for imputation
        numeric_cols_train = X_train_lin.select_dtypes(include=np.number).columns
        numeric_cols_val = X_val_lin.select_dtypes(include=np.number).columns

        if X_train_lin[numeric_cols_train].isnull().any().any():
            logging.warning("NaN values found in numeric features of X_train for Linear Regression. Imputing with mean.")
            imputer_train = SimpleImputer(strategy='mean')
            X_train_lin[numeric_cols_train] = imputer_train.fit_transform(X_train_lin[numeric_cols_train])
        
        if X_val_lin[numeric_cols_val].isnull().any().any():
            logging.warning("NaN values found in numeric features of X_val for Linear Regression. Imputing with mean.")
            # Use the imputer fitted on training data if available, or fit a new one for val data if train had no NaNs
            if 'imputer_train' in locals() and imputer_train:
                 X_val_lin[numeric_cols_val] = imputer_train.transform(X_val_lin[numeric_cols_val]) 
            else:
                imputer_val = SimpleImputer(strategy='mean')
                X_val_lin[numeric_cols_val] = imputer_val.fit_transform(X_val_lin[numeric_cols_val])

        # Ensure all columns are numeric after potential OHE or other encodings from previous steps
        # Linear regression cannot handle non-numeric data directly without further encoding.
        # This check assumes previous steps handled categorical to numeric conversion.
        non_numeric_cols = X_train_lin.select_dtypes(exclude=np.number).columns
        if not non_numeric_cols.empty:
            logging.error(f"Non-numeric columns found in X_train for Linear Regression: {non_numeric_cols.tolist()}. Please ensure all features are numeric.")
            # Attempt to drop them for the sake of the demo, but this is not ideal.
            logging.warning(f"Dropping non-numeric columns for Linear Regression demo: {non_numeric_cols.tolist()}")
            X_train_lin = X_train_lin.drop(columns=non_numeric_cols)
            X_val_lin = X_val_lin.drop(columns=non_numeric_cols)
            if X_train_lin.empty:
                logging.error("No numeric features left for Linear Regression after dropping non-numeric ones.")
                return None, None

        lr_model = LinearRegression(n_jobs=-1)
        
        try:
            lr_model.fit(X_train_lin, y_train)
            logging.info("Linear Regression model trained.")
        except Exception as e:
            logging.error(f"Error training Linear Regression: {e}")
            logging.error(f"X_train_lin dtypes:\n{X_train_lin.dtypes}")
            logging.error(f"X_train_lin head:\n{X_train_lin.head()}")
            return None, None

        # Evaluate on the validation set
        y_pred_val = lr_model.predict(X_val_lin)
        metrics = self.implement_evaluation_metrics(y_val, y_pred_val, model_name="LinearRegression_Baseline")

        # Coefficients (optional logging)
        if hasattr(lr_model, 'coef_'):
            try:
                coef_df = pd.DataFrame({'feature': X_train_lin.columns, 'coefficient': lr_model.coef_})
                logging.info(f"Linear Regression Coefficients (Top 5 by absolute value):\n{coef_df.reindex(coef_df.coefficient.abs().sort_values(ascending=False).index).head(5)}")
            except Exception as e:
                logging.warning(f"Could not display coefficients: {e}")

        self.save_model(lr_model, "LinearRegression_Baseline_Model")
        return lr_model, metrics

    def train_svr(self, X_train, y_train, X_val, y_val, cv_folds=3):
        """Trains a Support Vector Regressor, tunes hyperparameters, and evaluates."""
        if X_train is None or y_train is None or X_val is None or y_val is None:
            logging.error("Training/validation data missing for SVR.")
            return None, None

        logging.info("--- Training Support Vector Regressor (SVR) ---")

        # SVR is sensitive to feature scaling. Apply StandardScaler.
        # Also, ensure all features are numeric and handle NaNs.
        X_train_svr = X_train.copy()
        X_val_svr = X_val.copy()

        # Drop non-numeric columns if any (should have been handled by prior processing)
        non_numeric_cols_train = X_train_svr.select_dtypes(exclude=np.number).columns
        if not non_numeric_cols_train.empty:
            logging.warning(f"Dropping non-numeric columns for SVR: {non_numeric_cols_train.tolist()}")
            X_train_svr = X_train_svr.drop(columns=non_numeric_cols_train)
            X_val_svr = X_val_svr.drop(columns=non_numeric_cols_train)
        
        if X_train_svr.empty:
            logging.error("No numeric features left for SVR after dropping non-numeric ones.")
            return None, None

        # Impute NaNs
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train_svr)
        X_val_imputed = imputer.transform(X_val_svr)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)

        logging.info("Features imputed and scaled for SVR.")

        svr_model = SVR()

        # Simplified GridSearchCV for demonstration
        param_grid = {
            'kernel': ['linear', 'rbf'], # Common kernels
            'C': [0.1, 1], # Reduced range for speed
            'gamma': ['scale', 'auto'] # For RBF kernel
            # 'epsilon': [0.1, 0.2] # Epsilon parameter
        }

        grid_search = GridSearchCV(estimator=svr_model, param_grid=param_grid, 
                                   cv=cv_folds, scoring='neg_mean_absolute_error', 
                                   verbose=1, n_jobs=-1)
        
        logging.info(f"Starting GridSearchCV for SVR with {cv_folds} folds...")
        try:
            grid_search.fit(X_train_scaled, y_train)
        except Exception as e:
            logging.error(f"Error during SVR GridSearchCV: {e}")
            logging.error(f"X_train_scaled shape: {X_train_scaled.shape}")
            return None, None

        best_svr = grid_search.best_estimator_
        logging.info(f"Best SVR parameters: {grid_search.best_params_}")
        logging.info(f"Best CV MAE (negative): {grid_search.best_score_:.4f}")

        # Evaluate on the scaled validation set
        y_pred_val = best_svr.predict(X_val_scaled)
        metrics = self.implement_evaluation_metrics(y_val, y_pred_val, model_name="SVR_Optimized")

        self.save_model(best_svr, "SVR_Optimized_Model")
        return best_svr, metrics

    def save_model(self, model, model_name):
        """Saves a trained model."""
        # Placeholder for model saving logic (e.g., using joblib or pickle)
        # import joblib
        # model_path = os.path.join(self.output_path, f"{model_name}.joblib")
        # joblib.dump(model, model_path)
        # logging.info(f"Model {model_name} saved to {model_path}")
        logging.info(f"Placeholder: Model {model_name} would be saved here.")

def main():
    logging.info("--- Starting Phase 5: Model Training --- ")
    
    # Define paths - adjust if your project structure is different
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Assuming src/modeling/file.py
    processed_data_path = os.path.join(base_dir, 'data', 'featured')
    modeling_output_path = os.path.join(base_dir, 'data', 'modeling_outputs')

    trainer = Phase5ModelTraining(processed_data_path=processed_data_path, output_path=modeling_output_path)

    # 1. Load data
    if not trainer.load_data(train_file='train_features.csv', target_column='final_grade'):
        logging.error("Failed to load data. Exiting.")
        return

    # 2. Setup and demonstrate Cross-Validation (Task 5.1.1)
    logging.info("--- Task 5.1.1: Setup cross-validation framework ---")
    cv_framework = trainer.setup_cross_validation(n_splits=5)
    if cv_framework:
        logging.info("Cross-validation setup demonstrated successfully.")
        # Demonstrate evaluation metrics with dummy predictions from one fold
        # In a real scenario, this would be done after training actual models
        if trainer.X is not None and trainer.y is not None:
            # Using the last fold's validation set from setup_cross_validation for demonstration
            # This is a simplified example. Proper evaluation uses all folds or a dedicated test set.
            try:
                # Re-create one split to get y_val and y_pred_val for demonstration
                # This is just for showing the metrics function, not a proper CV evaluation loop here.
                if pd.api.types.is_numeric_dtype(trainer.y) and trainer.y.nunique() > 5 * 2:
                    y_binned_demo = pd.cut(trainer.y, bins=5, labels=False, include_lowest=True)
                else:
                    y_binned_demo = trainer.y
                
                _, (train_idx_demo, val_idx_demo) = next(iter(cv_framework.split(trainer.X, y_binned_demo)))
                X_train_demo, X_val_demo = trainer.X.iloc[train_idx_demo], trainer.X.iloc[val_idx_demo]
                y_train_demo, y_val_demo = trainer.y.iloc[train_idx_demo], trainer.y.iloc[val_idx_demo]
                
                dummy_model_demo = DummyRegressor(strategy="mean")
                dummy_model_demo.fit(X_train_demo, y_train_demo)
                y_pred_demo = dummy_model_demo.predict(X_val_demo)

                logging.info("--- Task 5.1.2: Implement model evaluation metrics (Demonstration) ---")
                trainer.implement_evaluation_metrics(y_val_demo, y_pred_demo, model_name="DummyRegressor_Demo")

                # --- Task 5.2.1: Implement Random Forest Regressor (Demonstration on first fold) ---
                logging.info("--- Task 5.2.1: Implement Random Forest Regressor (Demonstration) ---")
                # Using the first fold's data (X_train_demo, y_train_demo, X_val_demo, y_val_demo)
                # In a full pipeline, this would iterate through all CV folds
                rf_model, rf_metrics = trainer.train_random_forest(X_train_demo, y_train_demo, X_val_demo, y_val_demo, cv_folds=2) # cv_folds=2 for faster demo
                if rf_model:
                    logging.info("Random Forest Regressor trained and evaluated successfully on the first fold.")
                else:
                    logging.error("Random Forest Regressor training/evaluation failed on the first fold.")

                # --- Task 5.2.2: Implement XGBoost Regressor (Demonstration on first fold) ---
                logging.info("--- Task 5.2.2: Implement XGBoost Regressor (Demonstration) ---")
                xgb_model, xgb_metrics = trainer.train_xgboost(X_train_demo, y_train_demo, X_val_demo, y_val_demo, cv_folds=2) # cv_folds=2 for faster demo
                if xgb_model:
                    logging.info("XGBoost Regressor trained and evaluated successfully on the first fold.")
                else:
                    logging.error("XGBoost Regressor training/evaluation failed on the first fold.")

                # --- Task 5.2.3: Implement Linear Regression baseline (Demonstration on first fold) ---
                logging.info("--- Task 5.2.3: Implement Linear Regression baseline (Demonstration) ---")
                lr_model, lr_metrics = trainer.train_linear_regression(X_train_demo, y_train_demo, X_val_demo, y_val_demo)
                if lr_model:
                    logging.info("Linear Regression baseline trained and evaluated successfully on the first fold.")
                else:
                    logging.error("Linear Regression baseline training/evaluation failed on the first fold.")

                # --- Task 5.2.4: Implement Support Vector Regression (Demonstration on first fold) ---
                logging.info("--- Task 5.2.4: Implement SVR (Demonstration) ---")
                svr_model, svr_metrics = trainer.train_svr(X_train_demo, y_train_demo, X_val_demo, y_val_demo, cv_folds=2) # cv_folds=2 for faster demo
                if svr_model:
                    logging.info("SVR trained and evaluated successfully on the first fold.")
                else:
                    logging.error("SVR training/evaluation failed on the first fold.")

            except Exception as e:
                logging.error(f"Error during evaluation metrics, RF, XGBoost, LR or SVR demonstration: {e}")
    else:
        logging.error("Cross-validation setup failed.")

    logging.info("--- Phase 5: Model Training script finished --- ")

if __name__ == '__main__':
    main()