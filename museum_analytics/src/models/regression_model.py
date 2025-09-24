"""
Linear regression model for museum visitor and city population correlation.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Tiny toggles you can flip without changing the CLI ----
REMOVE_OUTLIERS = False        # was True; set False while dataset is small
USE_LOG_TRANSFORM = True       # fit on log1p(pop), log1p(visitors) and invert


class MuseumRegressionModel:
    """Linear regression model for museum visitor and city population correlation."""
    
    def __init__(self, model_path: str = "museum_regression_model.pkl"):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.is_trained = False
        # canonical names used internally
        self.feature_columns = ['city_population']
        self.target_column = 'annual_visitors'
        # allowed aliases coming from harmonization
        self._feature_aliases = ['city_population', 'population', 'city_pop']

    # --- normalize/rename the population column once ---
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'city_population' in df.columns:
            return df
        for alias in self._feature_aliases:
            if alias in df.columns:
                if alias != 'city_population':
                    df = df.rename(columns={alias: 'city_population'})
                    logger.info(f"Renamed '{alias}' -> 'city_population'")
                return df
        logger.error(f"No population column found. Available columns: {list(df.columns)}")
        return df

    def load_data(self, data_file: str) -> pd.DataFrame:
        """Load harmonized data for training."""
        try:
            df = pd.read_csv(data_file)
            logger.info(f"Loaded {len(df)} records from {data_file}")
            # normalize columns to expected names
            df = self._normalize_columns(df)
            # coerce types to numeric where relevant
            if 'city_population' in df.columns:
                df['city_population'] = pd.to_numeric(df['city_population'], errors='coerce')
            if 'annual_visitors' in df.columns:
                df['annual_visitors'] = pd.to_numeric(df['annual_visitors'], errors='coerce')
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training by removing missing values and (optionally) outliers."""
        if df.empty:
            logger.error("Empty dataframe provided")
            return pd.DataFrame(), pd.Series(dtype=float)
        
        # Remove rows with missing population/visitors
        df_clean = df.dropna(subset=['city_population', 'annual_visitors'])
        logger.info(f"Removed {len(df) - len(df_clean)} rows with missing data")
        
        if df_clean.empty:
            logger.error("No rows left after dropping missing values")
            return pd.DataFrame(), pd.Series(dtype=float)

        if REMOVE_OUTLIERS:
            # Remove outliers using IQR method on the target
            Q1 = df_clean['annual_visitors'].quantile(0.25)
            Q3 = df_clean['annual_visitors'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean = df_clean[
                (df_clean['annual_visitors'] >= lower_bound) & 
                (df_clean['annual_visitors'] <= upper_bound)
            ]
            logger.info(f"Removed outliers, {len(df_clean)} records remaining")
        else:
            logger.info(f"Outlier removal disabled; using {len(df_clean)} records")

        if df_clean.empty:
            logger.error("No rows left after filtering")
            return pd.DataFrame(), pd.Series(dtype=float)

        # Prepare features and target
        X = df_clean[self.feature_columns].copy()
        y = df_clean[self.target_column].copy()

        # Optional log1p transform for a more linear relation on counts
        if USE_LOG_TRANSFORM:
            X['city_population'] = np.log1p(X['city_population'])
            y = np.log1p(y)
            logger.info("Applied log1p transform to features and target")

        return X, y
    
    def train(self, data_file: str, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """Train the linear regression model."""
        logger.info("Starting model training")
        
        # Load and prepare data
        df = self.load_data(data_file)
        if df.empty:
            return {'error': 'Failed to load data'}
        
        X, y = self.prepare_data(df)
        if X.empty or y.empty:
            return {'error': 'Failed to prepare data'}
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features (even if log-transformed)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Predictions (in transformed space if using logs)
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        # Report metrics both in transformed space and original space for interpretability
        def _to_original(z):
            return np.expm1(z) if USE_LOG_TRANSFORM else z

        train_mse = mean_squared_error(_to_original(y_train), _to_original(y_train_pred))
        test_mse  = mean_squared_error(_to_original(y_test),  _to_original(y_test_pred))
        train_mae = mean_absolute_error(_to_original(y_train), _to_original(y_train_pred))
        test_mae  = mean_absolute_error(_to_original(y_test),  _to_original(y_test_pred))

        # R² is computed in transformed space when logs are used (common choice)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2  = r2_score(y_test,  y_test_pred)

        # Correlation is computed on the (possibly transformed) train split
        feat = self.feature_columns[0]
        correlation = float(np.corrcoef(X_train[feat], y_train)[0, 1])
        
        metrics = {
            'train_mse': float(train_mse),
            'test_mse': float(test_mse),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'correlation': float(correlation),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'coefficients': self.model.coef_.tolist(),
            'intercept': float(self.model.intercept_),
            'log_transform': USE_LOG_TRANSFORM
        }
        
        logger.info(f"Model training completed. Test R²: {test_r2:.4f}, Correlation: {correlation:.4f}")
        
        # Save model
        self.save_model()
        
        return metrics
    
    def predict(self, city_population: float) -> float:
        """Make prediction for a given city population (returns visitors in original units)."""
        if not self.is_trained:
            logger.error("Model not trained yet")
            return 0.0
        
        # Use a DataFrame with the correct column name to avoid sklearn warning
        x = city_population
        x_feat = np.log1p(x) if USE_LOG_TRANSFORM else float(x)
        X = pd.DataFrame({'city_population': [x_feat]})
        X_scaled = self.scaler.transform(X)
        y_hat = self.model.predict(X_scaled)[0]
        y_hat = np.expm1(y_hat) if USE_LOG_TRANSFORM else y_hat
        return float(y_hat)
    
    def save_model(self):
        """Save the trained model and scaler."""
        if not self.is_trained:
            logger.warning("Model not trained, cannot save")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'use_log_transform': USE_LOG_TRANSFORM
        }
        
        joblib.dump(model_data, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a previously trained model."""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            # keep current USE_LOG_TRANSFORM toggle if not present in saved file
            self.is_trained = True
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def create_visualizations(self, data_file: str, output_dir: str = "plots"):
        """Create visualization plots for the model."""
        if not self.is_trained:
            logger.error("Model not trained yet")
            return
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load data and prepare the same way as training (to keep transforms consistent)
        df = self.load_data(data_file)
        if df.empty:
            return
        
        X, y = self.prepare_data(df)
        if X.empty or y.empty:
            return
        
        # Predictions over all prepared rows
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        # For plots, show points in *original* units
        x_plot = X['city_population'].copy()
        y_plot = y.copy()
        y_pred_plot = pd.Series(y_pred, index=y.index)

        if USE_LOG_TRANSFORM:
            x_plot = np.expm1(x_plot)
            y_plot = np.expm1(y_plot)
            y_pred_plot = np.expm1(y_pred_plot)

        plt.style.use('seaborn-v0_8')
        
        # 1) Scatter with regression line (sorted by x for a smooth line)
        plt.figure(figsize=(12, 8))
        order = np.argsort(x_plot.values)
        plt.scatter(x_plot, y_plot, alpha=0.6, label='Actual')
        plt.plot(x_plot.values[order], y_pred_plot.values[order], linewidth=2, label='Regression Fit')
        plt.xlabel('City Population')
        plt.ylabel('Annual Visitors')
        title_suffix = " (log–log fit)" if USE_LOG_TRANSFORM else ""
        plt.title(f'Museum Visitors vs City Population{title_suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/regression_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2) Residual plot (original units)
        residuals = y_plot - y_pred_plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred_plot, residuals, alpha=0.6)
        plt.axhline(y=0, linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/residual_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3) Histogram of residuals
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/residual_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}/")
    
    def get_model_summary(self) -> Dict:
        """Get model summary information in *original* units."""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        a = float(self.model.coef_[0])
        b = float(self.model.intercept_)
        mu = float(self.scaler.mean_[0])
        sigma = float(self.scaler.scale_[0])

        # If using log transform, the linear model is in log space:
        # log1p(y) = a * ((z - mu)/sigma) + b  with z = log1p(x)
        # => y ≈ exp( a*( (log1p(x)-mu)/sigma ) + b ) - 1
        # There is no exact y = m*x + c in original units; we present the linearization around mean x.
        if USE_LOG_TRANSFORM:
            # give a local linear approximation slope at the mean (optional)
            # dy/dx at x=E[x] is approximately: (a/sigma) * (1 / (1+E[x])) * exp(b - a*mu/sigma)
            x_mean = np.expm1(mu)  # inverse of log1p
            slope_local = (a / sigma) * (1.0 / (1.0 + x_mean)) * np.exp(b - a * mu / sigma)
            intercept_local = np.expm1(b - a * mu / sigma) - slope_local * x_mean
            eq_text = (
                "log1p(visitors) = "
                f"{a:.4f} * z_scaled + {b:.4f}  (z = log1p(pop))\n"
                f"Approx. local linearization: visitors ≈ {slope_local:.4f} * population + {intercept_local:.2f}"
            )
            return {
                'model_type': 'Linear Regression (log–log space)',
                'feature': 'city_population (log1p)',
                'target': 'annual_visitors (log1p)',
                'coef_log_space': a,
                'intercept_log_space': b,
                'equation_log_space': f"log1p(visitors) = {a:.6f} * z_scaled + {b:.6f}",
                'approx_equation_original_space': f"visitors ≈ {slope_local:.4f} * population + {intercept_local:.2f}"
            }
        else:
            # Undo StandardScaler to express y = m*x + c in original x units
            # y = a * ((x - mu)/sigma) + b  =>  y = (a/sigma)*x + (b - a*mu/sigma)
            m_unscaled = a / sigma
            b_unscaled = b - a * mu / sigma
            return {
                'model_type': 'Linear Regression',
                'feature': 'city_population',
                'target': 'annual_visitors',
                'coefficient': m_unscaled,
                'intercept': b_unscaled,
                'equation': f"visitors = {m_unscaled:.2f} * population + {b_unscaled:.2f}"
            }


def main():
    """Main function to run the model training."""
    model = MuseumRegressionModel()
    
    # Check if harmonized data exists
    data_file = "harmonized_museum_data.csv"
    if not Path(data_file).exists():
        print("Please run the data harmonization script first to generate input data")
        return
    
    # Train model
    metrics = model.train(data_file)
    
    if 'error' in metrics:
        print(f"Error: {metrics['error']}")
        return
    
    print("Model Training Results:")
    print(f"Training R²: {metrics['train_r2']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print(f"Correlation: {metrics['correlation']:.4f}")
    print(f"Training MSE: {metrics['train_mse']:,.2f}")
    print(f"Test MSE: {metrics['test_mse']:,.2f}")
    print(f"Training MAE: {metrics['train_mae']:,.2f}")
    print(f"Test MAE: {metrics['test_mae']:,.2f}")
    
    # Get model summary
    summary = model.get_model_summary()
    if 'equation' in summary:
        print(f"\nModel Equation: {summary['equation']}")
    elif 'approx_equation_original_space' in summary:
        print(f"\n{summary['equation_log_space']}")
        print(f"Approx. original-space: {summary['approx_equation_original_space']}")
    
    # Create visualizations
    model.create_visualizations(data_file)
    print("\nVisualizations created in 'plots/' directory")
    
    # Test prediction
    test_population = 5_000_000
    prediction = model.predict(test_population)
    print(f"\nPrediction for city with {test_population:,} population: {prediction:,.0f} visitors")


if __name__ == "__main__":
    main()
