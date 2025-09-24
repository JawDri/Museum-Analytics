#!/usr/bin/env python3
"""
Test script to validate the museum analytics solution.
Creates sample data and tests the pipeline components.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'museum_analytics', 'src'))

def create_sample_data():
    """Create sample data for testing."""
    print("Creating sample data for testing...")
    
    # Create directories
    data_dir = Path("data")
    models_dir = Path("models")
    plots_dir = Path("plots")
    
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    # Sample museum data
    museum_data = [
        {
            'museum_name': 'Louvre',
            'city': 'Paris',
            'country': 'France',
            'annual_visitors': 9_600_000,
            'year': 2023
        },
        {
            'museum_name': 'National Museum of China',
            'city': 'Beijing',
            'country': 'China',
            'annual_visitors': 7_500_000,
            'year': 2023
        },
        {
            'museum_name': 'Metropolitan Museum of Art',
            'city': 'New York',
            'country': 'United States',
            'annual_visitors': 6_200_000,
            'year': 2023
        },
        {
            'museum_name': 'British Museum',
            'city': 'London',
            'country': 'United Kingdom',
            'annual_visitors': 5_800_000,
            'year': 2023
        },
        {
            'museum_name': 'Tate Modern',
            'city': 'London',
            'country': 'United Kingdom',
            'annual_visitors': 5_700_000,
            'year': 2023
        },
        {
            'museum_name': 'National Gallery',
            'city': 'London',
            'country': 'United Kingdom',
            'annual_visitors': 5_200_000,
            'year': 2023
        },
        {
            'museum_name': 'Vatican Museums',
            'city': 'Rome',
            'country': 'Italy',
            'annual_visitors': 5_000_000,
            'year': 2023
        },
        {
            'museum_name': 'Hermitage Museum',
            'city': 'Saint Petersburg',
            'country': 'Russia',
            'annual_visitors': 4_200_000,
            'year': 2023
        },
        {
            'museum_name': 'Prado Museum',
            'city': 'Madrid',
            'country': 'Spain',
            'annual_visitors': 3_500_000,
            'year': 2023
        },
        {
            'museum_name': 'Uffizi Gallery',
            'city': 'Florence',
            'country': 'Italy',
            'annual_visitors': 3_200_000,
            'year': 2023
        }
    ]
    
    museum_df = pd.DataFrame(museum_data)
    museum_df.to_csv(data_dir / "museum_data.csv", index=False)
    print(f"Created sample museum data: {len(museum_data)} museums")
    
    # Sample population data
    population_data = [
        {'city': 'Paris', 'population': 11_000_000},
        {'city': 'Beijing', 'population': 21_500_000},
        {'city': 'New York', 'population': 8_400_000},
        {'city': 'London', 'population': 9_000_000},
        {'city': 'Rome', 'population': 4_300_000},
        {'city': 'Saint Petersburg', 'population': 5_400_000},
        {'city': 'Madrid', 'population': 6_700_000},
        {'city': 'Florence', 'population': 1_500_000}
    ]
    
    population_df = pd.DataFrame(population_data)
    population_df.to_csv(data_dir / "city_population.csv", index=False)
    print(f"Created sample population data: {len(population_data)} cities")
    
    return museum_df, population_df

def test_harmonizer():
    """Test the data harmonizer."""
    print("\nTesting data harmonizer...")
    
    try:
        from data_processing.harmonizer import DataHarmonizer
        
        harmonizer = DataHarmonizer("data/museum_analytics.db")
        harmonized_df = harmonizer.harmonize_data("data/museum_data.csv", "data/city_population.csv")
        
        if not harmonized_df.empty:
            print(f"✓ Data harmonization successful: {len(harmonized_df)} records")
            print("Sample harmonized data:")

            # ---- NEW: resolve column aliases ----
            city_col = 'city' if 'city' in harmonized_df.columns else (
                'city_museum' if 'city_museum' in harmonized_df.columns else None
            )
            pop_col = 'city_population' if 'city_population' in harmonized_df.columns else (
                'population' if 'population' in harmonized_df.columns else None
            )

            if city_col is None or pop_col is None:
                missing = []
                if city_col is None: missing.append("city/city_museum")
                if pop_col is None:  missing.append("city_population/population")
                raise KeyError(f"Missing expected columns: {', '.join(missing)}. "
                               f"Available: {list(harmonized_df.columns)}")

            # show a tidy preview with consistent column names
            preview = harmonized_df[['museum_name', city_col, 'annual_visitors', pop_col]].head().copy()
            preview.columns = ['museum_name', 'city', 'annual_visitors', 'city_population']
            print(preview.to_string(index=False))

            return harmonized_df
        else:
            print("✗ Data harmonization failed")
            return None
            
    except Exception as e:
        print(f"✗ Error testing harmonizer: {e}")
        return None

def test_regression_model(harmonized_df):
    """Test the regression model."""
    print("\nTesting regression model...")
    
    try:
        from models.regression_model import MuseumRegressionModel
        
        model = MuseumRegressionModel("models/museum_regression_model.pkl")
        metrics = model.train("data/harmonized_museum_data.csv")
        
        if 'error' not in metrics:
            print("✓ Model training successful!")
            print(f"  - Training R²: {metrics['train_r2']:.4f}")
            print(f"  - Test R²: {metrics['test_r2']:.4f}")
            print(f"  - Correlation: {metrics['correlation']:.4f}")
            print(f"  - Test MSE: {metrics['test_mse']:,.2f}")
            
            # Test prediction
            test_population = 5_000_000
            prediction = model.predict(test_population)
            print(f"  - Prediction for {test_population:,} population: {prediction:,.0f} visitors")
            
            return True
        else:
            print(f"✗ Model training failed: {metrics['error']}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing model: {e}")
        return False

def test_database_manager():
    """Test the database manager."""
    print("\nTesting database manager...")
    
    try:
        from database.db_manager import DatabaseManager
        
        db_manager = DatabaseManager("data/museum_analytics.db")
        
        # Get database info
        info = db_manager.get_database_info()
        print(f"✓ Database contains {info['table_counts']['harmonized_data']} harmonized records")
        
        # Get top museums
        top_museums = db_manager.get_top_museums(3)
        print("Top 3 museums:")
        for _, row in top_museums.iterrows():
            print(f"  - {row['museum_name']}: {row['annual_visitors']:,} visitors")
        
        db_manager.close()
        return True
        
    except Exception as e:
        print(f"✗ Error testing database: {e}")
        return False

def main():
    """Main test function."""
    print("MUSEUM ANALYTICS - SOLUTION VALIDATION")
    print("=" * 50)
    
    # Create sample data
    museum_df, population_df = create_sample_data()
    
    # Test harmonizer
    harmonized_df = test_harmonizer()
    
    if harmonized_df is not None:
        # Test regression model
        model_success = test_regression_model(harmonized_df)
        
        # Test database manager
        db_success = test_database_manager()
        
        # Summary
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY:")
        print(f"✓ Sample data created: {len(museum_df)} museums, {len(population_df)} cities")
        print(f"✓ Data harmonization: {'PASS' if harmonized_df is not None else 'FAIL'}")
        print(f"✓ Model training: {'PASS' if model_success else 'FAIL'}")
        print(f"✓ Database operations: {'PASS' if db_success else 'FAIL'}")
        
        if harmonized_df is not None and model_success and db_success:
            print("\n🎉 ALL TESTS PASSED! The solution is working correctly.")
            print("\nTo run the complete pipeline:")
            print("1. docker-compose up --build")
            print("2. Access Jupyter Lab at http://localhost:8888")
            print("3. Open museum_analytics/notebooks/museum_analysis.ipynb")
        else:
            print("\n⚠️  Some tests failed. Check the error messages above.")
    else:
        print("\n✗ Critical failure: Data harmonization failed")

if __name__ == "__main__":
    main()