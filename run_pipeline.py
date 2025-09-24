#!/usr/bin/env python3
"""
Main script to run the complete museum analytics pipeline.
This script orchestrates the entire data engineering workflow.
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'museum_analytics', 'src'))

from data_extraction.wikipedia_scraper import WikipediaMuseumScraper
from data_extraction.population_data import PopulationDataExtractor
from data_processing.harmonizer import DataHarmonizer
from models.regression_model import MuseumRegressionModel
from database.db_manager import DatabaseManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the complete pipeline."""
    logger.info("Starting Museum Analytics Pipeline")
    logger.info("=" * 50)
    
    # Create necessary directories
    data_dir = Path("data")
    models_dir = Path("models")
    plots_dir = Path("plots")
    
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Extract museum data from Wikipedia
        logger.info("Step 1: Extracting museum data from Wikipedia...")
        museum_scraper = WikipediaMuseumScraper()
        museums = museum_scraper.extract_museum_data()
        
        if not museums:
            logger.error("Failed to extract museum data")
            return False
        
        museum_file = data_dir / "museum_data.csv"
        museum_df = museum_scraper.save_to_csv(museums, str(museum_file))
        logger.info(f"Successfully extracted {len(museums)} museums")
        
        # Step 2: Extract population data
        logger.info("Step 2: Extracting city population data...")
        population_extractor = PopulationDataExtractor()
        
        # Get unique cities from museum data
        unique_cities = list(set([museum['city'] for museum in museums]))
        population_data = population_extractor.get_city_population_data(unique_cities)
        
        if not population_data:
            logger.error("Failed to extract population data")
            return False
        
        population_file = data_dir / "city_population.csv"
        population_df = population_extractor.save_to_csv(population_data, str(population_file))
        logger.info(f"Successfully extracted population data for {len(population_data)} cities")
        
        # Step 3: Harmonize data
        logger.info("Step 3: Harmonizing museum and population data...")
        db_path = data_dir / "museum_analytics.db"
        harmonizer = DataHarmonizer(str(db_path))
        harmonized_df = harmonizer.harmonize_data(str(museum_file), str(population_file))
        
        if harmonized_df.empty:
            logger.error("Failed to harmonize data")
            return False
        
        harmonized_file = data_dir / "harmonized_museum_data.csv"
        harmonized_df.to_csv(harmonized_file, index=False)
        logger.info(f"Successfully harmonized {len(harmonized_df)} museum records")
        
        # Step 4: Train regression model
        logger.info("Step 4: Training linear regression model...")
        model_path = models_dir / "museum_regression_model.pkl"
        model = MuseumRegressionModel(str(model_path))
        
        metrics = model.train(str(harmonized_file))
        
        if 'error' in metrics:
            logger.error(f"Model training failed: {metrics['error']}")
            return False
        
        logger.info("Model training completed successfully!")
        logger.info(f"Training R²: {metrics['train_r2']:.4f}")
        logger.info(f"Test R²: {metrics['test_r2']:.4f}")
        logger.info(f"Correlation: {metrics['correlation']:.4f}")
        logger.info(f"Test MSE: {metrics['test_mse']:,.2f}")
        logger.info(f"Test MAE: {metrics['test_mae']:,.2f}")
        
        # Step 5: Create visualizations
        logger.info("Step 5: Creating visualizations...")
        model.create_visualizations(str(harmonized_file), str(plots_dir))
        logger.info(f"Visualizations saved to {plots_dir}/")
        
        # Step 6: Database analysis
        logger.info("Step 6: Performing database analysis...")
        db_manager = DatabaseManager(str(db_path))
        
        try:
            # Get database info
            info = db_manager.get_database_info()
            logger.info(f"Database contains {info['table_counts']['harmonized_data']} harmonized records")
            
            # Get correlation stats
            stats = db_manager.get_correlation_stats()
            logger.info(f"Average visitors: {stats['avg_visitors']:,.0f}")
            logger.info(f"Average city population: {stats['avg_population']:,.0f}")
            logger.info(f"Average visitor-population ratio: {stats['avg_ratio']:.4f}")
            
        except Exception as e:
            logger.warning(f"Database analysis failed: {e}")
        finally:
            db_manager.close()
        
        # Step 7: Generate insights
        logger.info("Step 7: Generating insights...")
        generate_insights(harmonized_df, metrics)
        
        logger.info("=" * 50)
        logger.info("Pipeline completed successfully!")
        logger.info("Files created:")
        logger.info(f"  - Museum data: {museum_file}")
        logger.info(f"  - Population data: {population_file}")
        logger.info(f"  - Harmonized data: {harmonized_file}")
        logger.info(f"  - Database: {db_path}")
        logger.info(f"  - Model: {model_path}")
        logger.info(f"  - Visualizations: {plots_dir}/")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return False


def generate_insights(harmonized_df, metrics):
    """Generate insights from the analysis."""
    logger.info("INSIGHTS AND RECOMMENDATIONS:")
    logger.info("-" * 30)

    # Resolve population column alias
    pop_col = None
    for cand in ['city_population', 'population', 'city_pop']:
        if cand in harmonized_df.columns:
            pop_col = cand
            break
    if pop_col is None:
        raise KeyError(
            "No population column found in harmonized data. "
            f"Columns present: {list(harmonized_df.columns)}"
        )

    # Resolve city column alias
    city_col = None
    for cand in ['city', 'city_museum', 'city_name']:
        if cand in harmonized_df.columns:
            city_col = cand
            break

    # 1) Data quality insights
    total_museums = len(harmonized_df)
    museums_with_population = len(harmonized_df.dropna(subset=[pop_col]))
    data_coverage = (museums_with_population / total_museums) * 100 if total_museums else 0.0
    logger.info(f"1. Data Coverage: {data_coverage:.1f}% of museums have population data")

    # 2) Correlation insights
    correlation = metrics.get('correlation', float('nan'))
    r2_score_val = metrics.get('test_r2', float('nan'))

    logger.info("2. Correlation Analysis:")
    logger.info(f"   - Correlation coefficient: {correlation:.4f}")
    logger.info(f"   - R² score: {r2_score_val:.4f}")

    if correlation > 0.7:
        correlation_strength = "strong"
    elif correlation > 0.5:
        correlation_strength = "moderate"
    elif correlation > 0.3:
        correlation_strength = "weak"
    else:
        correlation_strength = "very weak"
    logger.info(f"   - {correlation_strength.title()} positive correlation between city size and museum visitors")

    # 3) Model performance insights
    logger.info("3. Model Performance:")
    test_mse = metrics.get('test_mse', float('nan'))
    rmse = (test_mse ** 0.5) if not np.isnan(test_mse) else float('nan')
    logger.info(f"   - Model explains {r2_score_val*100:.1f}% of the variance in museum visitors")
    logger.info(f"   - Mean Absolute Error: {metrics.get('test_mae', float('nan')):,.0f} visitors")
    logger.info(f"   - Root Mean Square Error: {rmse:,.0f} visitors")

    # 4) Business insights
    analysis_df = harmonized_df.dropna(subset=[pop_col, 'annual_visitors'])
    logger.info("4. Business Insights:")
    if not analysis_df.empty and 'visitor_population_ratio' in analysis_df.columns:
        avg_ratio = analysis_df['visitor_population_ratio'].mean()
        logger.info(f"   - Average visitor-population ratio: {avg_ratio:.4f}")
        logger.info(f"   - Museums attract approximately {avg_ratio*100:.2f}% of city population annually")

        # Top performers
        top_performers = analysis_df.nlargest(3, 'visitor_population_ratio')
        logger.info("   - Top performing museums (by ratio):")
        for _, row in top_performers.iterrows():
            city_txt = row[city_col] if city_col and city_col in row else "N/A"
            logger.info(f"     * {row['museum_name']} ({city_txt}): {row['visitor_population_ratio']:.4f}")
    else:
        logger.info("   - Not enough data to compute visitor-population ratios.")

    # 5) Recommendations
    logger.info("5. Recommendations:")
    logger.info(f"   - City population is a {correlation_strength} predictor of museum attendance")
    logger.info("   - Museums in larger cities tend to have higher visitor numbers")
    logger.info("   - Consider city size when planning museum capacity and marketing strategies")
    logger.info("   - Further analysis could include factors like tourism, cultural significance, and museum type")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)