"""
Data harmonizer for combining museum and population data.
Creates a unified dataset for analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataHarmonizer:
    """Harmonizes museum and population data into a unified dataset."""
    
    def __init__(self, db_path: str = "museum_analytics.db"):
        self.db_path = db_path
        self.conn = None
    
    def connect_db(self):
        """Connect to SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        logger.info(f"Connected to database: {self.db_path}")
    
    def close_db(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def create_tables(self):
        """Create database tables for museums and cities."""
        if not self.conn:
            self.connect_db()
        
        # Create museums table
        museums_sql = """
        CREATE TABLE IF NOT EXISTS museums (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            museum_name TEXT NOT NULL,
            city TEXT NOT NULL,
            country TEXT,
            annual_visitors INTEGER NOT NULL,
            year INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Create cities table
        cities_sql = """
        CREATE TABLE IF NOT EXISTS cities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city_name TEXT NOT NULL UNIQUE,
            country TEXT,
            population INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Create harmonized table
        harmonized_sql = """
        CREATE TABLE IF NOT EXISTS harmonized_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            museum_name TEXT NOT NULL,
            city TEXT NOT NULL,
            country TEXT,
            annual_visitors INTEGER NOT NULL,
            city_population INTEGER,
            visitor_population_ratio REAL,
            year INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        cursor = self.conn.cursor()
        cursor.execute(museums_sql)
        cursor.execute(cities_sql)
        cursor.execute(harmonized_sql)
        self.conn.commit()
        
        logger.info("Database tables created successfully")
    
    def load_museum_data(self, museum_file: str) -> pd.DataFrame:
        """Load museum data from CSV file."""
        try:
            df = pd.read_csv(museum_file)
            logger.info(f"Loaded {len(df)} museum records from {museum_file}")
            return df
        except Exception as e:
            logger.error(f"Error loading museum data: {e}")
            return pd.DataFrame()
    
    def load_population_data(self, population_file: str) -> pd.DataFrame:
        """Load population data from CSV file."""
        try:
            df = pd.read_csv(population_file)
            logger.info(f"Loaded {len(df)} city records from {population_file}")
            return df
        except Exception as e:
            logger.error(f"Error loading population data: {e}")
            return pd.DataFrame()
    
    def normalize_city_names(self, city_name: str) -> str:
        """Normalize city names for better matching."""
        if not city_name:
            return ""
        
        # Convert to lowercase and strip whitespace
        normalized = city_name.lower().strip()
        
        # Remove common suffixes
        suffixes_to_remove = ['city', 'metropolitan area', 'metro area', 'greater', 'greater metropolitan area']
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        
        # Handle special cases
        special_cases = {
            'new york city': 'new york',
            'mexico city': 'mexico city',
            'washington dc': 'washington',
            'washington d.c.': 'washington',
            'são paulo': 'sao paulo',
            'rio de janeiro': 'rio de janeiro',
            'buenos aires': 'buenos aires',
            'santiago de chile': 'santiago',
            'bogotá': 'bogota',
            'méxico city': 'mexico city',
            'méxico': 'mexico city'
        }
        
        return special_cases.get(normalized, normalized)
    
    def match_cities(self, museum_df: pd.DataFrame, population_df: pd.DataFrame) -> pd.DataFrame:
        """Match museum cities with population data."""
        if museum_df.empty or population_df.empty:
            logger.warning("Empty dataframes provided for matching")
            return pd.DataFrame()
        
        # Create normalized city names for matching
        museum_df['city_normalized'] = museum_df['city'].apply(self.normalize_city_names)
        population_df['city_normalized'] = population_df['city'].apply(self.normalize_city_names)
        
        # Perform the merge
        merged_df = pd.merge(
            museum_df,
            population_df,
            left_on='city_normalized',
            right_on='city_normalized',
            how='left',
            suffixes=('_museum', '_pop')
        )
        
        # Clean up the merged dataframe
        merged_df = merged_df.drop(columns=['city_normalized'])
        
        # Calculate visitor-population ratio
        merged_df['visitor_population_ratio'] = merged_df['annual_visitors'] / merged_df['population']
        
        # Handle missing population data
        missing_population = merged_df['population'].isna().sum()
        if missing_population > 0:
            logger.warning(f"Missing population data for {missing_population} cities")
        
        logger.info(f"Successfully matched {len(merged_df)} museum records")
        return merged_df
    
    def save_to_database(self, df: pd.DataFrame, table_name: str):
        """Save dataframe to database table."""
        if not self.conn:
            self.connect_db()
        
        try:
            df.to_sql(table_name, self.conn, if_exists='replace', index=False)
            logger.info(f"Saved {len(df)} records to {table_name} table")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def harmonize_data(self, museum_file: str, population_file: str) -> pd.DataFrame:
        """Main method to harmonize museum and population data."""
        logger.info("Starting data harmonization process")
        
        # Load data
        museum_df = self.load_museum_data(museum_file)
        population_df = self.load_population_data(population_file)
        
        if museum_df.empty or population_df.empty:
            logger.error("Failed to load required data files")
            return pd.DataFrame()
        
        # Match cities
        harmonized_df = self.match_cities(museum_df, population_df)
        
        if harmonized_df.empty:
            logger.error("Failed to harmonize data")
            return pd.DataFrame()
        
        # Connect to database and create tables
        self.connect_db()
        self.create_tables()
        
        # Save individual datasets
        self.save_to_database(museum_df, 'museums')
        self.save_to_database(population_df, 'cities')
        self.save_to_database(harmonized_df, 'harmonized_data')
        
        # Save harmonized data to CSV
        output_file = "harmonized_museum_data.csv"
        harmonized_df.to_csv(output_file, index=False)
        logger.info(f"Saved harmonized data to {output_file}")
        
        # Close database connection
        self.close_db()
        
        return harmonized_df
    
    def get_harmonized_data(self) -> pd.DataFrame:
        """Retrieve harmonized data from database."""
        if not self.conn:
            self.connect_db()
        
        try:
            query = "SELECT * FROM harmonized_data WHERE city_population IS NOT NULL"
            df = pd.read_sql_query(query, self.conn)
            logger.info(f"Retrieved {len(df)} harmonized records from database")
            return df
        except Exception as e:
            logger.error(f"Error retrieving harmonized data: {e}")
            return pd.DataFrame()
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the harmonized data."""
        df = self.get_harmonized_data()
        
        if df.empty:
            return {}
        
        summary = {
            'total_museums': len(df),
            'cities_with_population': len(df[df['city_population'].notna()]),
            'cities_without_population': len(df[df['city_population'].isna()]),
            'total_visitors': df['annual_visitors'].sum(),
            'avg_visitors_per_museum': df['annual_visitors'].mean(),
            'median_visitors_per_museum': df['annual_visitors'].median(),
            'avg_city_population': df['city_population'].mean(),
            'median_city_population': df['city_population'].median(),
            'avg_visitor_population_ratio': df['visitor_population_ratio'].mean(),
            'median_visitor_population_ratio': df['visitor_population_ratio'].median()
        }
        
        return summary


def main():
    """Main function to run the harmonizer."""
    harmonizer = DataHarmonizer()
    
    # Check if input files exist
    museum_file = "museum_data.csv"
    population_file = "city_population.csv"
    
    if not Path(museum_file).exists() or not Path(population_file).exists():
        print("Please run the data extraction scripts first to generate input files")
        return
    
    # Harmonize data
    harmonized_df = harmonizer.harmonize_data(museum_file, population_file)
    
    if not harmonized_df.empty:
        print(f"Successfully harmonized {len(harmonized_df)} museum records")
        print("\nSample harmonized data:")
        print(harmonized_df.head())
        
        # Get summary statistics
        summary = harmonizer.get_data_summary()
        print("\nData Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:,.2f}")
            else:
                print(f"{key}: {value:,}")
    else:
        print("Failed to harmonize data")


if __name__ == "__main__":
    main()