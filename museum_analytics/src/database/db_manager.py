"""
Database manager for museum analytics.
Handles database operations and queries.
"""

import sqlite3
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database operations for museum analytics."""
    
    def __init__(self, db_path: str = "museum_analytics.db"):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Connect to the database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def execute_query(self, query: str, params: Tuple = ()) -> List[Dict]:
        """Execute a query and return results as list of dictionaries."""
        if not self.conn:
            self.connect()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    def execute_query_df(self, query: str, params: Tuple = ()) -> pd.DataFrame:
        """Execute a query and return results as pandas DataFrame."""
        if not self.conn:
            self.connect()
        
        try:
            return pd.read_sql_query(query, self.conn, params=params)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def get_museums_with_population(self) -> pd.DataFrame:
        """Get all museums with population data."""
        query = """
        SELECT 
            museum_name,
            COALESCE(city, city_museum)               AS city,
            country,
            annual_visitors,
            COALESCE(city_population, population)     AS city_population,
            visitor_population_ratio,
            year
        FROM harmonized_data 
        WHERE COALESCE(city_population, population) IS NOT NULL
        ORDER BY annual_visitors DESC
        """
        return self.execute_query_df(query)
    
    def get_top_museums(self, limit: int = 10) -> pd.DataFrame:
        """Get top museums by visitor count."""
        query = """
        SELECT 
            museum_name,
            COALESCE(city, city_museum)               AS city,
            country,
            annual_visitors,
            COALESCE(city_population, population)     AS city_population,
            visitor_population_ratio
        FROM harmonized_data 
        WHERE COALESCE(city_population, population) IS NOT NULL
        ORDER BY annual_visitors DESC
        LIMIT ?
        """
        return self.execute_query_df(query, (limit,))
    
    def get_cities_by_population(self, limit: int = 10) -> pd.DataFrame:
        """Get cities ordered by population."""
        query = """
        SELECT 
            COALESCE(city, city_museum)               AS city,
            country,
            COALESCE(city_population, population)     AS city_population,
            COUNT(*)                                  AS museum_count,
            SUM(annual_visitors)                      AS total_visitors,
            AVG(annual_visitors)                      AS avg_visitors
        FROM harmonized_data 
        WHERE COALESCE(city_population, population) IS NOT NULL
        GROUP BY COALESCE(city, city_museum), country, COALESCE(city_population, population)
        ORDER BY COALESCE(city_population, population) DESC
        LIMIT ?
        """
        return self.execute_query_df(query, (limit,))
    
    def get_correlation_stats(self) -> Dict:
        """Get correlation statistics."""
        query = """
        SELECT 
            COUNT(*)                                   AS total_records,
            AVG(annual_visitors)                       AS avg_visitors,
            AVG(COALESCE(city_population, population)) AS avg_population,
            AVG(visitor_population_ratio)              AS avg_ratio,
            MIN(annual_visitors)                       AS min_visitors,
            MAX(annual_visitors)                       AS max_visitors,
            MIN(COALESCE(city_population, population)) AS min_population,
            MAX(COALESCE(city_population, population)) AS max_population
        FROM harmonized_data 
        WHERE COALESCE(city_population, population) IS NOT NULL
        """
        results = self.execute_query(query)
        return results[0] if results else {}

    
    def get_museums_by_country(self) -> pd.DataFrame:
        """Get museum statistics by country."""
        query = """
        SELECT 
            country,
            COUNT(*)                                   AS museum_count,
            SUM(annual_visitors)                       AS total_visitors,
            AVG(annual_visitors)                       AS avg_visitors,
            AVG(COALESCE(city_population, population)) AS avg_city_population
        FROM harmonized_data 
        WHERE COALESCE(city_population, population) IS NOT NULL AND country IS NOT NULL
        GROUP BY country
        ORDER BY total_visitors DESC
        """
        return self.execute_query_df(query)
    
    def get_museums_without_population(self) -> pd.DataFrame:
        """Get museums without population data."""
        query = """
        SELECT 
            museum_name,
            COALESCE(city, city_museum)               AS city,
            country,
            annual_visitors,
            year
        FROM harmonized_data 
        WHERE COALESCE(city_population, population) IS NULL
        ORDER BY annual_visitors DESC
        """
        return self.execute_query_df(query)
    
    def search_museums(self, search_term: str) -> pd.DataFrame:
        """Search museums by name or city."""
        query = """
        SELECT 
            museum_name,
            COALESCE(city, city_museum)               AS city,
            country,
            annual_visitors,
            COALESCE(city_population, population)     AS city_population,
            visitor_population_ratio
        FROM harmonized_data 
        WHERE museum_name LIKE ? 
           OR COALESCE(city, city_museum) LIKE ? 
           OR country LIKE ?
        ORDER BY annual_visitors DESC
        """
        search_pattern = f"%{search_term}%"
        return self.execute_query_df(query, (search_pattern, search_pattern, search_pattern))
    
    def get_database_info(self) -> Dict:
        """Get database information and statistics."""
        if not self.conn:
            self.connect()
        
        try:
            cursor = self.conn.cursor()
            
            # Get table information
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            info = {
                'database_path': self.db_path,
                'tables': tables,
                'table_counts': {}
            }
            
            # Get record counts for each table
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                info['table_counts'][table] = count
            
            return info
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {}
    
    def export_to_csv(self, table_name: str, output_file: str):
        """Export table to CSV file."""
        query = f"SELECT * FROM {table_name}"
        df = self.execute_query_df(query)
        
        if not df.empty:
            df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(df)} records from {table_name} to {output_file}")
        else:
            logger.warning(f"No data found in table {table_name}")
    
    def create_backup(self, backup_path: str):
        """Create a backup of the database."""
        if not self.conn:
            self.connect()
        
        try:
            backup_conn = sqlite3.connect(backup_path)
            self.conn.backup(backup_conn)
            backup_conn.close()
            logger.info(f"Database backup created: {backup_path}")
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
    
    def get_analysis_ready_data(self) -> pd.DataFrame:
        """Get data ready for analysis (no missing values)."""
        query = """
        SELECT 
            museum_name,
            COALESCE(city, city_museum)               AS city,
            country,
            annual_visitors,
            COALESCE(city_population, population)     AS city_population,
            visitor_population_ratio,
            year
        FROM harmonized_data 
        WHERE COALESCE(city_population, population) IS NOT NULL 
          AND annual_visitors IS NOT NULL
          AND visitor_population_ratio IS NOT NULL
        ORDER BY annual_visitors DESC
        """
        return self.execute_query_df(query)


def main():
    """Main function to demonstrate database operations."""
    db_manager = DatabaseManager()
    
    # Check if database exists
    if not Path("museum_analytics.db").exists():
        print("Database not found. Please run the data harmonization script first.")
        return
    
    try:
        # Get database info
        info = db_manager.get_database_info()
        print("Database Information:")
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        # Get correlation stats
        stats = db_manager.get_correlation_stats()
        print("\nCorrelation Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:,.2f}")
            else:
                print(f"{key}: {value:,}")
        
        # Get top museums
        top_museums = db_manager.get_top_museums(5)
        print("\nTop 5 Museums by Visitors:")
        print(top_museums[['museum_name', 'city', 'annual_visitors']].to_string(index=False))
        
        # Get cities by population
        top_cities = db_manager.get_cities_by_population(5)
        print("\nTop 5 Cities by Population:")
        print(top_cities[['city', 'country', 'city_population', 'museum_count']].to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
