# Museum Analytics: Visitor-Population Correlation Analysis

A comprehensive data engineering solution that analyzes the correlation between museum visitor numbers and city population using machine learning. This project extracts museum data from Wikipedia, combines it with city population data, and builds a linear regression model to understand the relationship between city size and museum attendance.

## 🎯 Project Overview

This project demonstrates a complete data engineering workflow:

- **Data Extraction**: Scrapes museum data from Wikipedia API (museums with >2M annual visitors)
- **Data Integration**: Combines museum data with city population data from multiple sources
- **Data Processing**: Harmonizes and cleans data for analysis
- **Machine Learning**: Builds a linear regression model with log transformation
- **Database**: Stores data in SQLite for querying and analysis
- **Visualization**: Creates comprehensive plots and interactive Jupyter notebook
- **Containerization**: Docker setup for easy deployment and reproducibility

## 🏗️ Project Architecture

```
museum_analytics/
├── src/
│   ├── data_extraction/
│   │   ├── wikipedia_scraper.py    # Wikipedia API scraper
│   │   └── population_data.py      # Population data extractor
│   ├── data_processing/
│   │   └── harmonizer.py           # Data harmonization
│   ├── models/
│   │   └── regression_model.py     # ML model implementation
│   └── database/
│       └── db_manager.py           # Database operations
├── notebooks/
│   └── museum_analysis.ipynb       # Interactive analysis
├── data/                           # Generated data files
├── models/                         # Trained models
├── plots/                          # Visualization outputs
├── Dockerfile                      # Container configuration
├── docker-compose.yml             # Orchestration
└── run_pipeline.py                # Main execution script
```

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Navigate to the project
cd museum-analytics

# Build and run with Docker Compose
docker-compose up --build

# Access Jupyter Lab at http://localhost:8888
# Open museum_analytics/notebooks/museum_analysis.ipynb
```

### Option 2: Manual Setup

```bash
# Create and activate virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Upgrade pip (optional but recommended)
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Run the complete pipeline
python run_pipeline.py

# Or run individual components
python -m museum_analytics.src.data_extraction.wikipedia_scraper
python -m museum_analytics.src.data_extraction.population_data
python -m museum_analytics.src.data_processing.harmonizer
python -m museum_analytics.src.models.regression_model
```

## 📊 Data Sources

### Museum Data
- **Source**: Wikipedia API (List of most visited museums)
- **Criteria**: Museums with >2,000,000 annual visitors
- **Fields**: Museum name, city, country, annual visitors, year
- **Extraction**: Robust web scraping with fallback parsing methods

### Population Data
- **Sources**: 
  - REST Countries API (country-level data)
  - Manual city-specific data for major cities
- **Fields**: City name, country, population
- **Coverage**: Major metropolitan areas worldwide

## 🔧 Technical Implementation

### Data Pipeline
1. **Wikipedia Scraper**: Extracts museum data using BeautifulSoup with robust error handling
2. **Population Extractor**: Combines API data with manual mappings for comprehensive coverage
3. **Data Harmonizer**: Matches museum cities with population data using fuzzy matching
4. **Database Manager**: SQLite database for data persistence and complex queries
5. **Regression Model**: Linear regression with log transformation for better fit

### Machine Learning Model
- **Algorithm**: Linear regression with log1p transformation
- **Features**: City population (log-transformed)
- **Target**: Annual museum visitors (log-transformed)
- **Evaluation**: R², MSE, MAE, correlation coefficient
- **Preprocessing**: StandardScaler for feature normalization

### Key Features
- **Robust Error Handling**: Comprehensive logging and graceful failure handling
- **Scalable Architecture**: Modular design for easy extension
- **Data Validation**: Quality checks throughout the pipeline
- **Interactive Analysis**: Jupyter notebook for exploration and visualization
- **Docker Containerization**: Consistent deployment environment

## 📈 Model Performance

The linear regression model provides insights into the relationship between city population and museum attendance:

- **Correlation**: Measures the strength of the linear relationship
- **R² Score**: Indicates how well the model explains the variance in visitor numbers
- **Predictions**: Estimates visitor numbers for different city sizes
- **Log Transformation**: Improves model fit for count data

## 📋 Requirements

### System Requirements
- Python 3.8+
- Docker (optional)
- 2GB RAM minimum
- 1GB disk space

### Python Dependencies
- pandas >= 2.2
- numpy >= 2.0
- scikit-learn >= 1.5.2
- matplotlib >= 3.9.0
- seaborn == 0.13.0
- requests >= 2.31.0
- beautifulsoup4 >= 4.12.2
- jupyter >= 1.0.0
- lxml >= 5.3.0

## 🔍 Usage Examples

### Running the Complete Pipeline
```python
from museum_analytics.src.data_extraction.wikipedia_scraper import WikipediaMuseumScraper
from museum_analytics.src.models.regression_model import MuseumRegressionModel

# Extract data
scraper = WikipediaMuseumScraper()
museums = scraper.extract_museum_data()

# Train model
model = MuseumRegressionModel()
metrics = model.train("data/harmonized_museum_data.csv")

# Make predictions
prediction = model.predict(5_000_000)  # City with 5M population
print(f"Predicted visitors: {prediction:,.0f}")
```

### Database Queries
```python
from museum_analytics.src.database.db_manager import DatabaseManager

db = DatabaseManager("data/museum_analytics.db")
top_museums = db.get_top_museums(10)
stats = db.get_correlation_stats()
```

## 📊 Output Files

The pipeline generates several output files:

- `data/museum_data.csv`: Raw museum data from Wikipedia
- `data/city_population.csv`: City population data
- `data/harmonized_museum_data.csv`: Combined dataset
- `data/museum_analytics.db`: SQLite database
- `models/museum_regression_model.pkl`: Trained ML model
- `plots/`: Visualization files (PNG format)
  - `regression_plot.png`: Scatter plot with regression line
  - `residual_plot.png`: Residual analysis
  - `residual_distribution.png`: Distribution of residuals

## 🎯 Business Insights

The analysis reveals:

1. **Correlation Strength**: How strongly city population correlates with museum attendance
2. **Visitor Ratios**: Average percentage of city population that visits museums
3. **Top Performers**: Museums with highest visitor-population ratios
4. **Predictive Power**: Model accuracy for visitor predictions
5. **Geographic Patterns**: Regional differences in museum attendance


## 👥 Author

**Jawher** - Technical Assignment
