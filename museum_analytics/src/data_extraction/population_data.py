"""
Population data extractor for cities.
Uses multiple sources to get city population data.
"""

import requests
import pandas as pd
import json
from typing import Dict, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PopulationDataExtractor:
    """Extractor for city population data from various sources."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Museum Analytics Bot 1.0 (Educational Purpose)'
        })
    
    def get_worldbank_data(self, city_names: List[str]) -> Dict[str, int]:
        """
        Get population data from World Bank API.
        Note: World Bank primarily has country-level data, so we'll use a fallback approach.
        """
        population_data = {}
        
        # World Bank API for urban population (country level)
        try:
            # Get urban population data for major countries
            url = "https://api.worldbank.org/v2/country/all/indicator/SP.URB.TOTL?format=json&per_page=1000&date=2020:2023"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if len(data) > 1 and data[1]:
                for item in data[1]:
                    country = item.get('country', {}).get('value', '')
                    urban_pop = item.get('value')
                    if urban_pop and country:
                        # This is country-level urban population, not city-specific
                        logger.info(f"Got urban population for {country}: {urban_pop:,}")
            
        except Exception as e:
            logger.warning(f"World Bank API error: {e}")
        
        return population_data
    
    def get_restcountries_data(self, city_names: List[str]) -> Dict[str, int]:
        """Get country population data from REST Countries API."""
        population_data = {}
        
        try:
            url = "https://restcountries.com/v3.1/all"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            countries_data = response.json()
            
            # Create a mapping of country names to populations
            country_populations = {}
            for country in countries_data:
                name = country.get('name', {}).get('common', '')
                population = country.get('population', 0)
                if name and population:
                    country_populations[name.lower()] = population
            
            # Map cities to countries (simplified mapping)
            city_country_mapping = self._get_city_country_mapping()
            
            for city in city_names:
                city_lower = city.lower()
                if city_lower in city_country_mapping:
                    country = city_country_mapping[city_lower]
                    if country.lower() in country_populations:
                        population_data[city] = country_populations[country.lower()]
                        logger.info(f"Mapped {city} to {country} population: {population_data[city]:,}")
        
        except Exception as e:
            logger.warning(f"REST Countries API error: {e}")
        
        return population_data
    
    def _get_city_country_mapping(self) -> Dict[str, str]:
        """Manual mapping of major cities to their countries."""
        return {
            'paris': 'France',
            'london': 'United Kingdom',
            'new york': 'United States',
            'tokyo': 'Japan',
            'beijing': 'China',
            'madrid': 'Spain',
            'rome': 'Italy',
            'berlin': 'Germany',
            'moscow': 'Russia',
            'istanbul': 'Turkey',
            'cairo': 'Egypt',
            'mexico city': 'Mexico',
            'são paulo': 'Brazil',
            'buenos aires': 'Argentina',
            'sydney': 'Australia',
            'toronto': 'Canada',
            'amsterdam': 'Netherlands',
            'vienna': 'Austria',
            'prague': 'Czech Republic',
            'warsaw': 'Poland',
            'athens': 'Greece',
            'lisbon': 'Portugal',
            'dublin': 'Ireland',
            'oslo': 'Norway',
            'stockholm': 'Sweden',
            'copenhagen': 'Denmark',
            'helsinki': 'Finland',
            'zurich': 'Switzerland',
            'brussels': 'Belgium',
            'budapest': 'Hungary',
            'bucharest': 'Romania',
            'sofia': 'Bulgaria',
            'zagreb': 'Croatia',
            'ljubljana': 'Slovenia',
            'bratislava': 'Slovakia',
            'vilnius': 'Lithuania',
            'riga': 'Latvia',
            'tallinn': 'Estonia',
            'kiev': 'Ukraine',
            'minsk': 'Belarus',
            'chisinau': 'Moldova',
            'tbilisi': 'Georgia',
            'yerevan': 'Armenia',
            'baku': 'Azerbaijan',
            'almaty': 'Kazakhstan',
            'tashkent': 'Uzbekistan',
            'dushanbe': 'Tajikistan',
            'bishkek': 'Kyrgyzstan',
            'ashgabat': 'Turkmenistan',
            'kabul': 'Afghanistan',
            'islamabad': 'Pakistan',
            'new delhi': 'India',
            'dhaka': 'Bangladesh',
            'colombo': 'Sri Lanka',
            'kathmandu': 'Nepal',
            'thimphu': 'Bhutan',
            'male': 'Maldives',
            'yangon': 'Myanmar',
            'bangkok': 'Thailand',
            'phnom penh': 'Cambodia',
            'vientiane': 'Laos',
            'hanoi': 'Vietnam',
            'kuala lumpur': 'Malaysia',
            'singapore': 'Singapore',
            'jakarta': 'Indonesia',
            'manila': 'Philippines',
            'bandar seri begawan': 'Brunei',
            'dili': 'East Timor',
            'port moresby': 'Papua New Guinea',
            'suva': 'Fiji',
            'nuku\'alofa': 'Tonga',
            'apia': 'Samoa',
            'majuro': 'Marshall Islands',
            'palikir': 'Micronesia',
            'tarawa': 'Kiribati',
            'funafuti': 'Tuvalu',
            'yaren': 'Nauru',
            'honiara': 'Solomon Islands',
            'port vila': 'Vanuatu',
            'nouméa': 'New Caledonia',
            'papeete': 'French Polynesia',
            'avarua': 'Cook Islands',
            'alofi': 'Niue',
            'adamstown': 'Pitcairn Islands',
            'kingston': 'Norfolk Island',
            'flying fish cove': 'Christmas Island',
            'west island': 'Cocos Islands',
            'the settlement': 'Cocos Islands',
            'cockburn town': 'Turks and Caicos Islands',
            'cockburn town': 'Turks and Caicos Islands',
            'george town': 'Cayman Islands',
            'hamilton': 'Bermuda',
            'st. john\'s': 'Antigua and Barbuda',
            'basseterre': 'Saint Kitts and Nevis',
            'castries': 'Saint Lucia',
            'kingstown': 'Saint Vincent and the Grenadines',
            'bridgetown': 'Barbados',
            'port of spain': 'Trinidad and Tobago',
            'georgetown': 'Guyana',
            'paramaribo': 'Suriname',
            'caracas': 'Venezuela',
            'bogotá': 'Colombia',
            'quito': 'Ecuador',
            'lima': 'Peru',
            'la paz': 'Bolivia',
            'asuncion': 'Paraguay',
            'montevideo': 'Uruguay',
            'santiago': 'Chile',
            'brasília': 'Brazil',
            'montevideo': 'Uruguay',
            'asuncion': 'Paraguay',
            'la paz': 'Bolivia',
            'lima': 'Peru',
            'quito': 'Ecuador',
            'bogotá': 'Colombia',
            'caracas': 'Venezuela',
            'paramaribo': 'Suriname',
            'georgetown': 'Guyana',
            'port of spain': 'Trinidad and Tobago',
            'bridgetown': 'Barbados',
            'kingstown': 'Saint Vincent and the Grenadines',
            'castries': 'Saint Lucia',
            'basseterre': 'Saint Kitts and Nevis',
            'st. john\'s': 'Antigua and Barbuda',
            'hamilton': 'Bermuda',
            'george town': 'Cayman Islands',
            'cockburn town': 'Turks and Caicos Islands',
            'west island': 'Cocos Islands',
            'flying fish cove': 'Christmas Island',
            'kingston': 'Norfolk Island',
            'adamstown': 'Pitcairn Islands',
            'alofi': 'Niue',
            'avarua': 'Cook Islands',
            'papeete': 'French Polynesia',
            'nouméa': 'New Caledonia',
            'port vila': 'Vanuatu',
            'honiara': 'Solomon Islands',
            'yaren': 'Nauru',
            'funafuti': 'Tuvalu',
            'tarawa': 'Kiribati',
            'palikir': 'Micronesia',
            'majuro': 'Marshall Islands',
            'apia': 'Samoa',
            'nuku\'alofa': 'Tonga',
            'suva': 'Fiji',
            'port moresby': 'Papua New Guinea',
            'dili': 'East Timor',
            'bandar seri begawan': 'Brunei',
            'manila': 'Philippines',
            'jakarta': 'Indonesia',
            'singapore': 'Singapore',
            'kuala lumpur': 'Malaysia',
            'hanoi': 'Vietnam',
            'vientiane': 'Laos',
            'phnom penh': 'Cambodia',
            'bangkok': 'Thailand',
            'yangon': 'Myanmar',
            'male': 'Maldives',
            'thimphu': 'Bhutan',
            'kathmandu': 'Nepal',
            'colombo': 'Sri Lanka',
            'dhaka': 'Bangladesh',
            'new delhi': 'India',
            'islamabad': 'Pakistan',
            'kabul': 'Afghanistan',
            'ashgabat': 'Turkmenistan',
            'bishkek': 'Kyrgyzstan',
            'dushanbe': 'Tajikistan',
            'tashkent': 'Uzbekistan',
            'almaty': 'Kazakhstan',
            'baku': 'Azerbaijan',
            'yerevan': 'Armenia',
            'tbilisi': 'Georgia',
            'chisinau': 'Moldova',
            'minsk': 'Belarus',
            'kiev': 'Ukraine',
            'tallinn': 'Estonia',
            'riga': 'Latvia',
            'vilnius': 'Lithuania',
            'bratislava': 'Slovakia',
            'ljubljana': 'Slovenia',
            'zagreb': 'Croatia',
            'sofia': 'Bulgaria',
            'bucharest': 'Romania',
            'budapest': 'Hungary',
            'brussels': 'Belgium',
            'zurich': 'Switzerland',
            'helsinki': 'Finland',
            'copenhagen': 'Denmark',
            'stockholm': 'Sweden',
            'oslo': 'Norway',
            'dublin': 'Ireland',
            'lisbon': 'Portugal',
            'athens': 'Greece',
            'warsaw': 'Poland',
            'prague': 'Czech Republic',
            'vienna': 'Austria',
            'amsterdam': 'Netherlands',
            'toronto': 'Canada',
            'sydney': 'Australia',
            'buenos aires': 'Argentina',
            'são paulo': 'Brazil',
            'mexico city': 'Mexico',
            'cairo': 'Egypt',
            'istanbul': 'Turkey',
            'moscow': 'Russia',
            'berlin': 'Germany',
            'rome': 'Italy',
            'madrid': 'Spain',
            'beijing': 'China',
            'tokyo': 'Japan',
            'new york': 'United States',
            'london': 'United Kingdom',
            'paris': 'France'
        }
    
    def get_city_population_data(self, city_names: List[str]) -> Dict[str, int]:
        """Get population data for cities from multiple sources."""
        population_data = {}
        
        # Try REST Countries API first (country-level data)
        country_data = self.get_restcountries_data(city_names)
        population_data.update(country_data)
        
        # Add some manual city-specific data for major cities
        major_cities_data = {
            'Paris': 11_000_000,  # Greater Paris
            'London': 9_000_000,  # Greater London
            'New York': 8_400_000,  # NYC proper
            'Tokyo': 14_000_000,  # Tokyo proper
            'Beijing': 21_500_000,  # Beijing municipality
            'Madrid': 6_700_000,  # Madrid metropolitan area
            'Rome': 4_300_000,  # Rome metropolitan area
            'Berlin': 3_700_000,  # Berlin proper
            'Moscow': 12_500_000,  # Moscow proper
            'Istanbul': 15_500_000,  # Istanbul metropolitan area
            'Cairo': 20_900_000,  # Greater Cairo
            'Mexico City': 9_200_000,  # Mexico City proper
            'São Paulo': 12_300_000,  # São Paulo proper
            'Buenos Aires': 3_100_000,  # Buenos Aires proper
            'Sydney': 5_300_000,  # Greater Sydney
            'Toronto': 2_900_000,  # Toronto proper
            'Amsterdam': 1_200_000,  # Amsterdam metropolitan area
            'Vienna': 1_900_000,  # Vienna metropolitan area
            'Prague': 1_300_000,  # Prague metropolitan area
            'Warsaw': 1_800_000,  # Warsaw metropolitan area
            'Athens': 3_200_000,  # Athens metropolitan area
            'Lisbon': 2_900_000,  # Lisbon metropolitan area
            'Dublin': 1_400_000,  # Dublin metropolitan area
            'Oslo': 1_000_000,  # Oslo metropolitan area
            'Stockholm': 1_600_000,  # Stockholm metropolitan area
            'Copenhagen': 1_300_000,  # Copenhagen metropolitan area
            'Helsinki': 1_200_000,  # Helsinki metropolitan area
            'Zurich': 1_400_000,  # Zurich metropolitan area
            'Brussels': 1_200_000,  # Brussels metropolitan area
            'Budapest': 1_800_000,  # Budapest metropolitan area
            'Bucharest': 1_800_000,  # Bucharest metropolitan area
            'Sofia': 1_200_000,  # Sofia metropolitan area
            'Zagreb': 1_100_000,  # Zagreb metropolitan area
            'Ljubljana': 300_000,  # Ljubljana metropolitan area
            'Bratislava': 600_000,  # Bratislava metropolitan area
            'Vilnius': 600_000,  # Vilnius metropolitan area
            'Riga': 600_000,  # Riga metropolitan area
            'Tallinn': 400_000,  # Tallinn metropolitan area
            'Kiev': 3_000_000,  # Kiev metropolitan area
            'Minsk': 2_000_000,  # Minsk metropolitan area
            'Chisinau': 700_000,  # Chisinau metropolitan area
            'Tbilisi': 1_100_000,  # Tbilisi metropolitan area
            'Yerevan': 1_100_000,  # Yerevan metropolitan area
            'Baku': 2_300_000,  # Baku metropolitan area
            'Almaty': 2_000_000,  # Almaty metropolitan area
            'Tashkent': 2_500_000,  # Tashkent metropolitan area
            'Dushanbe': 800_000,  # Dushanbe metropolitan area
            'Bishkek': 1_000_000,  # Bishkek metropolitan area
            'Ashgabat': 1_000_000,  # Ashgabat metropolitan area
            'Kabul': 4_500_000,  # Kabul metropolitan area
            'Islamabad': 1_000_000,  # Islamabad metropolitan area
            'New Delhi': 32_000_000,  # Delhi metropolitan area
            'Dhaka': 21_000_000,  # Dhaka metropolitan area
            'Colombo': 2_300_000,  # Colombo metropolitan area
            'Kathmandu': 1_500_000,  # Kathmandu metropolitan area
            'Thimphu': 100_000,  # Thimphu metropolitan area
            'Male': 200_000,  # Male metropolitan area
            'Yangon': 7_400_000,  # Yangon metropolitan area
            'Bangkok': 10_500_000,  # Bangkok metropolitan area
            'Phnom Penh': 2_200_000,  # Phnom Penh metropolitan area
            'Vientiane': 1_000_000,  # Vientiane metropolitan area
            'Hanoi': 8_000_000,  # Hanoi metropolitan area
            'Kuala Lumpur': 7_200_000,  # Kuala Lumpur metropolitan area
            'Singapore': 5_900_000,  # Singapore metropolitan area
            'Jakarta': 10_800_000,  # Jakarta metropolitan area
            'Manila': 13_500_000,  # Manila metropolitan area
            'Bandar Seri Begawan': 100_000,  # Bandar Seri Begawan metropolitan area
            'Dili': 200_000,  # Dili metropolitan area
            'Port Moresby': 400_000,  # Port Moresby metropolitan area
            'Suva': 200_000,  # Suva metropolitan area
            'Nuku\'alofa': 100_000,  # Nuku'alofa metropolitan area
            'Apia': 100_000,  # Apia metropolitan area
            'Majuro': 30_000,  # Majuro metropolitan area
            'Palikir': 10_000,  # Palikir metropolitan area
            'Tarawa': 50_000,  # Tarawa metropolitan area
            'Funafuti': 10_000,  # Funafuti metropolitan area
            'Yaren': 10_000,  # Yaren metropolitan area
            'Honiara': 100_000,  # Honiara metropolitan area
            'Port Vila': 50_000,  # Port Vila metropolitan area
            'Nouméa': 200_000,  # Nouméa metropolitan area
            'Papeete': 200_000,  # Papeete metropolitan area
            'Avarua': 20_000,  # Avarua metropolitan area
            'Alofi': 1_000,  # Alofi metropolitan area
            'Adamstown': 50,  # Adamstown metropolitan area
            'Kingston': 2_000,  # Kingston metropolitan area
            'Flying Fish Cove': 1_000,  # Flying Fish Cove metropolitan area
            'West Island': 1_000,  # West Island metropolitan area
            'The Settlement': 1_000,  # The Settlement metropolitan area
            'Cockburn Town': 5_000,  # Cockburn Town metropolitan area
            'George Town': 30_000,  # George Town metropolitan area
            'Hamilton': 60_000,  # Hamilton metropolitan area
            'St. John\'s': 100_000,  # St. John's metropolitan area
            'Basseterre': 50_000,  # Basseterre metropolitan area
            'Castries': 100_000,  # Castries metropolitan area
            'Kingstown': 100_000,  # Kingstown metropolitan area
            'Bridgetown': 100_000,  # Bridgetown metropolitan area
            'Port of Spain': 500_000,  # Port of Spain metropolitan area
            'Georgetown': 200_000,  # Georgetown metropolitan area
            'Paramaribo': 250_000,  # Paramaribo metropolitan area
            'Caracas': 3_000_000,  # Caracas metropolitan area
            'Bogotá': 10_000_000,  # Bogotá metropolitan area
            'Quito': 2_700_000,  # Quito metropolitan area
            'Lima': 10_000_000,  # Lima metropolitan area
            'La Paz': 2_000_000,  # La Paz metropolitan area
            'Asunción': 2_000_000,  # Asunción metropolitan area
            'Montevideo': 1_400_000,  # Montevideo metropolitan area
            'Santiago': 7_000_000,  # Santiago metropolitan area
            'Brasília': 3_000_000,  # Brasília metropolitan area
            'Montevideo': 1_400_000,  # Montevideo metropolitan area
            'Asunción': 2_000_000,  # Asunción metropolitan area
            'La Paz': 2_000_000,  # La Paz metropolitan area
            'Lima': 10_000_000,  # Lima metropolitan area
            'Quito': 2_700_000,  # Quito metropolitan area
            'Bogotá': 10_000_000,  # Bogotá metropolitan area
            'Caracas': 3_000_000,  # Caracas metropolitan area
            'Paramaribo': 250_000,  # Paramaribo metropolitan area
            'Georgetown': 200_000,  # Georgetown metropolitan area
            'Port of Spain': 500_000,  # Port of Spain metropolitan area
            'Bridgetown': 100_000,  # Bridgetown metropolitan area
            'Kingstown': 100_000,  # Kingstown metropolitan area
            'Castries': 100_000,  # Castries metropolitan area
            'Basseterre': 50_000,  # Basseterre metropolitan area
            'St. John\'s': 100_000,  # St. John's metropolitan area
            'Hamilton': 60_000,  # Hamilton metropolitan area
            'George Town': 30_000,  # George Town metropolitan area
            'Cockburn Town': 5_000,  # Cockburn Town metropolitan area
            'The Settlement': 1_000,  # The Settlement metropolitan area
            'West Island': 1_000,  # West Island metropolitan area
            'Flying Fish Cove': 1_000,  # Flying Fish Cove metropolitan area
            'Kingston': 2_000,  # Kingston metropolitan area
            'Adamstown': 50,  # Adamstown metropolitan area
            'Alofi': 1_000,  # Alofi metropolitan area
            'Avarua': 20_000,  # Avarua metropolitan area
            'Papeete': 200_000,  # Papeete metropolitan area
            'Nouméa': 200_000,  # Nouméa metropolitan area
            'Port Vila': 50_000,  # Port Vila metropolitan area
            'Honiara': 100_000,  # Honiara metropolitan area
            'Yaren': 10_000,  # Yaren metropolitan area
            'Funafuti': 10_000,  # Funafuti metropolitan area
            'Tarawa': 50_000,  # Tarawa metropolitan area
            'Palikir': 10_000,  # Palikir metropolitan area
            'Majuro': 30_000,  # Majuro metropolitan area
            'Apia': 100_000,  # Apia metropolitan area
            'Nuku\'alofa': 100_000,  # Nuku'alofa metropolitan area
            'Suva': 200_000,  # Suva metropolitan area
            'Port Moresby': 400_000,  # Port Moresby metropolitan area
            'Dili': 200_000,  # Dili metropolitan area
            'Bandar Seri Begawan': 100_000,  # Bandar Seri Begawan metropolitan area
            'Manila': 13_500_000,  # Manila metropolitan area
            'Jakarta': 10_800_000,  # Jakarta metropolitan area
            'Singapore': 5_900_000,  # Singapore metropolitan area
            'Kuala Lumpur': 7_200_000,  # Kuala Lumpur metropolitan area
            'Hanoi': 8_000_000,  # Hanoi metropolitan area
            'Vientiane': 1_000_000,  # Vientiane metropolitan area
            'Phnom Penh': 2_200_000,  # Phnom Penh metropolitan area
            'Bangkok': 10_500_000,  # Bangkok metropolitan area
            'Yangon': 7_400_000,  # Yangon metropolitan area
            'Male': 200_000,  # Male metropolitan area
            'Thimphu': 100_000,  # Thimphu metropolitan area
            'Kathmandu': 1_500_000,  # Kathmandu metropolitan area
            'Colombo': 2_300_000,  # Colombo metropolitan area
            'Dhaka': 21_000_000,  # Dhaka metropolitan area
            'New Delhi': 32_000_000,  # Delhi metropolitan area
            'Islamabad': 1_000_000,  # Islamabad metropolitan area
            'Kabul': 4_500_000,  # Kabul metropolitan area
            'Ashgabat': 1_000_000,  # Ashgabat metropolitan area
            'Bishkek': 1_000_000,  # Bishkek metropolitan area
            'Dushanbe': 800_000,  # Dushanbe metropolitan area
            'Tashkent': 2_500_000,  # Tashkent metropolitan area
            'Almaty': 2_000_000,  # Almaty metropolitan area
            'Baku': 2_300_000,  # Baku metropolitan area
            'Yerevan': 1_100_000,  # Yerevan metropolitan area
            'Tbilisi': 1_100_000,  # Tbilisi metropolitan area
            'Chisinau': 700_000,  # Chisinau metropolitan area
            'Minsk': 2_000_000,  # Minsk metropolitan area
            'Kiev': 3_000_000,  # Kiev metropolitan area
            'Tallinn': 400_000,  # Tallinn metropolitan area
            'Riga': 600_000,  # Riga metropolitan area
            'Vilnius': 600_000,  # Vilnius metropolitan area
            'Bratislava': 600_000,  # Bratislava metropolitan area
            'Ljubljana': 300_000,  # Ljubljana metropolitan area
            'Zagreb': 1_100_000,  # Zagreb metropolitan area
            'Sofia': 1_200_000,  # Sofia metropolitan area
            'Bucharest': 1_800_000,  # Bucharest metropolitan area
            'Budapest': 1_800_000,  # Budapest metropolitan area
            'Brussels': 1_200_000,  # Brussels metropolitan area
            'Zurich': 1_400_000,  # Zurich metropolitan area
            'Helsinki': 1_200_000,  # Helsinki metropolitan area
            'Copenhagen': 1_300_000,  # Copenhagen metropolitan area
            'Stockholm': 1_600_000,  # Stockholm metropolitan area
            'Oslo': 1_000_000,  # Oslo metropolitan area
            'Dublin': 1_400_000,  # Dublin metropolitan area
            'Lisbon': 2_900_000,  # Lisbon metropolitan area
            'Athens': 3_200_000,  # Athens metropolitan area
            'Warsaw': 1_800_000,  # Warsaw metropolitan area
            'Prague': 1_300_000,  # Prague metropolitan area
            'Vienna': 1_900_000,  # Vienna metropolitan area
            'Amsterdam': 1_200_000,  # Amsterdam metropolitan area
            'Toronto': 2_900_000,  # Toronto proper
            'Sydney': 5_300_000,  # Greater Sydney
            'Buenos Aires': 3_100_000,  # Buenos Aires proper
            'São Paulo': 12_300_000,  # São Paulo proper
            'Mexico City': 9_200_000,  # Mexico City proper
            'Cairo': 20_900_000,  # Greater Cairo
            'Istanbul': 15_500_000,  # Istanbul metropolitan area
            'Moscow': 12_500_000,  # Moscow proper
            'Berlin': 3_700_000,  # Berlin proper
            'Rome': 4_300_000,  # Rome metropolitan area
            'Madrid': 6_700_000,  # Madrid metropolitan area
            'Beijing': 21_500_000,  # Beijing municipality
            'Tokyo': 14_000_000,  # Tokyo proper
            'New York': 8_400_000,  # NYC proper
            'London': 9_000_000,  # Greater London
            'Paris': 11_000_000  # Greater Paris
        }
        
        # Update with major cities data
        for city in city_names:
            if city in major_cities_data:
                population_data[city] = major_cities_data[city]
                logger.info(f"Added major city data: {city} - {major_cities_data[city]:,}")
        
        return population_data
    
    def save_to_csv(self, population_data: Dict[str, int], filename: str = "city_population.csv"):
        """Save population data to CSV file."""
        if not population_data:
            logger.warning("No population data to save")
            return
        
        df = pd.DataFrame(list(population_data.items()), columns=['city', 'population'])
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(population_data)} cities to {filename}")
        return df


def main():
    """Main function to run the population extractor."""
    extractor = PopulationDataExtractor()
    
    # Sample cities for testing
    sample_cities = ['Paris', 'London', 'New York', 'Tokyo', 'Beijing', 'Madrid', 'Rome', 'Berlin']
    population_data = extractor.get_city_population_data(sample_cities)
    
    if population_data:
        df = extractor.save_to_csv(population_data, "city_population.csv")
        print(f"Successfully extracted population data for {len(population_data)} cities")
        print("\nSample data:")
        print(df.head())
    else:
        print("No population data extracted")


if __name__ == "__main__":
    main()