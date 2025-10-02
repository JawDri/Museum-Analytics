"""
Population data extractor for cities.
Uses multiple sources to get city population data.
"""

import requests
import pandas as pd
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PopulationDataExtractor:
    """Extractor for city population data from various sources."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Museum Analytics'
        })
        # Minimal country name -> Wikidata QID map (extend as needed)
        self.country_qids: Dict[str, str] = {
            'France': 'Q142',
            'United Kingdom': 'Q145',
            'United States': 'Q30',
            'Japan': 'Q17',
            'China': 'Q148',
            'Spain': 'Q29',
            'Italy': 'Q38',
            'Germany': 'Q183',
            'Turkey': 'Q43',
            'Egypt': 'Q79',
            'Mexico': 'Q96',
            'Brazil': 'Q155',
            'Argentina': 'Q414',
            'Australia': 'Q408',
            'Canada': 'Q16',
            'Netherlands': 'Q55',
            'Austria': 'Q40',
            'Czech Republic': 'Q213',
            'Poland': 'Q36',
            'Greece': 'Q41',
            'Portugal': 'Q45',
            'Ireland': 'Q27',
            'Norway': 'Q20',
            'Sweden': 'Q34',
            'Denmark': 'Q35',
            'Finland': 'Q33',
            'Switzerland': 'Q39',
            'Belgium': 'Q31',
            'Hungary': 'Q28',
            'Romania': 'Q218',
            'Bulgaria': 'Q219',
            'Croatia': 'Q224',
            'Slovenia': 'Q215',
            'Slovakia': 'Q214',
            'Lithuania': 'Q37',
            'Latvia': 'Q211',
            'Estonia': 'Q191',
            'Ukraine': 'Q212',
            'Belarus': 'Q184',
            'Moldova': 'Q217',
            'Georgia': 'Q230',
            'Armenia': 'Q399',
            'Azerbaijan': 'Q227',
            'Kazakhstan': 'Q232',
            'Uzbekistan': 'Q265',
            'Tajikistan': 'Q863',
            'Kyrgyzstan': 'Q813',
            'Turkmenistan': 'Q874',
            'Afghanistan': 'Q889',
            'Pakistan': 'Q843',
            'India': 'Q668',
            'Bangladesh': 'Q902',
            'Sri Lanka': 'Q854',
            'Nepal': 'Q837',
            'Bhutan': 'Q917',
            'Maldives': 'Q826',
            'Myanmar': 'Q836',
            'Thailand': 'Q869',
            'Cambodia': 'Q424',
            'Laos': 'Q819',
            'Vietnam': 'Q881',
            'Malaysia': 'Q833',
            'Singapore': 'Q334',
            'Indonesia': 'Q252',
            'Philippines': 'Q928',
            'Brunei': 'Q921',
            'Timor-Leste': 'Q574',
            'Papua New Guinea': 'Q691',
            'Fiji': 'Q712',
        }

    def get_worldbank_data(self, city_names: List[str]) -> Dict[str, int]:
        """
        Get population data from World Bank API.
        Note: World Bank primarily has country-level data, so we'll use a fallback approach.
        """
        population_data: Dict[str, int] = {}
        try:
            url = "https://api.worldbank.org/v2/country/all/indicator/SP.URB.TOTL?format=json&per_page=1000&date=2020:2023"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if len(data) > 1 and data[1]:
                for item in data[1]:
                    country = item.get('country', {}).get('value', '')
                    urban_pop = item.get('value')
                    if urban_pop and country:
                        logger.info(f"Got urban population for {country}: {urban_pop:,}")
        except Exception as e:
            logger.warning(f"World Bank API error: {e}")
        return population_data

    def get_restcountries_data(self, city_names: List[str]) -> Dict[str, int]:
        """Get country population data from REST Countries API (coarse fallback)."""
        population_data: Dict[str, int] = {}
        try:
            url = "https://restcountries.com/v3.1/all"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            countries_data = response.json()

            country_populations = {}
            for country in countries_data:
                name = country.get('name', {}).get('common', '')
                population = country.get('population', 0)
                if name and population:
                    country_populations[name.lower()] = population

            city_country_mapping = self._get_city_country_mapping()
            for city in city_names:
                cn = city_country_mapping.get(city.lower())
                if cn:
                    pop = country_populations.get(cn.lower())
                    if pop:
                        population_data[city] = pop
                        logger.info(f"Mapped {city} to {cn} population: {pop:,}")
        except Exception as e:
            logger.warning(f"REST Countries API error: {e}")
        return population_data

    def get_wikidata_city_populations(self, city_names: List[str]) -> Dict[str, int]:
        """
        Fetch city population (P1082) from Wikidata via SPARQL.
        Disambiguation:
          - restrict to human settlements (Q486972) to avoid rivers/regions,
          - if we know the country, constrain with wdt:P17 to that country,
          - prefer the MOST RECENT statement first, then larger population.
        """
        results: Dict[str, int] = {}
        endpoint = "https://query.wikidata.org/sparql"
        headers = {**self.session.headers, "Accept": "application/sparql-results+json"}

        city_country_mapping = self._get_city_country_mapping()

        for city in city_names:
            country_name = city_country_mapping.get(city.lower())
            country_qid = self.country_qids.get(country_name) if country_name else None

            # Build an optional country constraint if we know the QID
            country_clause = f"?item wdt:P17 wd:{country_qid} ." if country_qid else ""

            # Prefer newest population statement; if multiple on the same date, take the largest.
            query = f"""
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?pop ?date WHERE {{
              ?item rdfs:label "{city}"@en .
              ?item wdt:P31/wdt:P279* wd:Q486972 .
              {country_clause}
              ?item p:P1082 ?popStmt .
              ?popStmt ps:P1082 ?pop .
              OPTIONAL {{ ?popStmt pq:P585 ?date . }}
              BIND(COALESCE(?date, "0000-01-01T00:00:00Z"^^xsd:dateTime) AS ?dateSort)
            }}
            ORDER BY DESC(?dateSort) DESC(?pop)
            LIMIT 1
            """
            try:
                r = self.session.get(endpoint, params={"query": query}, headers=headers, timeout=20)
                r.raise_for_status()
                data = r.json()
                bindings = data.get("results", {}).get("bindings", [])
                if bindings:
                    pop_str = bindings[0].get("pop", {}).get("value")
                    if pop_str:
                        pop_val = int(float(pop_str))
                        results[city] = pop_val
                        d = bindings[0].get("date", {}).get("value")
                        if d:
                            logger.info(f"Wikidata population for {city}: {pop_val:,} (as of {d})")
                        else:
                            logger.info(f"Wikidata population for {city}: {pop_val:,}")
                else:
                    logger.info(f"Wikidata: no population found for {city}")
            except Exception as e:
                logger.warning(f"Wikidata error for {city}: {e}")
        return results
    
    def _get_city_country_mapping(self) -> Dict[str, str]:
        """Manual mapping of major cities to their countries (for REST Countries fallback and Wikidata country constraint)."""
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
        }
    
    def get_city_population_data(self, city_names: List[str]) -> Dict[str, int]:
        """Get population data for cities from multiple sources."""
        population_data: Dict[str, int] = {}
        
        # 1) Country-level fallback (coarse, keeps compatibility)
        country_data = self.get_restcountries_data(city_names)
        population_data.update(country_data)
        
        # 2) City-level values from Wikidata (override country totals when available)
        wikidata_data = self.get_wikidata_city_populations(city_names)
        population_data.update(wikidata_data)
        
        return population_data
    
    def save_to_csv(self, population_data: Dict[str, int], filename: str = "./data/city_population.csv"):
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
    
    # Use ALL cities defined in the internal mapping (deduplicated + title-cased)
    mapping = extractor._get_city_country_mapping()
    sample_cities = list(dict.fromkeys(k.title() for k in mapping.keys()))

    population_data = extractor.get_city_population_data(sample_cities)
    
    if population_data:
        df = extractor.save_to_csv(population_data, "./data/city_population.csv")
        print(f"Successfully extracted population data for {len(population_data)} cities")
        print("\nSample data:")
        print(df.head())
    else:
        print("No population data extracted")


if __name__ == "__main__":
    main()
