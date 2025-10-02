"""
Wikipedia scraper for museum data extraction.
Extracts museum information from the Wikipedia page listing most visited museums.
"""

import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WikipediaMuseumScraper:
    """Scraper for extracting museum data from Wikipedia."""
    
    def __init__(self):
        self.base_url = "https://en.wikipedia.org"
        self.museum_list_url = "https://en.wikipedia.org/wiki/List_of_most_visited_museums"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Museum Analytics WS'
        })
    
    def get_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse Wikipedia page content."""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_visitor_number(self, text: str) -> Optional[int]:
        """Extract visitor number from text, handling various formats."""
        if not text:
            return None
            
        # Remove common text patterns
        text = re.sub(r'\([^)]*\)', '', text)  # Remove parentheses content
        text = re.sub(r'\[[^\]]*\]', '', text)  # Remove references [1], [2], ...
        
        # million / billion
        million_match = re.search(r'(\d+(?:\.\d+)?)\s*million', text.lower())
        if million_match:
            return int(float(million_match.group(1)) * 1_000_000)
        
        billion_match = re.search(r'(\d+(?:\.\d+)?)\s*billion', text.lower())
        if billion_match:
            return int(float(billion_match.group(1)) * 1_000_000_000)
        
        # raw number with commas or dots
        number_match = re.search(r'(\d{1,3}(?:[,\.\s]\d{3})*(?:\.\d+)?)', text)
        if number_match:
            number_str = number_match.group(1)
            number_str = number_str.replace(',', '').replace(' ', '')
            try:
                number = float(number_str)
            except ValueError:
                return None
            if number < 1000:
                return int(number * 1_000_000)
            return int(number)
        
        return None
    
    def extract_museum_data(self) -> List[Dict]:
        """Extract museum data from the Wikipedia page."""
        soup = self.get_page_content(self.museum_list_url)
        if not soup:
            return []
        
        museums: List[Dict] = []
        
        # ---------- PATH 1: Classic wikitable parsing (kept; works if tables exist) ----------
        tables = soup.find_all('table', {'class': 'wikitable'})
        for table in tables:
            rows = table.find_all('tr')
            if not rows:
                continue

            header_cells = [th.get_text(strip=True).lower() for th in rows[0].find_all('th')]
            if not header_cells:
                header_cells = [td.get_text(strip=True).lower() for td in rows[0].find_all('td')]

            def idx(*candidates: str) -> Optional[int]:
                for name in candidates:
                    for i, h in enumerate(header_cells):
                        if name in h:
                            return i
                return None

            museum_idx   = idx('museum', 'name')
            city_idx     = idx('city')
            country_idx  = idx('country')
            visitors_idx = idx('visitors', 'visits', 'attendance')
            year_idx     = idx('year')

            if museum_idx is None or city_idx is None or visitors_idx is None:
                continue

            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < max(museum_idx, city_idx, visitors_idx) + 1:
                    continue
                
                try:
                    museum_name_cell = cells[museum_idx]
                    museum_link = museum_name_cell.find('a')
                    museum_name = (museum_link.get_text(strip=True)
                                   if museum_link and museum_link.get_text(strip=True)
                                   else museum_name_cell.get_text(strip=True))
                    if not museum_name:
                        continue

                    city_cell = cells[city_idx]
                    city_name = city_cell.get_text(strip=True)

                    visitors_cell = cells[visitors_idx]
                    visitors_text = visitors_cell.get_text(strip=True)
                    annual_visitors = self.extract_visitor_number(visitors_text)

                    country_cell = cells[country_idx] if (country_idx is not None and len(cells) > country_idx) else city_cell
                    country = self._extract_country(country_cell)

                    year_cell = cells[year_idx] if (year_idx is not None and len(cells) > year_idx) else visitors_cell
                    year = self._extract_year(year_cell)

                    if annual_visitors and annual_visitors >= 2_000_000:
                        museum_data = {
                            'museum_name': museum_name,
                            'city': city_name,
                            'annual_visitors': annual_visitors,
                            'country': country,
                            'year': year
                        }
                        museums.append(museum_data)
                        logger.info(f"Extracted (table): {museum_name} in {city_name} - {annual_visitors:,} visitors")

                except Exception as e:
                    logger.warning(f"Error processing row: {e}")
                    continue
        
        if museums:
            return museums
        # ---------- PATH 2: Fallback — parse the 2024 section rendered as inline text ----------
        # Find the <span id="Most-visited_museums_in_2024"> or the h2 with that text
        section_anchor = soup.find(id='Most-visited_museums_in_2024')
        section_h2 = section_anchor.find_parent(['h2','h3']) if section_anchor else None
        if not section_h2:
            # Try by heading text as a fallback
            for h2 in soup.select('h2'):
                if 'Most-visited museums in 2024' in h2.get_text(strip=True):
                    section_h2 = h2
                    break
        if not section_h2:
            logger.warning("Could not locate the 2024 section. No data extracted.")
            return []

        # Walk siblings until next h2, collect lines of content that contain museum entries
        current = section_h2.next_sibling
        while current and getattr(current, 'name', None) != 'h2':
            # consider paragraphs, lists, and generic divs that might hold the lines
            if getattr(current, 'name', None) in (None, 'p', 'div', 'ul', 'ol'):
                # Each “line” is effectively the concatenated text of this block.
                # We'll split heuristically by newline or bullet items if present.
                # Build a list of candidate chunks: either <li> items or the block itself.
                items = current.find_all('li') if hasattr(current, 'find_all') else []
                if not items and getattr(current, 'get_text', None):
                    items = [current]  # treat the whole block as one item

                for item in items:
                    # Must contain a number and at least 3 anchors (museum, city, country)
                    text = item.get_text(" ", strip=True)
                    if not text or not re.search(r'\d', text):
                        continue
                    anchors = item.find_all('a')
                    if len(anchors) < 2:  # some rows have 2+ (museum + city/country)
                        continue

                    museum_name = anchors[0].get_text(strip=True)
                    if not museum_name:
                        continue

                    # Country = last anchor's text
                    country = anchors[-1].get_text(strip=True) or None

                    # City = anchors between first and last (join if multiple like "Vatican City, Rome")
                    mid_anchors = anchors[1:-1]
                    city_name = ", ".join(a.get_text(strip=True) for a in mid_anchors) if mid_anchors else None

                    annual_visitors = self.extract_visitor_number(text)
                    year = self._extract_year_text(text)

                    if annual_visitors and annual_visitors >= 2_000_000:
                        museum_data = {
                            'museum_name': museum_name,
                            'city': city_name,
                            'annual_visitors': annual_visitors,
                            'country': country,
                            'year': year
                        }
                        museums.append(museum_data)
                        logger.info(f"Extracted (fallback): {museum_name} in {city_name} - {annual_visitors:,} visitors")
            current = current.next_sibling

        return museums
    
    def _extract_country(self, cell) -> Optional[str]:
        """Extract country information from a cell (usually city or a separate country cell)."""
        text = cell.get_text(" ", strip=True)
        country_match = re.search(r'\(([^)]+)\)', text)
        if country_match:
            return country_match.group(1).strip()

        link = cell.find('a', title=True)
        if link:
            title = link.get('title', '').strip()
            if title:
                return title

        # last resort: if cell is just a simple word, return it
        simple = text.strip()
        return simple if simple and len(simple.split()) <= 4 else None
    
    def _extract_year(self, visitors_cell) -> Optional[int]:
        """Extract year information from visitors cell."""
        text = visitors_cell.get_text()
        year_match = re.search(r'(20\d{2})', text)
        if year_match:
            return int(year_match.group(1))
        return None

    def _extract_year_text(self, text: str) -> Optional[int]:
        """Extract year from arbitrary text (fallback path)."""
        year_match = re.search(r'(20\d{2})', text)
        if year_match:
            return int(year_match.group(1))
        return None
    
    def save_to_csv(self, museums: List[Dict], filename: str = "./data/museum_data.csv"):
        """Save museum data to CSV file."""
        if not museums:
            logger.warning("No museum data to save")
            return
        
        df = pd.DataFrame(museums)
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(museums)} museums to {filename}")
        return df


def main():
    """Main function to run the scraper."""
    scraper = WikipediaMuseumScraper()
    museums = scraper.extract_museum_data()
    
    if museums:
        df = scraper.save_to_csv(museums, "./data/museum_data.csv")
        print(f"Successfully extracted {len(museums)} museums")
        print("\nSample data:")
        print(df.head())
    else:
        print("No museum data extracted")


if __name__ == "__main__":
    main()
