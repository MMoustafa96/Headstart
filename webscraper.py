import requests
import re
import pandas as pd
from bs4 import BeautifulSoup as bs
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def extract_genres(soup):
    """Extracts and returns a list of genres from the parsed HTML soup."""
    genres = []
    for category in soup.find_all('div', class_='category-wrap'):
        dt_element = category.find('dt')
        if dt_element and re.search('Genre', dt_element.text, re.IGNORECASE):
            genres.extend(g.text for g in category.find_all('rt-link'))
    return genres

def extract_year(soup):
    """Extracts the first 4-digit year from the metadata in the parsed HTML soup."""
    for element in soup.find_all('rt-text', slot='metadataProp'):
        if match := re.search(r'\d{4}', element.text):
            return match.group()
    return None  # Return None if no year is found

def fetch_movie_data(movie):
    """Fetches and processes data for a single movie."""
    url = f'https://www.rottentomatoes.com/m/{movie.replace(" ", "_")}'
    r = requests.get(url)
    
    if r.status_code != 200:
        return None
    
    soup = bs(r.content, 'html.parser')
    
    try:
        audience_score = soup.find_all('rt-text', attrs={'slot': 'audienceScore'})[0].text
        critic_score = soup.find_all('rt-text', attrs={'slot': 'criticsScore'})[0].text
        genres = extract_genres(soup)
        year = extract_year(soup)

        return [(genre, year, audience_score, critic_score) for genre in genres]
        
    except IndexError:
        return None

if __name__ == '__main__':
    imdb_basics_url = "https://datasets.imdbws.com/title.basics.tsv.gz"
    imdb_akas_url = "https://datasets.imdbws.com/title.akas.tsv.gz"

    # Load datasets with necessary columns
    df_basics = pd.read_csv(imdb_basics_url, sep='\t', compression='gzip', usecols=['tconst', 'titleType', 'primaryTitle', 'startYear'], low_memory=False)
    df_akas = pd.read_csv(imdb_akas_url, sep='\t', compression='gzip', usecols=['titleId', 'language'], low_memory=False)

    # Convert 'startYear' to numeric, setting errors='coerce' to handle non-numeric values
    df_basics['startYear'] = pd.to_numeric(df_basics['startYear'], errors='coerce')

    # Filter for English-language movies
    df_english = df_akas[df_akas['language'] == 'en']
    df_merged = df_basics.merge(df_english, left_on='tconst', right_on='titleId', how='inner')

    # Filter for movies released after 1998
    movies = list(df_merged.loc[(df_merged['titleType'] == 'movie') & (df_merged['startYear'] > 1998), 'primaryTitle'].unique())

    del df_basics, df_akas, df_english, df_merged

    pd.DataFrame(movies, columns=['Movie']).to_csv('movies.csv', index=False)
    DATA = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_movie_data, movie): movie for movie in movies}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching movie data"):
            result = future.result()
            if result:
                for genre, year, audience_score, critic_score in result:
                    if audience_score != '' and critic_score != '':
                        audience_score = int(audience_score.replace('%', ''))
                        critic_score = int(critic_score.replace('%', ''))
                        if genre in DATA:
                            DATA[genre].append([year, audience_score, critic_score])
                        else:
                            DATA[genre] = [[year, audience_score, critic_score]]
                    else:
                        continue
    
    # Convert each genre's data to a DataFrame
    dfs = []
    for genre, data in DATA.items():
        df = pd.DataFrame(data, columns=['Year', 'Audience Score', 'Critic Score'])
        df['Genre'] = genre
        dfs.append(df)
    
    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)

    df.to_csv('rottentomatoes_scraped.csv', index=False)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Year'] = df['Year'].fillna(0).astype(int)

    summarised = df.groupby(['Genre', 'Year']).agg({'Audience Score': 'mean', 'Critic Score': 'mean'}).reset_index()
    
    print(df.head())
