import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D

# Load your CSV
df = pd.read_csv("life-expectancy-of-women-vs-life-expectancy-of-men.csv")

# Define continent color mapping
continent_colors = {
    'Africa': 'red',
    'Europe': 'blue',
    'Asia': 'green',
    'Oceania': 'purple',
    'Americas': 'orange'
}

# Manual country name-to-continent mapping
country_name_to_continent = {
    'Afghanistan': 'Asia', 'Albania': 'Europe', 'Algeria': 'Africa', 'Argentina': 'Americas',
    'Australia': 'Oceania', 'Austria': 'Europe', 'Bangladesh': 'Asia', 'Belgium': 'Europe',
    'Brazil': 'Americas', 'Canada': 'Americas', 'Chile': 'Americas', 'China': 'Asia',
    'Colombia': 'Americas', 'Cuba': 'Americas', 'Czechia': 'Europe', 'Denmark': 'Europe',
    'Egypt': 'Africa', 'Ethiopia': 'Africa', 'Finland': 'Europe', 'France': 'Europe',
    'Germany': 'Europe', 'Ghana': 'Africa', 'Greece': 'Europe', 'Hungary': 'Europe',
    'India': 'Asia', 'Indonesia': 'Asia', 'Iran': 'Asia', 'Iraq': 'Asia', 'Israel': 'Asia',
    'Italy': 'Europe', 'Japan': 'Asia', 'Jordan': 'Asia', 'Kenya': 'Africa',
    'Malaysia': 'Asia', 'Mexico': 'Americas', 'Morocco': 'Africa', 'Netherlands': 'Europe',
    'New Zealand': 'Oceania', 'Nigeria': 'Africa', 'Norway': 'Europe', 'Pakistan': 'Asia',
    'Peru': 'Americas', 'Philippines': 'Asia', 'Poland': 'Europe', 'Portugal': 'Europe',
    'Romania': 'Europe', 'Russia': 'Europe', 'Saudi Arabia': 'Asia', 'Singapore': 'Asia',
    'South Africa': 'Africa', 'South Korea': 'Asia', 'Spain': 'Europe', 'Sri Lanka': 'Asia',
    'Sweden': 'Europe', 'Switzerland': 'Europe', 'Syria': 'Asia', 'Thailand': 'Asia',
    'Tunisia': 'Africa', 'Turkey': 'Asia', 'Ukraine': 'Europe', 'United Kingdom': 'Europe',
    'United States': 'Americas', 'Vietnam': 'Asia', 'Yemen': 'Asia', 'Zambia': 'Africa'
}

# Clean and filter dataset
df = df.dropna(subset=[
    'Life expectancy - Sex: female - Age: 0 - Variant: estimates',
    'Life expectancy - Sex: male - Age: 0 - Variant: estimates',
    'Population - Sex: all - Age: all - Variant: estimates'
])

df['Continent'] = df['Entity'].map(country_name_to_continent)
df = df[df['Continent'].notna()]
df['Color'] = df['Continent'].map(continent_colors)
df['Life Expectancy Gap'] = df['Life expectancy - Sex: female - Age: 0 - Variant: estimates'] - df['Life expectancy - Sex: male - Age: 0 - Variant: estimates']
df['Is Outlier'] = df['Life Expectancy Gap'].abs() > 10
key_countries = {'United States', 'China', 'India', 'Nigeria', 'Japan', 'United Kingdom', 'South Africa'}
df['Show Label'] = df['Entity'].isin(key_countries) | df['Is Outlier']

# Create animation
fig, ax = plt.subplots(figsize=(12, 8))
years = list(range(1950, 2024))

def update(year):
    ax.clear()
    data = df[df['Year'] == year]
    ax.scatter(
        data['Life expectancy - Sex: male - Age: 0 - Variant: estimates'],
        data['Life expectancy - Sex: female - Age: 0 - Variant: estimates'],
        s=data['Population - Sex: all - Age: all - Variant: estimates'] / 1e6,
        c=data['Color'],
        alpha=0.6,
        edgecolors='k',
        marker='o'
    )

    for _, row in data[data['Show Label']].iterrows():
        ax.text(row['Life expectancy - Sex: male - Age: 0 - Variant: estimates'],
                row['Life expectancy - Sex: female - Age: 0 - Variant: estimates'],
                row['Entity'], fontsize=8, ha='center', va='bottom')

    ax.text(0.95, 0.05, str(year), transform=ax.transAxes,
            fontsize=40, color='gray', alpha=0.5, ha='right', va='bottom')

    ax.set_title("Female vs Male Life Expectancy by Country", fontsize=14)
    ax.set_xlabel("Male Life Expectancy")
    ax.set_ylabel("Female Life Expectancy")
    ax.set_xlim(10, 90)
    ax.set_ylim(10, 95)
    ax.grid(True)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=continent,
               markerfacecolor=color, markeredgecolor='k', markersize=8)
        for continent, color in continent_colors.items()
    ]
    ax.legend(handles=legend_elements, title='Continent', loc='upper left')

ani = FuncAnimation(fig, update, frames=years, interval=1000, repeat=False)

# Save to MP4
ani.save("life_expectancy_1950_to_2023.mp4", writer=FFMpegWriter(fps=1), dpi=100)
