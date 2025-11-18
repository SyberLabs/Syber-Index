import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. LOAD THE DATA ---
print("Loading the Engine...")
df_eci = pd.read_csv('growth_proj_eci_rankings.csv')
df_so = pd.read_csv('stackoverflow_ratios.csv')
df_gh = pd.read_csv('github_construction.csv')
df_gdp = pd.read_csv('world_bank_gdp.csv', skiprows=4)
df_pat = pd.read_csv('patents_data.csv')

# --- 2. DATA CLEANING ---
df_eci = df_eci[['country_iso3_code', 'year', 'eci_hs92']]
df_eci.columns = ['Country_Code', 'Year', 'ECI']

# GDP Cleaning
df_gdp = df_gdp.melt(id_vars=['Country Code'], var_name='Year', value_name='GDP_Per_Capita')
df_gdp = df_gdp[['Country Code', 'Year', 'GDP_Per_Capita']]
df_gdp.columns = ['Country_Code', 'Year', 'GDP_Per_Capita']
df_gdp['Year'] = pd.to_numeric(df_gdp['Year'], errors='coerce')
df_gdp['GDP_Per_Capita'] = pd.to_numeric(df_gdp['GDP_Per_Capita'], errors='coerce')
df_gdp = df_gdp.dropna()

# Patent Cleaning
iso2_to_iso3 = {
    'US': 'USA', 'CN': 'CHN', 'JP': 'JPN', 'DE': 'DEU', 'IN': 'IND', 'GB': 'GBR',
    'FR': 'FRA', 'BR': 'BRA', 'IT': 'ITA', 'CA': 'CAN', 'RU': 'RUS', 'KR': 'KOR',
    'AU': 'AUS', 'ES': 'ESP', 'MX': 'MEX', 'ID': 'IDN', 'NL': 'NLD', 'TR': 'TUR',
    'CH': 'CHE', 'SE': 'SWE', 'PL': 'POL', 'BE': 'BEL', 'TH': 'THA', 'AT': 'AUT',
    'IL': 'ISR', 'IE': 'IRL', 'NG': 'NGA', 'EG': 'EGY', 'PK': 'PAK', 'VN': 'VNM',
    'SG': 'SGP', 'MY': 'MYS', 'AR': 'ARG', 'CO': 'COL', 'ZA': 'ZAF', 'UA': 'UKR'
}
df_pat['Country_Code'] = df_pat['country_code'].map(iso2_to_iso3)
df_pat = df_pat.dropna(subset=['Country_Code'])
df_pat = df_pat.rename(columns={'year': 'Year'})

# Normalization Function
def normalize_country(location_str):
    if not isinstance(location_str, str): return None
    loc = location_str.lower()
    if 'united states' in loc or 'usa' in loc: return 'USA'
    if 'china' in loc: return 'CHN'
    if 'japan' in loc: return 'JPN'
    if 'germany' in loc: return 'DEU'
    if 'india' in loc: return 'IND'
    if 'united kingdom' in loc or 'uk' in loc: return 'GBR'
    if 'france' in loc: return 'FRA'
    if 'brazil' in loc or 'brasil' in loc: return 'BRA'
    if 'italy' in loc: return 'ITA'
    if 'canada' in loc: return 'CAN'
    if 'russia' in loc: return 'RUS'
    if 'south korea' in loc or 'korea' in loc: return 'KOR'
    if 'australia' in loc: return 'AUS'
    if 'spain' in loc: return 'ESP'
    if 'mexico' in loc: return 'MEX'
    if 'indonesia' in loc: return 'IDN'
    if 'netherlands' in loc: return 'NLD'
    if 'turkey' in loc: return 'TUR'
    if 'switzerland' in loc: return 'CHE'
    if 'sweden' in loc: return 'SWE'
    if 'poland' in loc: return 'POL'
    if 'belgium' in loc: return 'BEL'
    if 'thailand' in loc: return 'THA'
    if 'austria' in loc: return 'AUT'
    if 'israel' in loc: return 'ISR'
    if 'ireland' in loc: return 'IRL'
    if 'nigeria' in loc: return 'NGA'
    if 'egypt' in loc: return 'EGY'
    if 'pakistan' in loc: return 'PAK'
    if 'vietnam' in loc: return 'VNM'
    if 'bangladesh' in loc: return 'BGD'
    if 'south africa' in loc: return 'ZAF'
    if 'philippines' in loc: return 'PHL'
    if 'ukraine' in loc: return 'UKR'
    if 'singapore' in loc: return 'SGP'
    if 'malaysia' in loc: return 'MYS'
    if 'argentina' in loc: return 'ARG'
    if 'colombia' in loc: return 'COL'
    return None

df_so['Country_Code'] = df_so['location'].apply(normalize_country)
df_gh['Country_Code'] = df_gh['location'].apply(normalize_country)

df_so_nat = df_so.groupby(['Country_Code', 'year']).sum().reset_index().rename(columns={'year':'Year'})
df_gh_nat = df_gh.groupby(['Country_Code', 'year']).sum().reset_index().rename(columns={'year':'Year'})

# --- 3. MERGE & METRICS ---
print("Merging...")
master = pd.merge(df_so_nat, df_eci, on=['Country_Code', 'Year'], how='inner')
master = pd.merge(master, df_gh_nat, on=['Country_Code', 'Year'], how='left', suffixes=('_SO', '_GH'))
master = pd.merge(master, df_gdp, on=['Country_Code', 'Year'], how='inner')
master = pd.merge(master, df_pat, on=['Country_Code', 'Year'], how='left')

# Handle NaNs (Crucial for Clustering)
master = master.fillna(0)

# Metrics Calculation
# Syber Index (Software)
master['Score_Intent'] = master['tech_high_count'] / (master['tech_low_count'] + 1)
master['Score_Construction'] = master['tech_high_projects'] / (master['tech_low_projects'] + 1)
master['Total_Volume_Soft'] = master['total_posts'] + master['total_projects']
master['Syber_Index'] = ((master['Score_Intent'] + master['Score_Construction']) / 2) * np.log(master['Total_Volume_Soft'] + 1)

# Patent Index (Hardware)
master['Patent_Quality'] = master['tech_hard_patents'] / (master['total_patents'] + 1)
master['Patent_Index'] = master['Patent_Quality'] * np.log(master['total_patents'] + 1)

# --- 4. CLUSTERING (The "Cluster Map") ---
print("\n" + "="*60)
print("THE INNOVATION CLUSTER MAP")
print("="*60)

# Select features for clustering (Year 2015 snapshot for stability)
# We focus on the two engines: Software (Syber) and Hardware (Patent)
cluster_df = master[master['Year'] == 2015].copy()
features = cluster_df[['Syber_Index', 'Patent_Index']]

# Standardize features (Scale them so one doesn't dominate)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Run K-Means with 3 Clusters
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_df['Cluster'] = kmeans.fit_predict(scaled_features)

# Analyze the Clusters
cluster_summary = cluster_df.groupby('Cluster')[['Syber_Index', 'Patent_Index', 'ECI', 'GDP_Per_Capita']].mean()
print(cluster_summary)

# Name the Clusters automatically based on the centroids
# Find the "Dual Engine" Cluster (High both)
dual_id = cluster_summary['Syber_Index'].idxmax() 
# Find the "Digital" Cluster (High Syber, Low Patent)
# Find the "Developing" Cluster (Low Both)
# We'll just map them manually after seeing the print output, or let the plot speak.

# --- 5. VISUALIZATION ---
plt.figure(figsize=(12, 8))
sns.scatterplot(data=cluster_df, x='Syber_Index', y='Patent_Index', 
                hue='Cluster', size='GDP_Per_Capita', sizes=(50, 500), palette='viridis', style='Cluster')

plt.title('The Strategic Map of Nations (2015)\nClustered by Innovation Strategy (Software vs Hardware)')
plt.xlabel('Software Power (Syber Index)')
plt.ylabel('Hardware Power (Patent Index)')
plt.legend(title='Cluster ID')
plt.grid(True, alpha=0.3)

# Label countries
for i, row in cluster_df.iterrows():
    plt.text(row['Syber_Index']+0.02, row['Patent_Index'], row['Country_Code'], fontsize=9)

plt.savefig('innovation_clusters.png')
print("Saved plot to 'innovation_clusters.png'")

# Save final dataset
cluster_df.to_csv('final_clusters.csv', index=False)