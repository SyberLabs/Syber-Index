import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. LOAD THE QUINTET ---
print("Loading the Civilization Engine...")
df_eci = pd.read_csv('growth_proj_eci_rankings.csv')
df_so = pd.read_csv('stackoverflow_ratios.csv')
df_gh = pd.read_csv('github_construction.csv')
df_gdp = pd.read_csv('world_bank_gdp.csv', skiprows=4)
df_edu = pd.read_csv('world_bank_education.csv', skiprows=4)
# NEW: Patents
df_pat = pd.read_csv('patents_data.csv')

# --- 2. DATA REFINEMENT ---

# Standard Cleaners
df_eci = df_eci[['country_iso3_code', 'year', 'eci_hs92']]
df_eci.columns = ['Country_Code', 'Year', 'ECI']

def clean_world_bank(df, value_name):
    df_clean = df.melt(id_vars=['Country Code'], var_name='Year', value_name=value_name)
    df_clean = df_clean[['Country Code', 'Year', value_name]]
    df_clean.columns = ['Country_Code', 'Year', value_name]
    df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
    df_clean[value_name] = pd.to_numeric(df_clean[value_name], errors='coerce')
    return df_clean.dropna()

df_gdp_clean = clean_world_bank(df_gdp, 'GDP_Per_Capita')
df_edu_clean = clean_world_bank(df_edu, 'Education_Tertiary')

# Normalize Patents
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

# FIX: Ensure Year is capitalized to match other dataframes
df_pat = df_pat.rename(columns={'year': 'Year'})

# Reuse our robust normalizer for SO/GH
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

# --- 3. MERGE EVERYTHING ---
print("Merging...")
master = pd.merge(df_so_nat, df_eci, on=['Country_Code', 'Year'], how='inner')
master = pd.merge(master, df_gh_nat, on=['Country_Code', 'Year'], how='left', suffixes=('_SO', '_GH'))
master = pd.merge(master, df_gdp_clean, on=['Country_Code', 'Year'], how='inner')
master = pd.merge(master, df_edu_clean, on=['Country_Code', 'Year'], how='inner')
# Merge Patents
master = pd.merge(master, df_pat, on=['Country_Code', 'Year'], how='left')
master['total_patents'] = master['total_patents'].fillna(0)
master['tech_hard_patents'] = master['tech_hard_patents'].fillna(0)

# --- 4. METRICS CONSTRUCTION ---

# Syber Index (Software)
master['tech_high_projects'] = master['tech_high_projects'].fillna(0)
master['tech_low_projects'] = master['tech_low_projects'].fillna(0)
master['total_projects'] = master['total_projects'].fillna(0)
master['Score_Intent'] = master['tech_high_count'] / (master['tech_low_count'] + 1)
master['Score_Construction'] = master['tech_high_projects'] / (master['tech_low_projects'] + 1)
master['Total_Volume_Soft'] = master['total_posts'] + master['total_projects']
master['Syber_Index'] = ((master['Score_Intent'] + master['Score_Construction']) / 2) * np.log(master['Total_Volume_Soft'])

# Patent Index (Hardware)
# Ratio of Hard Tech (Physics/Elec) to Total
master['Patent_Quality'] = master['tech_hard_patents'] / (master['total_patents'] + 1)
# Composite Patent Score (Quality * Log Volume)
master['Patent_Index'] = master['Patent_Quality'] * np.log(master['total_patents'] + 1)

# --- 5. THE "DUAL ENGINE" ANALYSIS ---
print("\n" + "="*60)
print("THE INNOVATION METABOLISM: SOFTWARE vs HARDWARE")
print("="*60)

# Time Shifting
master = master.sort_values(['Country_Code', 'Year'])
master['ECI_Lead_3'] = master.groupby('Country_Code')['ECI'].shift(-3)

test_df = master.dropna(subset=['Syber_Index', 'Patent_Index', 'ECI_Lead_3'])

# 1. Correlation Matrix
corr_soft = test_df['Syber_Index'].corr(test_df['ECI_Lead_3'])
corr_hard = test_df['Patent_Index'].corr(test_df['ECI_Lead_3'])
corr_cross = test_df['Syber_Index'].corr(test_df['Patent_Index'])

print(f"1. Software Signal (Syber -> Future ECI):   {corr_soft:.4f}")
print(f"2. Hardware Signal (Patents -> Future ECI): {corr_hard:.4f}")
print(f"3. Coupling (Software <-> Hardware):        {corr_cross:.4f}")

print("-" * 60)
if corr_soft > corr_hard:
    print("INSIGHT: Software is currently a stronger predictor than Hardware.")
else:
    print("INSIGHT: Hardware/Patents remains the dominant predictor.")

if corr_cross > 0.5:
    print("INSIGHT: The two engines are highly coupled (Hardware Nations are Software Nations).")
else:
    print("INSIGHT: Decoupling detected! Some nations are 'Software Pure' vs 'Hardware Pure'.")

# --- 6. VISUALIZATION ---
plt.figure(figsize=(10, 8))
sns.scatterplot(data=test_df, x='Syber_Index', y='Patent_Index', 
                hue='ECI_Lead_3', size='GDP_Per_Capita', sizes=(20, 200), palette='viridis')
plt.title('The Matrix: Software (Syber) vs Hardware (Patents)')
plt.xlabel('Software Complexity (Syber Index)')
plt.ylabel('Hardware Complexity (Patent Index)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Label Quadrants
top_soft = test_df.sort_values('Syber_Index', ascending=False).head(5)
top_hard = test_df.sort_values('Patent_Index', ascending=False).head(5)
labels = pd.concat([top_soft, top_hard])
for i, row in labels.iterrows():
    plt.text(row['Syber_Index'], row['Patent_Index'], row['Country_Code'])

plt.savefig('dual_engine_plot.png')
print("Saved plot to 'dual_engine_plot.png'")