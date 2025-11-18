import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. LOAD THE QUARTET ---
print("Loading Datasets...")
df_eci = pd.read_csv('growth_proj_eci_rankings.csv')
df_so = pd.read_csv('stackoverflow_ratios.csv')
df_gh = pd.read_csv('github_construction.csv')
df_gdp = pd.read_csv('world_bank_gdp.csv', skiprows=4)
# NEW: Education Data
df_edu = pd.read_csv('world_bank_education.csv', skiprows=4)

# --- 2. DATA REFINEMENT ---

# Clean ECI
df_eci = df_eci[['country_iso3_code', 'year', 'eci_hs92']]
df_eci.columns = ['Country_Code', 'Year', 'ECI']

# Helper to clean World Bank Data (GDP & Education share structure)
def clean_world_bank(df, value_name):
    df_clean = df.melt(id_vars=['Country Code'], var_name='Year', value_name=value_name)
    df_clean = df_clean[['Country Code', 'Year', value_name]]
    df_clean.columns = ['Country_Code', 'Year', value_name]
    df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
    df_clean[value_name] = pd.to_numeric(df_clean[value_name], errors='coerce')
    return df_clean.dropna()

df_gdp_clean = clean_world_bank(df_gdp, 'GDP_Per_Capita')
df_edu_clean = clean_world_bank(df_edu, 'Education_Tertiary')

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

# --- 3. MERGE & CALCULATE INDEX ---
print("Forging the Index...")
master = pd.merge(df_so_nat, df_eci, on=['Country_Code', 'Year'], how='inner')
master = pd.merge(master, df_gh_nat, on=['Country_Code', 'Year'], how='left', suffixes=('_SO', '_GH'))
master = pd.merge(master, df_gdp_clean, on=['Country_Code', 'Year'], how='inner')
# NEW: Merge Education
master = pd.merge(master, df_edu_clean, on=['Country_Code', 'Year'], how='inner')

master['tech_high_projects'] = master['tech_high_projects'].fillna(0)
master['tech_low_projects'] = master['tech_low_projects'].fillna(0)
master['total_projects'] = master['total_projects'].fillna(0)

# Metrics
master['Score_Intent'] = master['tech_high_count'] / (master['tech_low_count'] + 1)
master['Score_Construction'] = master['tech_high_projects'] / (master['tech_low_projects'] + 1)
master['Total_Volume'] = master['total_posts'] + master['total_projects']

# The Syber Index
master['Syber_Index'] = ((master['Score_Intent'] + master['Score_Construction']) / 2) * np.log(master['Total_Volume'])

# --- 4. THE CAUSAL CHAIN ANALYSIS ---
print("\n" + "="*60)
print("THE CAUSAL CHAIN: EDUCATION -> CODE -> WEALTH")
print("="*60)

# We want to see if Education predicts Syber Index, and Syber Index predicts ECI.
# We use lags to simulate the flow of time.

master = master.sort_values(['Country_Code', 'Year'])

# Create Lags
# Education happens first (t-5)
# Syber Index happens next (t)
# ECI happens last (t+5)

master['Education_Lag_5'] = master.groupby('Country_Code')['Education_Tertiary'].shift(5) # Education 5 years ago
master['ECI_Lead_3'] = master.groupby('Country_Code')['ECI'].shift(-3) # ECI 3 years in future

test_df = master.dropna(subset=['Education_Lag_5', 'Syber_Index', 'ECI_Lead_3'])

print(f"Analyzing {len(test_df)} Country-Year timelines...")

# 1. Link A: Does Education create Cognitive Complexity?
corr_edu_syber = test_df['Education_Lag_5'].corr(test_df['Syber_Index'])
print(f"Link 1: Education (t-5) -> Syber Index (t):   {corr_edu_syber:.4f}")

# 2. Link B: Does Cognitive Complexity create Economic Complexity?
corr_syber_eci = test_df['Syber_Index'].corr(test_df['ECI_Lead_3'])
print(f"Link 2: Syber Index (t) -> ECI (t+3):         {corr_syber_eci:.4f}")

# 3. The Direct Path (for comparison)
corr_edu_eci = test_df['Education_Lag_5'].corr(test_df['ECI_Lead_3'])
print(f"Link 3: Education (t-5) -> ECI (t+3):         {corr_edu_eci:.4f}")

print("-" * 60)
if corr_syber_eci > corr_edu_eci:
    print("VERDICT: The Syber Index is a BETTER predictor than Education alone.")
    print("It captures 'applied' knowledge, not just 'enrolled' students.")
else:
    print("VERDICT: Education is the primary driver.")

# --- 5. VISUALIZATION ---
plt.figure(figsize=(12, 6))

# Plot Link 1
plt.subplot(1, 2, 1)
sns.regplot(data=test_df, x='Education_Lag_5', y='Syber_Index', color='blue', scatter_kws={'alpha':0.4})
plt.title(f'Step 1: Education -> Code\n(Corr: {corr_edu_syber:.2f})')
plt.xlabel('Tertiary Enrollment (5 Years Ago)')
plt.ylabel('Syber Index (Today)')

# Plot Link 2
plt.subplot(1, 2, 2)
sns.regplot(data=test_df, x='Syber_Index', y='ECI_Lead_3', color='green', scatter_kws={'alpha':0.4})
plt.title(f'Step 2: Code -> Wealth\n(Corr: {corr_syber_eci:.2f})')
plt.xlabel('Syber Index (Today)')
plt.ylabel('Economic Complexity (3 Years Future)')

plt.tight_layout()
plt.savefig('causal_chain_plot.png')
print("Saved causal chain plot to 'causal_chain_plot.png'")