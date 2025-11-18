import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. LOAD THE TRINITY ---
print("Loading Datasets...")
# Ground Truth
df_eci = pd.read_csv('growth_proj_eci_rankings.csv')
# Intent Signal
df_so = pd.read_csv('stackoverflow_ratios.csv')
# Construction Signal
df_gh = pd.read_csv('github_construction.csv')
# Control Signal (Wealth)
df_gdp = pd.read_csv('world_bank_gdp.csv', skiprows=4)

# --- 2. DATA REFINEMENT ---

# Clean ECI (Target)
df_eci = df_eci[['country_iso3_code', 'year', 'eci_hs92']]
df_eci.columns = ['Country_Code', 'Year', 'ECI']

# Clean GDP (Control)
# Melt from Wide (Years as Columns) to Tall (Year as Row)
df_gdp_clean = df_gdp.melt(id_vars=['Country Code'], var_name='Year', value_name='GDP_Per_Capita')
df_gdp_clean = df_gdp_clean[['Country Code', 'Year', 'GDP_Per_Capita']]
df_gdp_clean.columns = ['Country_Code', 'Year', 'GDP_Per_Capita']
# Force numeric to handle ".." errors
df_gdp_clean['Year'] = pd.to_numeric(df_gdp_clean['Year'], errors='coerce')
df_gdp_clean['GDP_Per_Capita'] = pd.to_numeric(df_gdp_clean['GDP_Per_Capita'], errors='coerce')
df_gdp_clean = df_gdp_clean.dropna()

# Normalization Function (The Rosetta Stone for Country Names)
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

# Aggregate to National Level
df_so_nat = df_so.groupby(['Country_Code', 'year']).sum().reset_index().rename(columns={'year':'Year'})
df_gh_nat = df_gh.groupby(['Country_Code', 'year']).sum().reset_index().rename(columns={'year':'Year'})

# --- 3. THE "SYBER INDEX" CONSTRUCTION ---
print("Forging the Index...")

# Merge Intent + Ground Truth
master = pd.merge(df_so_nat, df_eci, on=['Country_Code', 'Year'], how='inner')
# Merge Construction (Left Join as GitHub data might be sparser)
master = pd.merge(master, df_gh_nat, on=['Country_Code', 'Year'], how='left', suffixes=('_SO', '_GH'))
# Merge Wealth (Control)
master = pd.merge(master, df_gdp_clean, on=['Country_Code', 'Year'], how='inner')

# Fill GitHub NaNs with 0 for calculation safety
master['tech_high_projects'] = master['tech_high_projects'].fillna(0)
master['tech_low_projects'] = master['tech_low_projects'].fillna(0)
master['total_projects'] = master['total_projects'].fillna(0)

# Metric 1: Intent Ratio (Stack Overflow) - "What are they thinking?"
# (High Tech / Low Tech)
master['Score_Intent'] = master['tech_high_count'] / (master['tech_low_count'] + 1)

# Metric 2: Construction Ratio (GitHub) - "What are they building?"
master['Score_Construction'] = master['tech_high_projects'] / (master['tech_low_projects'] + 1)

# Metric 3: Scale - "How loud is the signal?"
# Combine SO and GH volume
master['Total_Volume'] = master['total_posts'] + master['total_projects']

# THE SYBER INDEX (Composite)
# (Average of Quality Scores) * Log(Volume)
# This rewards countries that are BOTH "Smart" and "Big Enough"
master['Syber_Index'] = ((master['Score_Intent'] + master['Score_Construction']) / 2) * np.log(master['Total_Volume'])

# --- 4. THE PREDICTIVE TEST (Lead-Lag) ---
# We shift ECI into the future to see if Index(t) predicts ECI(t+3)
master = master.sort_values(['Country_Code', 'Year'])
master['ECI_Lead_3'] = master.groupby('Country_Code')['ECI'].shift(-3)

# Filter for valid data pairs
test_df = master.dropna(subset=['Syber_Index', 'ECI_Lead_3', 'GDP_Per_Capita'])

# --- 5. THE FINAL VERDICT (Partial Correlation) ---
print("\n" + "="*50)
print(f"FINAL ANALYSIS (N={len(test_df)} Country-Years)")
print("="*50)

# A. Baseline: Does Money predict Complexity? (Beta)
corr_gdp = test_df['GDP_Per_Capita'].corr(test_df['ECI_Lead_3'])
print(f"1. WEALTH Signal (GDP vs Future ECI):      {corr_gdp:.4f}")

# B. Our Model: Does Code predict Complexity? (Raw)
corr_raw = test_df['Syber_Index'].corr(test_df['ECI_Lead_3'])
print(f"2. SYBER Signal (Index vs Future ECI):     {corr_raw:.4f}")

# C. The Test: Does Code predict Complexity INDEPENDENT of Money? (Alpha)
def partial_corr(x, y, control):
    # Regress X on Control
    slope_x, intercept_x, _, _, _ = stats.linregress(control, x)
    res_x = x - (slope_x * control + intercept_x)
    
    # Regress Y on Control
    slope_y, intercept_y, _, _, _ = stats.linregress(control, y)
    res_y = y - (slope_y * control + intercept_y)
    
    # Correlate Residuals
    return res_x.corr(res_y)

corr_partial = partial_corr(test_df['Syber_Index'], test_df['ECI_Lead_3'], test_df['GDP_Per_Capita'])
print(f"3. INNOVATION Signal (Controlled):         {corr_partial:.4f}")
print("-" * 50)

# --- 6. VISUALIZATION ---
plt.figure(figsize=(10, 6))
# We plot the raw correlation for the "Money Shot" visual
sns.regplot(data=test_df, x='Syber_Index', y='ECI_Lead_3', 
            scatter_kws={'alpha':0.5}, line_kws={'color':'purple'})
plt.title(f'The Syber Index: Predicting Future Economic Complexity\n(Correlation: {corr_raw:.2f})')
plt.xlabel('Syber Index (Digital Intent + Construction)')
plt.ylabel('Future Economic Complexity (t+3 Years)')
plt.grid(True, alpha=0.3)

# Label top outliers for storytelling
top_performers = test_df.sort_values('Syber_Index', ascending=False).head(10)
for i, row in top_performers.iterrows():
    if row['Year'] == 2015: # Pick a specific year to avoid clutter
        plt.text(row['Syber_Index'], row['ECI_Lead_3'], row['Country_Code'])

plt.savefig('syber_index_final_v3.png')
print("Saved plot to 'syber_index_final_v3.png'")
test_df.to_csv('final_trinity_dataset.csv', index=False)