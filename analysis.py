import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. LOAD DATA ---
print("Loading datasets...")
df_eci = pd.read_csv('growth_proj_eci_rankings.csv')
df_so = pd.read_csv('stackoverflow_ratios.csv')

# --- 2. CLEAN DATA ---
df_eci = df_eci[['country_iso3_code', 'year', 'eci_hs92']]
df_eci.columns = ['Country_Code', 'Year', 'ECI']

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
df_so_clean = df_so.dropna(subset=['Country_Code'])

# --- 3. AGGREGATE & METRICS ---
df_national = df_so_clean.groupby(['Country_Code', 'year']).sum().reset_index()
df_national = df_national.rename(columns={'year': 'Year'})

# Metric 1: The R&D Ratio (Quality)
df_national['CCI_Ratio'] = df_national['tech_high_count'] / (df_national['tech_low_count'] + 1)

# Metric 2: The Composite Score (Quality * Scale)
# We multiply the Ratio by the Log of Total Posts.
# This rewards countries that are BOTH "Smart" and "Active".
df_national['CCI_Composite'] = df_national['CCI_Ratio'] * np.log(df_national['total_posts'])

# --- 4. MERGE & CREATE LAGS ---
# We merge carefully to ensure we have a continuous timeline
final_df = pd.merge(df_national, df_eci, on=['Country_Code', 'Year'], how='inner')

# Sort for time shifting
final_df = final_df.sort_values(['Country_Code', 'Year'])

# CREATE FUTURE TARGETS (The Time Machine)
# We shift the ECI column UP by N rows (within each country group)
# ECI_Lead_3 means: The ECI of this country 3 years in the future.
final_df['ECI_Lead_1'] = final_df.groupby('Country_Code')['ECI'].shift(-1)
final_df['ECI_Lead_3'] = final_df.groupby('Country_Code')['ECI'].shift(-3)
final_df['ECI_Lead_5'] = final_df.groupby('Country_Code')['ECI'].shift(-5)

# --- 5. THE FINAL RESULTS ---
print("\n" + "="*50)
print("THE SYBERLABS INDEX: PREDICTIVE POWER")
print("="*50)

print("\n--- Test A: Does R&D Ratio predict the future? ---")
# We drop NaNs (which happen at the end of the timeline due to shifting)
print(f"Current Year (t):   {final_df[['CCI_Ratio', 'ECI']].corr().iloc[0,1]:.4f}")
print(f"Future Year (t+1):  {final_df[['CCI_Ratio', 'ECI_Lead_1']].corr().iloc[0,1]:.4f}")
print(f"Future Year (t+3):  {final_df[['CCI_Ratio', 'ECI_Lead_3']].corr().iloc[0,1]:.4f}")
print(f"Future Year (t+5):  {final_df[['CCI_Ratio', 'ECI_Lead_5']].corr().iloc[0,1]:.4f}")

print("\n--- Test B: Does Composite Score (Quality + Scale) predict the future? ---")
print(f"Current Year (t):   {final_df[['CCI_Composite', 'ECI']].corr().iloc[0,1]:.4f}")
print(f"Future Year (t+1):  {final_df[['CCI_Composite', 'ECI_Lead_1']].corr().iloc[0,1]:.4f}")
print(f"Future Year (t+3):  {final_df[['CCI_Composite', 'ECI_Lead_3']].corr().iloc[0,1]:.4f}")
print(f"Future Year (t+5):  {final_df[['CCI_Composite', 'ECI_Lead_5']].corr().iloc[0,1]:.4f}")

# --- 6. VISUALIZE THE BEST RESULT ---
plt.figure(figsize=(12, 8))
# Plotting Composite vs Future ECI (3 years)
sns.regplot(data=final_df, x='CCI_Composite', y='ECI_Lead_3', scatter_kws={'alpha':0.5}, line_kws={'color':'purple'})
plt.title('The Crystal Ball: CCI Composite (t) vs Economic Complexity (t+3)')
plt.xlabel('SyberLabs CCI (Composite)')
plt.ylabel('Future Economic Complexity (ECI in 3 Years)')
plt.grid(True, alpha=0.3)

# Label top performers
df_clean = final_df.dropna(subset=['ECI_Lead_3'])
top_countries = df_clean[df_clean['Year'] == 2015].sort_values('CCI_Composite', ascending=False).head(10)
for i, row in top_countries.iterrows():
    plt.text(row['CCI_Composite'], row['ECI_Lead_3'], row['Country_Code'])

plt.savefig('cci_predictive_model.png')
print("\nSaved predictive model plot to 'cci_predictive_model.png'")