#!/usr/bin/env python
# coding: utf-8

# Article - Cost Minimization Analysis of Digital-first Healthcare Pathways in Primary Care
# Alexandra Dahlberg, Sakari Jukarainen, Taavi Kaartinen, Petja Orre

# Python script converted from a Jupyter notebook for data governance 

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 200) 

import warnings
warnings.filterwarnings('ignore')

# Defining the diagnosis codes for the research
diagnosis_codes = {
    'respiratory': {
        'icd10': {f'J{str(i).zfill(2)}' for i in range(23)},
        'icpc2': {'R05', 'R08', 'R09', 'R21', 'R25', 'R71', 'R72', 'R74', 'R75', 'R76', 'R77', 'R78', 'R80', 'R83'}
    }, 
    'skin': {
        'icd10': {f'L{str(i).zfill(2)}' for i in range(100)},
        'icpc2': {f'S{str(i).zfill(2)}' for i in list(range(1, 77)) + list(range(81, 100))}
    },
    'urinary': {
        'icd10': {'N30.0', 'N30.9'},
        'icpc2': {'U01', 'U02', 'U05', 'U07', 'U13', 'U70', 'U71', 'U72'}
    },
    'eye': {
        'icd10': {'H00', 'H01', 'H04.10', 'H10', 'H11.0', 'H11.3'},
        'icpc2': {f'F{str(i).zfill(2)}' for i in list(range(1, 4)) + list(range(13, 17)) + ['F29'] + list(range(70, 76))}
    },
    'gastro': {
        'icd10': {'A08', 'A09'},
        'icpc2': {'D73'}
    }
}

# ## Episode definition
# Importing file
df_diagnosis = pd.read_csv('R:\path_to_df_diagnosis.csv')
# Columns(['id', 'date', 'unit_name', 'contact_mode', 'profession')]

# Modifying the date column
df_diagnosis['date'] = pd.to_datetime(df_diagnosis['date'],  format='ISO8601')

# Ensure 'icd10' and 'icpc2' are strings
df_diagnosis['icd10'] = df_diagnosis['icd10'].astype(str)
df_diagnosis['icpc2'] = df_diagnosis['icpc2'].astype(str)

# Create a new column to store the diagnosis group
df_diagnosis['diagnosis_group'] = None

# Iterate over each diagnosis group and its codes
for group, codes in diagnosis_codes.items():
    # Assign group for matching ICD-10 codes
    matches_icd10 = df_diagnosis['icd10'].str.startswith(tuple(codes['icd10']), na=False) # Partial matches allowed 
    
    # Assign group for matching ICPC-2 codes
    matches_icpc2 = df_diagnosis['icpc2'].str.startswith(tuple(codes['icpc2']), na=False) # Partial matches allowed
    
    # Update the diagnoosiryhmä column for matches
    df_diagnosis.loc[matches_icd10 | matches_icpc2, 'diagnosis_group'] = group # First based on icd10, if not available then icpc2

# Import dataframe
df_contacts = pd.read_csv('R:\path_to_df_contacts.csv')
# Columns(['id', 'date', 'digi', 'icpc2', 'icd10')]

# Modifying the date column 
df_contacts['date'] = pd.to_datetime(df_contacts['date'],  format='ISO8601')

# Ensure both 'date' columns are in datetime format
df_contacts['date'] = pd.to_datetime(df_contacts['date'], errors='coerce')
df_diagnosis['date'] = pd.to_datetime(df_diagnosis['date'], errors='coerce')

# Perform the merge
df_contacts_diagn = pd.merge(
    df_contacts,
    df_diagnosis,
    how='right',  # Keep all rows from df_contacts
    on=['date', 'id']  # Merge on both 'date' and 'id'
)

# Replace NaN values in the 'profession' column with 'Sairaanhoitaja'
df_contacts_diagn['profession'].fillna('Sairaanhoitaja', inplace=True)
df_contacts_diagn.reset_index(drop=True, inplace=True)

df_contacts_diagn[df_contacts_diagn['profession'].isna()]


filtered_df = df_contacts_diagn[(df_contacts_diagn['digi'] == 'DIGI') & (df_contacts_diagn['diagnosis_group'].notna())]
digi_df = df_contacts_diagn[df_contacts_diagn['digi'] == 'DIGI']
# Calculate the percentage
percentage = (len(filtered_df) / len(digi_df)) * 100

print(f"The percentage of rows where 'digi' is 'DIGI' and 'diagnoosiryhmä' is not None is {percentage:.2f}%")


df_contacts_diagn[df_contacts_diagn['diagnosis_group'].notna()] # N of rows = N of encounters based on diagnosis after exclusion based on diagnosisgroups (for thesis)


# Assuming df_contacts_diagn is already defined
df = pd.DataFrame(df_contacts_diagn)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["diagnosis_group"] = df["diagnosis_group"].replace("nan", np.nan)

# Sort by id and datetime
df = df.sort_values(by=["id", "date"])

# Initialize the episode_number
df["episode_number"] = 0


# Function to assign episode numbers
def assign_episode_numbers(group):
    episode_number = 1
    episode_start_date = group["date"].iloc[0]
    group["episode_number"].iloc[0] = episode_number

    for i in range(1, len(group)):
        current_date = group["date"].iloc[i]
        previous_date = group["date"].iloc[i - 1]

        if (current_date - episode_start_date).days > 14 and (
            current_date - previous_date
        ).days > 14:
            episode_number += 1
            episode_start_date = current_date

        group["episode_number"].iloc[i] = episode_number

    return group


# Apply the function to each group
df = df.groupby("id", group_keys=False).apply(assign_episode_numbers)


# Function to propagate the first non-NaN diagnosis_group value
def propagate_diagnosis_group(group):
    for episode in group["episode_number"].unique():
        episode_group = group[group["episode_number"] == episode]
        # Get the first non-NaN diagnosis_group value
        non_nan_diagnosis = episode_group["diagnosis_group"].dropna()
        if not non_nan_diagnosis.empty:
            first_non_nan_diagnosis = non_nan_diagnosis.iloc[0]
            group.loc[episode_group.index, "diagnosis_group_new"] = (
                first_non_nan_diagnosis
            )
    return group


# Apply the function to propagate diagnosis_group
df = df.groupby("id", group_keys=False).apply(propagate_diagnosis_group)

# Display the DataFrame with the new column
print(df.head())

# only keep first row of each episode
df = df.drop_duplicates(subset=["id", "episode_number"], keep="first")

# Drop those first two weeks
df = df[df['date'] >= '2023-05-15']

df_episodi = df[
    (df['diagnosis_group_new'].notna())
].copy()


# Drop the 'ensi_kontakti' column (should be True for all rows)
df_episodi = df_episodi.drop(columns=['diagnosis_group'])

# Rename the column 'diagnosis_group_new' to 'diagnosis_group'
df_episodi.rename(columns={"diagnosis_group_new": "diagnosis_group"}, inplace=True)

# Create new columns for 'alku_pvm' and 'loppu_pvm'
df_episodi['alku_pvm'] = df_episodi['date']
df_episodi['loppu_pvm'] = df_episodi['date'] + pd.Timedelta(days=14)

# Assign a unique ID to each episode
df_episodi['episode_id'] = range(1, len(df_episodi) + 1)

df_episodi.reset_index(drop=True, inplace=True)


# PRICE-CATALOUG [Päijät-Häme Wellbeing Services County]


# Data for Avosairaanhoito
avosairaanhoito_data = {
    "profession": ["Lääkäri", "Sairaanhoitaja", "Fysioterapeutti"],
    "Vastaanottokäynti": [166, 98, 105],
    "Puhelu": [100, 61, 0], # Varmistettu Päijät-Hämeeltä että samanhintainen
    "Kirje": [0, 0, 0],
    "Sähköinen yhteydenotto": [100, 61, 0],
    "Konsultaatio": [55, 0, 0],
    "Kotikäynti": [320, 176, 0],
    "Asiakkaan asian hoito": [100, 71, 47], #Sähköinen asiointi
    "Muu": [100, 71, 0] # Sähköinen asiointi
}

# Reshape Avosairaanhoito to long format
df_avosairaanhoito_long = df_avosairaanhoito.melt(
    id_vars=["profession"],
    var_name="contact_mode",
    value_name="price"
).dropna(subset=["price"])

# Combine all data into a single DataFrame
df_hinnasto = pd.concat([df_avosairaanhoito_long], ignore_index=True)

# Display the resulting DataFrame
df_hinnasto.reset_index(drop=True, inplace=True)

df_contacts_diagn['contact_mode'].unique()

# Dictionary for replacements
replacements = {
    "Muu": "Muu",
    "Käynti": "Vastaanottokäynti",
    "Konsultaatio": "Konsultaatio",
    "Asiakkaan asian hoito" : "Asiakkaan asian hoito", 
    "Puhelinkontakti" : "Puhelu", 
    "Avustava suorite" : "Muu", 
    "Sähköinen palvelukanava" : "Sähköinen yhteydenotto",
    "Peruuttamaton poisjäänti" : "Muu", 
    "Käynti muualla kuin koti/työ" : "Kotikäynti", 
    "Ryhmätilaisuus" : "Muu",
    "Sähköposti" : "Sähköinen yhteydenotto",
    "Kirje" : "Kirje",
    "Työpaikkakäynti" : "Kotikäynti",
    "Kotikäynti" : "Kotikäynti",
    "Neuvottelu" : "Muu", 
    "Vuodeosastokäynti" : "Osastohoito",

}

df_contacts_diagn["contact_mode"] = df_contacts_diagn["contact_mode"].replace(replacements)


# Dictionary for replacements
replacements = {
    "Sairaanhoitajat ym.": "Sairaanhoitaja",
    "Lääkärit": "Lääkäri",
    "Perushoitajat, lähihoitajat ym.": "Sairaanhoitaja",
    "Ylihoitajat ja osastonhoitajat" : "Sairaanhoitaja", 
    "Lääkärit, proviisorit ja muut terveydenhuollon eri" : "Lääkäri",
    "Sairaanhoitajat, kätilöt ym." : "Sairaanhoitaja", # Tai pitäisik
    "Kampaajat, parturit, kosmestologit ym" : "Kampaaja ym.", # Mitä pitäisi tämän kanssa tehdä? 
    "Fysioterapeutit, toimintaterapeutit ym." : "Fysioterapeutti",
    "Sihteerit, tekstinkäsittelijät ym." : "Erityistyöntekijä", # Tämäkin ?
    "Tietotekniikan tukihenkilöt, operaattorit ym." : "Erityistyöntekijä", #?
}

# Replace values in the 'profession' column
df_contacts_diagn["profession"] = df_contacts_diagn["profession"].replace(replacements)
df_contacts_diagn


df_contacts_diagn = df_contacts_diagn[~df_contacts_diagn['profession'].str.contains('Kampaajat, parturit, kosmetologit ym.', na=False)]

#Reset the index 
df_contacts_diagn.reset_index(drop=True, inplace=True)


# Merge df_hinnasto directly with df_episodi to get the prices for each contact
df_contacts_diagn = df_contacts_diagn.merge(
    df_hinnasto,  
    on=["profession", "contact_mode"],  # Join on profession and contact mode
    how="left"  # Ensure all rows in df_episodi are preserved
)


# Count the total number of episodes under each diagnosis group (diagnoosiryhmä)
diagnosis_group_counts = df_episodi.groupby('diagnosis_group').size().reset_index(name='total_episodes')

# Count the number of digital and non-digital episodes for each diagnosis group
digi_counts = df_episodi[df_episodi['digi'] == 'DIGI'].groupby('diagnosis_group').size().reset_index(name='digital_episodes')
non_digi_counts = df_episodi[df_episodi['digi'] == 'MUU'].groupby('diagnosis_group').size().reset_index(name='non_digital_episodes')

# Merge the counts into one DataFrame
statistics = pd.merge(diagnosis_group_counts, digi_counts, on='diagnosis_group', how='left')
statistics = pd.merge(statistics, non_digi_counts, on='diagnosis_group', how='left')

# Fill NaN values with 0 (for cases where there are no digital or non-digital episodes)
statistics = statistics.fillna(0)



# ## Laboratoriodata

df_lab = pd.read_csv('R:\path_to_df__lab.csv')
# Columns(['id', 'date', 'code_kuntaliitto', 'abbreviation', 'name', 'price')]

df_lab = df_lab.drop(columns=['price'])

# Lab hinnasto [Fimlab]

lab_hinnat = {
    'koodi': [
        1,2,3...
    ],
    'price': [
        1,2,3...

    ]
}

# Create the DataFrame
df_lab_hinnat = pd.DataFrame(lab_hinnat)

df_lab = df_lab.merge(df_lab_hinnat, left_on='code_kuntaliitto', right_on='koodi', how='left')
df_lab['price'] = df_lab['price'].fillna(0)
df_lab = df_lab.drop(columns=['koodi'])

# ## Kuvantamisdata


df_rtg = pd.read_csv('R:\path_to_df_rtg.csv')
# Columns(['id', 'date', 'code', 'name', 'price')]
df_rtg = df_rtg.drop(columns=['price'])


# PRICES 

data = [
    {"code": "XXXX", "amount": 1, "price_total": 1.00},
    ...,
]

# Create the DataFrame
df = pd.DataFrame(data)

# Calculate the price per item
df['price'] = df['price_total'] / df['amount']

# Drop unnecessary columns if needed (e.g., 'amount', 'price_total')
df_rtg_hinnat = df[['code', 'price']]

# Remove duplicates and keep the first occurrence
df_rtg_hinnat = df_rtg_hinnat.drop_duplicates(subset="code", keep="first")
df_rtg_hinnat = df_rtg_hinnat.rename(columns={'code': 'koodi'})


df_rtg = df_rtg.merge(df_rtg_hinnat, left_on='code', right_on='koodi', how='left')
df_rtg = df_rtg.drop(columns=['koodi'])
df_rtg['price'] = df_rtg['price'].fillna(0)


# ## -> Volyymidatasta outcome-muuttajat
# Episodi_df:issä jokainen episodi määritelty, kustannus_df:it (df_contacts & df_lab) perusteella kustannuksien aggregointi näihin episodeihin.

# Ensure consistent datetime formats, just to be sure one more time
df_episodi['alku_pvm'] = df_episodi['alku_pvm'].dt.date      # Date without time
df_episodi['loppu_pvm'] = df_episodi['loppu_pvm'].dt.date      # Date without time

df_episodi['alku_pvm'] = pd.to_datetime(df_episodi['alku_pvm'])
df_episodi['loppu_pvm'] = pd.to_datetime(df_episodi['loppu_pvm'])
df_lab['date'] = pd.to_datetime(df_lab['date'])
df_rtg['date'] = pd.to_datetime(df_rtg['date'])


df_contacts_diagn['time'] = df_contacts_diagn['date'].dt.time      # Time without date
df_contacts_diagn['date'] = df_contacts_diagn['date'].dt.date      # Date without time

df_contacts_diagn['date'] = pd.to_datetime(df_contacts_diagn['date']) # Date as datetime, varmuuden vuoksi


# Ensure `price` column is properly aligned in df_lab and df_rtg
df_lab['price'] = df_lab['price'].fillna(0)
df_rtg['price'] = df_rtg['price'].fillna(0)

# Contacts
df_contacts_diagn_merged = df_contacts_diagn.merge(
    df_episodi[['id', 'episode_id', 'alku_pvm', 'loppu_pvm']], 
    on='id', 
    how='inner'
)
df_contacts_diagn_filtered = df_contacts_diagn_merged[
    (df_contacts_diagn_merged['date'] >= df_contacts_diagn_merged['alku_pvm']) &
    (df_contacts_diagn_merged['date'] <= df_contacts_diagn_merged['loppu_pvm'])
]

# Repeat the process for labs
df_lab_merged = df_lab.merge(
    df_episodi[['id', 'episode_id', 'alku_pvm', 'loppu_pvm']], 
    on='id', 
    how='inner'
)
df_lab_filtered = df_lab_merged[
    (df_lab_merged['date'] >= df_lab_merged['alku_pvm']) &
    (df_lab_merged['date'] <= df_lab_merged['loppu_pvm'])
]

# Repeat the process for rtg
df_rtg_merged = df_rtg.merge(
    df_episodi[['id', 'episode_id', 'alku_pvm', 'loppu_pvm']], 
    on='id', 
    how='inner'
)
df_rtg_filtered = df_rtg_merged[
    (df_rtg_merged['date'] >= df_rtg_merged['alku_pvm']) &
    (df_rtg_merged['date'] <= df_rtg_merged['loppu_pvm'])
]

# Aggregate costs and counts by episode_id
contacts_costs = df_contacts_diagn_filtered.groupby('episode_id')['price'].sum().rename('contacts_cost')
labs_costs = df_lab_filtered.groupby('episode_id')['price'].sum().rename('labs_cost')
rtg_costs = df_rtg_filtered.groupby('episode_id')['price'].sum().rename('rtg_cost')

# Count number of contacts, labs, and rtg per episode_id
n_contacts = df_contacts_diagn_filtered.groupby('episode_id').size().rename('n_contacts')
n_labs = df_lab_filtered.groupby('episode_id').size().rename('n_labs')
n_rtg = df_rtg_filtered.groupby('episode_id').size().rename('n_rtg')


# Merge aggregated data into df_episodi
df_episodi = df_episodi.merge(contacts_costs, on='episode_id', how='left')
df_episodi = df_episodi.merge(labs_costs, on='episode_id', how='left')
df_episodi = df_episodi.merge(rtg_costs, on='episode_id', how='left')
df_episodi = df_episodi.merge(n_contacts, on='episode_id', how='left')
df_episodi = df_episodi.merge(n_labs, on='episode_id', how='left')
df_episodi = df_episodi.merge(n_rtg, on='episode_id', how='left')

# Fill missing values in cost and count columns
df_episodi['contacts_cost'] = df_episodi['contacts_cost'].fillna(0)
df_episodi['labs_cost'] = df_episodi['labs_cost'].fillna(0)
df_episodi['rtg_cost'] = df_episodi['rtg_cost'].fillna(0)
df_episodi['n_contacts'] = df_episodi['n_contacts'].fillna(0).astype(int)
df_episodi['n_labs'] = df_episodi['n_labs'].fillna(0).astype(int)
df_episodi['n_rtg'] = df_episodi['n_rtg'].fillna(0).astype(int)

# Calculate total costs
df_episodi['kustannus'] = df_episodi['contacts_cost'] + df_episodi['labs_cost'] + df_episodi['rtg_cost']

filtered_rtg = df_rtg_filtered[df_rtg_filtered['price'] == 0]
unique_code_name = filtered_rtg[['code', 'name']].drop_duplicates()
unique_code_name_list = unique_code_name.to_dict('records')


# Import data on covariates
df_covariates = pd.read_csv('R:\path_to_df_covariates.csv')
# Columns(['id', 'age', 'sex', 'cci', 'n_laakari')]

# Merge df_covariates into df_episodi based on the id
df_episodi = df_episodi.merge(
    df_covariates,
    on='id',  # Match based on the 'id' column
    how='left'  # Use left join to keep all episodes in df_episodi
)

df_episodi= df_episodi.drop_duplicates()
# Reset the episode_id to start from 1
df_episodi['episode_id'] = range(1, len(df_episodi) + 1)



# Create a new column 'hoitoketju' based on the condition in 'digi'
df_episodi.rename(columns={'digi': 'hoitoketju'}, inplace=True)

# Update the values in the 'hoitoketju' column
df_episodi['hoitoketju'] = df_episodi['hoitoketju'].apply(lambda x: 'digi' if x == 'DIGI' else 'physical' if x == 'MUU' else x)

# Ryhmien muut rajaukset

# Filter out rows where diagnosiryhmä is 'urinary' and age > 65 or sex is 'Mies'
df_episodi = df_episodi[
    ~((df_episodi['diagnosis_group'] == 'urinary') & 
      ((df_episodi['age'] > 65)))
]

df_eye = df_episodi[df_episodi['diagnosis_group'] == 'eye'].copy()
df_skin = df_episodi[df_episodi['diagnosis_group'] == 'skin'].copy()
df_urinary = df_episodi[df_episodi['diagnosis_group'] == 'urinary'].copy()
df_respiratory = df_episodi[df_episodi['diagnosis_group'] == 'respiratory'].copy()
df_gastro = df_episodi[df_episodi['diagnosis_group'] == 'gastro'].copy()

# Numbers for my thesis
df_kustannus_zero = df_episodi[df_episodi['kustannus'] == 0]
# < 20

# Exclusion
df_episodi = df_episodi[df_episodi['kustannus'] != 0].copy() 


# ## (Tables for fun)

## Define age bins and labels
age_bins = [0, 20, 40, 60, 80, 100]
age_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']

# Create a new column for age group in the dataframe
df_episodi['age_group'] = pd.cut(df_episodi['age'], bins=age_bins, labels=age_labels, right=True)

# Create a summary table
summary_table = pd.DataFrame()

# Count patients in each group
summary_table['All Patients'] = df_episodi.groupby('age_group')['episode_id'].nunique()

# Count for Physical
physical_patients = df_episodi[df_episodi['hoitoketju'] == 'physical']
summary_table['Physical'] = physical_patients.groupby('age_group')['episode_id'].nunique()

# Count for Digital
digital_patients = df_episodi[df_episodi['hoitoketju'] == 'digi']
summary_table['Digital'] = digital_patients.groupby('age_group')['episode_id'].nunique()

# Calculate percentage digital within each age group
summary_table['% Digital'] = (summary_table['Digital'] / summary_table['All Patients']) * 100

# Count Male and Female patients
summary_table.loc['Male'] = [
    df_episodi[df_episodi['sex'] == 'Mies']['episode_id'].nunique(),
    physical_patients[physical_patients['sex'] == 'Mies']['episode_id'].nunique(),
    digital_patients[digital_patients['sex'] == 'Mies']['episode_id'].nunique(),
    (digital_patients[digital_patients['sex'] == 'Mies']['episode_id'].nunique() /
     df_episodi[df_episodi['sex'] == 'Mies']['episode_id'].nunique()) * 100
]

summary_table.loc['Female'] = [
    df_episodi[df_episodi['sex'] == 'Nainen']['episode_id'].nunique(),
    physical_patients[physical_patients['sex'] == 'Nainen']['episode_id'].nunique(),
    digital_patients[digital_patients['sex'] == 'Nainen']['episode_id'].nunique(),
    (digital_patients[digital_patients['sex'] == 'Nainen']['episode_id'].nunique() /
     df_episodi[df_episodi['sex'] == 'Nainen']['episode_id'].nunique()) * 100
]

# Total number of patients in each group
summary_table.loc['N of Episodes'] = [
    df_episodi['episode_id'].nunique(),
    physical_patients['episode_id'].nunique(),
    digital_patients['episode_id'].nunique(),
    (digital_patients['episode_id'].nunique() / df_episodi['episode_id'].nunique()) * 100
]

summary_table



# Grouping the data
def calculate_statistics(data, column, group=None):
    """Helper function to calculate mean and standard deviation."""
    if group:
        group_data = data[data['hoitoketju'] == group]
        return {
            "Mean": group_data[column].mean(),
            "SD": group_data[column].std(),
            "N": group_data.shape[0]
        }
    else:
        return {
            "Mean": data[column].mean(),
            "SD": data[column].std(),
            "N": data.shape[0]
        }

# Initialize dictionary for storing results
table_stats = {
    "All patients": {},
    "Physical": {},
    "Digital": {}
}

# Define age bins
age_bins = [0, 20, 40, 60, 80, 100]
age_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']

# Add age group
df_episodi['age_group'] = pd.cut(df_episodi['age'], bins=age_bins, labels=age_labels, right=True)

# Calculate stats for Age groups
for group in age_labels:
    table_stats["All patients"][group] = calculate_statistics(
        df_episodi[df_episodi['age_group'] == group], 'age'
    )
    table_stats["Physical"][group] = calculate_statistics(
        df_episodi[df_episodi['age_group'] == group], 'age', group='physical'
    )
    table_stats["Digital"][group] = calculate_statistics(
        df_episodi[df_episodi['age_group'] == group], 'age', group='digi'
    )

# Calculate stats for gender
for gender in ['Mies', 'Nainen']:
    table_stats["All patients"][gender] = calculate_statistics(
        df_episodi[df_episodi['sex'] == gender], 'age'
    )
    table_stats["Physical"][gender] = calculate_statistics(
        df_episodi[(df_episodi['sex'] == gender) & (df_episodi['hoitoketju'] == 'physical')], 'age'
    )
    table_stats["Digital"][gender] = calculate_statistics(
        df_episodi[(df_episodi['sex'] == gender) & (df_episodi['hoitoketju'] == 'digi')], 'age'
    )

# Create DataFrame to display results
table_df = pd.DataFrame.from_dict(table_stats, orient='index')
table_df = table_df.applymap(lambda x: f"{x['Mean']:.2f} ± {x['SD']:.2f}" if isinstance(x, dict) else x)

# Display the final table
table_df



from scipy.stats import chi2_contingency

# Create a summary table for diagnosis groups
diagnosis_groups = df_episodi['diagnosis_group'].unique()

summary_table = pd.DataFrame(columns=[
    'All Patients (N)', 'All Patients (%)',
    'Physical (N)', 'Physical (%)',
    'Digital (N)', 'Digital (%)',
    'P-value'
])

# Total number of patients
total_patients = len(df_episodi)

for group in diagnosis_groups:
    # Filter for the diagnosis group
    group_data = df_episodi[df_episodi['diagnosis_group'] == group]
    
    # All patients
    all_patients_n = len(group_data)
    all_patients_pct = (all_patients_n / total_patients) * 100

    # Physical group
    physical_data = group_data[group_data['hoitoketju'] == 'physical']
    physical_n = len(physical_data)
    physical_pct = (physical_n / total_patients) * 100

    # Digital group
    digital_data = group_data[group_data['hoitoketju'] == 'digi']
    digital_n = len(digital_data)
    digital_pct = (digital_n / total_patients) * 100

    # Chi-square test for independence
    contingency_table = [
        [physical_n, digital_n],
        [len(group_data[group_data['hoitoketju'] != 'physical']),
         len(group_data[group_data['hoitoketju'] != 'digi'])]
    ]

    print(f"Contingency Table for group '{group}': {contingency_table}")

    _, p_value, _, _ = chi2_contingency(contingency_table)

    # Add row to the summary table
    summary_table.loc[group] = [
        all_patients_n, round(all_patients_pct, 2),
        physical_n, round(physical_pct, 2),
        digital_n, round(digital_pct, 2),
        '{:.2e}'.format(p_value)
    ]

summary_table.round(2)



from scipy.stats import ttest_ind

# Ensure dates are in datetime format
df_episodi['alku_pvm'] = pd.to_datetime(df_episodi['alku_pvm'])
df_episodi['loppu_pvm'] = pd.to_datetime(df_episodi['loppu_pvm'])

# Calculate the total encounters (contacts) for each episode
df_episodi['total_encounters'] = df_episodi['n_contacts']

# Group by diagnosis group and treatment type (hoitoketju)
mean_encounters_summary = (
    df_episodi.groupby(['diagnosis_group', 'hoitoketju'])['total_encounters']
    .mean()
    .reset_index()
)

# Pivot the table to separate columns for digital and physical encounters
pivot_summary = mean_encounters_summary.pivot(
    index='diagnosis_group',
    columns='hoitoketju',
    values='total_encounters'
).reset_index()

# Rename columns for clarity
pivot_summary.columns = ['diagnosis_group', 'mean_digi_encounters', 'mean_physical_encounters']

# Calculate p-values using independent t-tests for each diagnosis group
p_values = []
for group in pivot_summary['diagnosis_group']:
    group_data = df_episodi[df_episodi['diagnosis_group'] == group]
    
    digi_data = group_data[group_data['hoitoketju'] == 'digi']['total_encounters']
    physical_data = group_data[group_data['hoitoketju'] == 'physical']['total_encounters']
    
    if len(digi_data) > 1 and len(physical_data) > 1:  # Ensure sufficient data points
        _, p = ttest_ind(digi_data, physical_data, equal_var=False)  # Welch's t-test
        p_values.append(p)
    else:
        p_values.append(None)

# Add p-values to the summary table
pivot_summary['p_value'] = p_values

# Display the result
pivot_summary.round(2)



# Calculate mean price for all episodes in each diagnosis group
mean_price_all = (
    df_episodi
    .groupby('diagnosis_group')['kustannus']
    .mean()
    .reset_index(name='mean_price_all')
)

# Calculate mean price for digital episodes in each diagnosis group
mean_price_digi = (
    df_episodi[df_episodi['hoitoketju'] == 'digi']
    .groupby('diagnosis_group')['kustannus']
    .mean()
    .reset_index(name='mean_price_digi')
)

# Calculate mean price for physical episodes in each diagnosis group
mean_price_physical = (
    df_episodi[df_episodi['hoitoketju'] == 'physical']
    .groupby('diagnosis_group')['kustannus']
    .mean()
    .reset_index(name='mean_price_physical')
)

# Merge the mean prices into one DataFrame
mean_price_statistics = mean_price_all.merge(mean_price_digi, on='diagnosis_group', how='left')
mean_price_statistics = mean_price_statistics.merge(mean_price_physical, on='diagnosis_group', how='left')

# Fill NaN values with 0 for missing mean prices
mean_price_statistics = mean_price_statistics.fillna(0)

# Display the final statistics
mean_price_statistics.round(2)



# ## Analytics before PSM


# Calculate statistics for matched data
def calculate_statistics(data, group_column, treatment_column):
    # Initialize a results dictionary to store statistics
    results = []

    # Process each diagnosis group
    for group, group_data in data.groupby(group_column):
        # Process each treatment group (digi and physical)
        for treatment, treatment_data in group_data.groupby(treatment_column):
            # Calculate statistics
            mean_age = treatment_data['age'].mean()
            mean_n_contacts = treatment_data['n_contacts'].mean()
            mean_n_labs = treatment_data['n_labs'].mean()
            mean_n_rtg = treatment_data['n_rtg'].mean()*100
            mean_contacts_costs = treatment_data['contacts_cost'].mean()
            mean_labs_costs = treatment_data['labs_cost'].mean()
            mean_rtg_costs = treatment_data['rtg_cost'].mean()
            mean_kustannus = treatment_data['kustannus'].mean()
            percent_female = (treatment_data['sex'] == 'Nainen').mean() * 100  # Use 1 for "Nainen"
            percent_male = (treatment_data['sex'] == 'Mies').mean() * 100  # Use 0 for "Mies"
            count_patients = len(treatment_data)

            # Append the results
            results.append({
                'diagnosis_group': group,
                'treatment': treatment,
                'mean_age': mean_age,
                'mean_n_contacts': mean_n_contacts,
                'mean_n_labs': mean_n_labs,
                'mean_n_rtg_per_100': mean_n_rtg,
                'mean_contacts_costs': mean_contacts_costs,
                'mean_labs_costs' : mean_labs_costs,
                'mean_rtg_costs' : mean_rtg_costs,
                'mean_kustannus': mean_kustannus,
                'percent_female': percent_female,
                'percent_male': percent_male,
                'count_patients': count_patients
            })

    # Convert results to a DataFrame
    stats_df = pd.DataFrame(results)
    return stats_df

statistics = calculate_statistics(df_episodi, group_column='diagnosis_group', treatment_column='hoitoketju')

statistics.round(2)


from scipy.stats import ttest_ind

def perform_ttest_by_group(matched_data, outcome_column, group_column, treatment_column):
    """
    Perform independent two-sample t-test for each group in the matched data.
    
    Parameters:
    -----------
    matched_data : pandas.DataFrame
        DataFrame containing matched data with group, treatment, and outcome columns.
    outcome_column : str
        Name of the outcome column to test (e.g., 'kustannus').
    group_column : str
        Name of the column indicating groups (e.g., 'diagnoosiryhmä').
    treatment_column : str
        Name of the treatment column (e.g., 'hoitoketju').
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with t-test results for each group.
    """
    results = []

    # Group by the specified group column (e.g., 'diagnoosiryhmä')
    for group, group_data in matched_data.groupby(group_column):
        print(f"Processing group: {group}")
        
        # Separate treatment and control groups
        treatment_group = group_data[group_data[treatment_column] == 'digi'][outcome_column]
        control_group = group_data[group_data[treatment_column] == 'physical'][outcome_column]

        # Check if there are enough samples in both groups
        if len(treatment_group) < 2 or len(control_group) < 2:
            print(f"Skipping group {group} due to insufficient data.")
            continue

        # Perform independent two-sample t-test
        t_stat, p_value = ttest_ind(
            treatment_group,
            control_group,
            equal_var=False  # Use Welch's t-test to account for unequal variances
        )

        # Store results
        results.append({
            group_column: group,
            'mean_digital (euro)': treatment_group.mean(),
            'mean_physical (euro)': control_group.mean(),
            't_stat': t_stat,
            'p_value': p_value,
            'n_digital': len(treatment_group),
            'n_physical': len(control_group)
        })

    # Convert results to a DataFrame for easy interpretation
    return pd.DataFrame(results)

# Example Usage
# Perform t-tests for each diagnoosiryhmä
outcome_column = 'kustannus'
group_column = 'diagnosis_group'
treatment_column = 'hoitoketju'

ttest_results = perform_ttest_by_group(
    matched_data=df_episodi,
    outcome_column=outcome_column,
    group_column=group_column,
    treatment_column=treatment_column
)

# Display results
ttest_results


# Enhance results with savings calculations
def calculate_savings(ttest_results):
    ttest_results['percentage_saved'] = (
        (ttest_results['mean_physical (euro)'] - ttest_results['mean_digital (euro)']) /
        ttest_results['mean_physical (euro)'] * 100
    ).round(2)

    ttest_results['total_savings'] = (
        (ttest_results['mean_physical (euro)'] - ttest_results['mean_digital (euro)']) *
        ttest_results['n_digital']
    ).round(2)

    # Calculate overall savings and append to the DataFrame
    overall_mean_physical = (
        ttest_results['mean_physical (euro)'] * ttest_results['n_physical']
    ).sum() / ttest_results['n_physical'].sum()

    overall_mean_digital = (
        ttest_results['mean_digital (euro)'] * ttest_results['n_digital']
    ).sum() / ttest_results['n_digital'].sum()

    overall_savings = overall_mean_physical - overall_mean_digital
    overall_percentage_saved = (overall_savings / overall_mean_physical * 100).round(2)
    overall_total_savings = (
        overall_savings * ttest_results['n_digital'].sum()
    ).round(2)

    # Append overall results as a new row
    overall_summary = pd.DataFrame({
        'diagnosis_group': ['Overall'],
        'mean_digital (euro)': [overall_mean_digital],
        'mean_physical (euro)': [overall_mean_physical],
        'percentage_saved': [overall_percentage_saved],
        'total_savings': [overall_total_savings],
        'n_digital': [ttest_results['n_digital'].sum()],
        'n_physical': [ttest_results['n_physical'].sum()],
        't_stat': [None],
        'p_value': [None]
    })

    ttest_results = pd.concat([ttest_results, overall_summary], ignore_index=True)

    return ttest_results


# Apply savings calculation
business_summary = calculate_savings(ttest_results)

# Display business summary
business_summary



from scipy.stats import ttest_ind
import pandas as pd

def calculate_statistics_and_ttests(data, group_column, treatment_column, cost_columns):
    """
    Calculates statistics and performs t-tests for matched data.
    
    Parameters:
    - data: DataFrame containing matched data.
    - group_column: The column name for diagnosis groups.
    - treatment_column: The column name indicating treatment types ('digi' or 'physical').
    - cost_columns: List of column names for cost-related data.
    
    Returns:
    - stats_df: DataFrame with aggregated statistics.
    - ttest_results: DataFrame with t-test results.
    """
    # Initialize results dictionary for statistics
    stats_results = []
    ttest_results = []

    # Process each diagnosis group
    for group, group_data in data.groupby(group_column):
        # Process each treatment group (digital and physical)
        for treatment, treatment_data in group_data.groupby(treatment_column):
            # Calculate statistics
            mean_age = treatment_data['age'].mean()
            mean_n_contacts = treatment_data['n_contacts'].mean()
            mean_n_labs = treatment_data['n_labs'].mean()
            mean_n_rtg = treatment_data['n_rtg'].mean() * 100
            mean_contacts_costs = treatment_data['contacts_cost'].mean()
            mean_labs_costs = treatment_data['labs_cost'].mean()
            mean_rtg_costs = treatment_data['rtg_cost'].mean()
            mean_kustannus = treatment_data['kustannus'].mean()
            percent_female = (treatment_data['sex'] == 1).mean() * 100
            percent_male = (treatment_data['sex'] == 0).mean() * 100
            count_patients = len(treatment_data)

            # Append statistics to results
            stats_results.append({
                'diagnosis_group': group,
                'treatment': treatment,
                'mean_age': mean_age,
                'mean_n_contacts': mean_n_contacts,
                'mean_n_labs': mean_n_labs,
                'mean_n_rtg_per_100': mean_n_rtg,
                'mean_contacts_costs': mean_contacts_costs,
                'mean_labs_costs': mean_labs_costs,
                'mean_rtg_costs': mean_rtg_costs,
                'mean_kustannus': mean_kustannus,
                'percent_female': percent_female,
                'percent_male': percent_male,
                'count_patients': count_patients
            })

        # Perform t-tests for the specified cost columns
        digital_data = group_data[group_data[treatment_column] == 'digi']
        traditional_data = group_data[group_data[treatment_column] == 'physical']

        for cost_column in cost_columns:
            if cost_column in data.columns:
                digital_values = digital_data[cost_column]
                traditional_values = traditional_data[cost_column]
                
                # Perform independent t-test
                t_stat, p_value = ttest_ind(digital_values, traditional_values, equal_var=False)
                
                # Append t-test results
                ttest_results.append({
                    'diagnosis_group': group,
                    'cost_category': cost_column,
                    'mean_digital': digital_values.mean(),
                    'mean_traditional': traditional_values.mean(),
                    't_stat': t_stat,
                    'p_value': p_value
                })

    # Convert results to DataFrames
    stats_df = pd.DataFrame(stats_results)
    ttest_results_df = pd.DataFrame(ttest_results)

    return stats_df, ttest_results_df

# Specify cost columns for t-tests
cost_columns = ['contacts_cost', 'labs_cost', 'rtg_cost', 'kustannus']

# Run the integrated function on matched data
stats_summary, ttest_summary = calculate_statistics_and_ttests(
    data=df_episodi,
    group_column='diagnosis_group',
    treatment_column='hoitoketju',
    cost_columns=cost_columns
)


ttest_summary.round(4)



from scipy.stats import ttest_ind

# Grouping data into digital and physical groups
physical_data = df_episodi[df_episodi['hoitoketju'] == 'physical']
digital_data = df_episodi[df_episodi['hoitoketju'] == 'digi']

# Function to calculate mean costs and p-value
def calculate_cost_stats(physical_cost, digital_cost):
    # Calculate mean for both groups
    physical_mean = physical_cost.mean()
    digital_mean = digital_cost.mean()
    
    # Perform independent t-test to get p-value
    _, p_value = ttest_ind(physical_cost, digital_cost, equal_var=False)
    
    return physical_mean, digital_mean, p_value

# Initialize an empty dictionary to store the results
table_stats = {}

# Visit Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['contacts_cost'], digital_data['contacts_cost']
)
table_stats['Visit costs'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}

# Laboratory Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['labs_cost'], digital_data['labs_cost']
)
table_stats['Laboratory costs'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}

# Imaging Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['rtg_cost'], digital_data['rtg_cost']
)
table_stats['Imaging costs'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}


# Total Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['kustannus'], digital_data['kustannus']
)
table_stats['Total'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}

# Create a DataFrame to display the results
cost_table_df = pd.DataFrame.from_dict(table_stats, orient='index')

# Formatting the results to two decimal points for readability
cost_table_df = cost_table_df.applymap(lambda x: f"{x:.10e}" if isinstance(x, float) else x)

# Display the resulting table
print(cost_table_df)



# ## Propensity score matching


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder


def nearest_neighbor_matching(data, treatment, propensity_scores, caliper):
    """
    Perform nearest neighbor matching without replacement.

    Parameters:
    -----------
    data: pandas DataFrame
        Original dataset
    treatment: numpy array
        Binary treatment indicator
    propensity_scores: numpy array
        Estimated propensity scores
    caliper: float
        Multiplier for the standard deviation of the logit of propensity scores

    Returns:
    --------
    pandas DataFrame with matched samples
    """

    # Calculate logit of propensity scores
    # Add small constant to avoid log(0) or log(1)
    eps = 1e-8
    logit_ps = np.log(propensity_scores + eps) - np.log(1 - propensity_scores + eps)

    # Calculate the standard deviation of the logit of the propensity score
    logit_ps_std = np.std(logit_ps)

    # Calculate the actual caliper width
    caliper_width = caliper * logit_ps_std

    print(f"Caliper Width for this run: {caliper_width}")

    # Create array of indices and logit propensity scores
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    treated_logit_ps = logit_ps[treated_idx]
    control_logit_ps = logit_ps[control_idx]

    # Initialize arrays to store matches
    matched_treated = []
    matched_control = []

    # For each treated unit, find the closest control unit
    for i, treated_index in enumerate(treated_idx):
        logit_ps_treated = treated_logit_ps[i]

        # Calculate distance between treated unit and all control units
        distances = np.abs(logit_ps_treated - control_logit_ps)

        # Find the closest control unit within caliper
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]

        if min_distance <= caliper_width:
            matched_treated.append(treated_index)
            matched_control.append(control_idx[min_distance_idx])

            # Remove matched control unit
            control_idx = np.delete(control_idx, min_distance_idx)
            control_logit_ps = np.delete(control_logit_ps, min_distance_idx)

    # Combine matched indices
    matched_indices = np.concatenate([matched_treated, matched_control])

    return data.iloc[matched_indices].copy()



# treatment ~ age +〖age〗^2 +〖age〗^3 + n_MD +〖n_MD〗^2+〖n_MD〗^3 + sex + CCI    
# 
# n_MD = n_lääkäri = lääkärikäyntien määrä avosairaanhoidossa aikavälillä "2021-05-01"- "2023-05-01" (2v ennen seurannan alkua)
# 


def perform_psm_with_polynomial_features(data, group_column, treatment_column, covariate_columns, caliper):
    """
    Perform propensity score matching for each group in the data, with polynomial features 

    Parameters:
    -----------
    data: pandas DataFrame
        The dataset containing the data.
    group_column: str
        The column specifying the groups (e.g., diagnoosiryhmä).
    treatment_column: str
        The column specifying the treatment indicator.
    covariate_columns: list
        The list of covariate column names to use in the logistic regression.
    caliper: float
        The caliper for nearest neighbor matching.

    Returns:
    --------
    pandas DataFrame containing the matched data.
    """
    results = []

    for group in data[group_column].unique():
        print(f"Processing group: {group}")
        group_data = data[data[group_column] == group].copy()

        # Encode categorical variables if necessary
        if 'sex' in covariate_columns and group_data['sex'].dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            group_data['sex'] = encoder.fit_transform(group_data['sex'])

        # Create polynomial features for 'age' and 'n_lääkäri'
        poly = PolynomialFeatures(degree=3, include_bias=False)

        # Generate polynomial terms for 'age' and 'n_lääkäri'
        age_poly = poly.fit_transform(group_data[['age']])[:, :3]  # age, age^2, age^3
        n_laakari_poly = poly.fit_transform(group_data[['n_laakari']])[:, :3]  # n_lääkäri, n_lääkäri^2, n_lääkäri^3

        # Combine polynomial features with other covariates
        age_poly_df = pd.DataFrame(age_poly, columns=['age', 'age^2', 'age^3'], index=group_data.index)
        n_lääkäri_poly_df = pd.DataFrame(n_laakari_poly, columns=['n_laakari', 'n_laakari^2', 'n_laakari^3'], index=group_data.index)

        # Combine all covariates
        covariates = pd.concat([group_data[['sex', 'cci']], age_poly_df, n_lääkäri_poly_df], axis=1)

        # Define the treatment variable
        treatment = (group_data[treatment_column] == 'digi').astype(int)

        # Fit logistic regression model to calculate propensity scores
        from sklearn.linear_model import LogisticRegression
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(covariates, treatment)
        propensity_scores = log_reg.predict_proba(covariates)[:, 1]

        # Perform nearest neighbor matching
        matched_group_data = nearest_neighbor_matching(
            data=group_data,
            treatment=treatment.values,
            propensity_scores=propensity_scores,
            caliper=caliper
        )

        results.append(matched_group_data)

    # Combine matched data for all groups
    matched_data = pd.concat(results, ignore_index=True)
    return matched_data




covariate_columns = ['sex', 'age', 'cci', 'n_laakari']  # Base columns (additional polynomial terms will be created automatically)

# Perform PSM for each diagnoosiryhmä
matched_data = perform_psm_with_polynomial_features(
    data=df_episodi,  
    group_column='diagnosis_group',
    treatment_column='hoitoketju',  
    covariate_columns=covariate_columns,
    caliper=0.2 
)

matched_data




# Define age bins and labels
age_bins = [0, 20, 40, 60, 80, 100]
age_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']

# Create a new column for age group in the dataframe
matched_data['age_group'] = pd.cut(matched_data['age'], bins=age_bins, labels=age_labels, right=True)

# Create a summary table
summary_table = pd.DataFrame()

# Calculate 'All Episodes' per age group
summary_table['All Episodes'] = matched_data.groupby('age_group')['episode_id'].nunique()

# Count patients in physical and digital pathways
physical_patients = matched_data[matched_data['hoitoketju'] == 'physical']
digital_patients = matched_data[matched_data['hoitoketju'] == 'digi']

summary_table['Physical'] = physical_patients.groupby('age_group')['episode_id'].nunique()
summary_table['Digital'] = digital_patients.groupby('age_group')['episode_id'].nunique()

# Calculate percentage digital within each age group
summary_table['% Digital'] = (
    (summary_table['Digital'] / summary_table['All Episodes']) * 100
).fillna(0)

# Count Male and Female patients
summary_table.loc['Male', ['All Episodes', 'Physical', 'Digital']] = [
    matched_data[matched_data['sex'] == 0]['episode_id'].nunique(),
    physical_patients[physical_patients['sex'] == 0]['episode_id'].nunique(),
    digital_patients[digital_patients['sex'] == 0]['episode_id'].nunique()
]

summary_table.loc['Male', '% Digital'] = (
    summary_table.loc['Male', 'Digital'] / summary_table.loc['Male', 'All Episodes'] * 100
    if summary_table.loc['Male', 'All Episodes'] > 0 else 0
)

summary_table.loc['Female', ['All Episodes', 'Physical', 'Digital']] = [
    matched_data[matched_data['sex'] == 1]['episode_id'].nunique(),
    physical_patients[physical_patients['sex'] == 1]['episode_id'].nunique(),
    digital_patients[digital_patients['sex'] == 1]['episode_id'].nunique()
]

summary_table.loc['Female', '% Digital'] = (
    summary_table.loc['Female', 'Digital'] / summary_table.loc['Female', 'All Episodes'] * 100
    if summary_table.loc['Female', 'All Episodes'] > 0 else 0
)

# Calculate total number of episodes in each group
summary_table.loc['N of Episodes', ['All Episodes', 'Physical', 'Digital']] = [
    matched_data['episode_id'].nunique(),
    physical_patients['episode_id'].nunique(),
    digital_patients['episode_id'].nunique()
]

summary_table.loc['N of Episodes', '% Digital'] = (
    summary_table.loc['N of Episodes', 'Digital'] / summary_table.loc['N of Episodes', 'All Episodes'] * 100
    if summary_table.loc['N of Episodes', 'All Episodes'] > 0 else 0
)

# Display the updated summary table
print(summary_table)




# Grouping the data
def calculate_statistics(data, column, group=None):
    """Helper function to calculate mean and standard deviation."""
    if group:
        group_data = data[data['hoitoketju'] == group]
        return {
            "Mean": group_data[column].mean(),
            "SD": group_data[column].std(),
            "N": group_data.shape[0]
        }
    else:
        return {
            "Mean": data[column].mean(),
            "SD": data[column].std(),
            "N": data.shape[0]
        }

# Initialize dictionary for storing results
table_stats = {
    "All patients": {},
    "Physical": {},
    "Digital": {}
}

# Define age bins
age_bins = [0, 20, 40, 60, 80, 100]
age_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']

# Add age group
matched_data['age_group'] = pd.cut(matched_data['age'], bins=age_bins, labels=age_labels, right=True)

# Calculate stats for Age groups
for group in age_labels:
    table_stats["All patients"][group] = calculate_statistics(
        matched_data[matched_data['age_group'] == group], 'age'
    )
    table_stats["Physical"][group] = calculate_statistics(
        matched_data[matched_data['age_group'] == group], 'age', group='physical'
    )
    table_stats["Digital"][group] = calculate_statistics(
        matched_data[matched_data['age_group'] == group], 'age', group='digi'
    )

# Calculate stats for gender
for gender in [0, 1]:  # 0 = Male, 1 = Female
    table_stats["All patients"][gender] = calculate_statistics(
        matched_data[matched_data['sex'] == gender], 'age'
    )
    table_stats["Physical"][gender] = calculate_statistics(
        matched_data[(matched_data['sex'] == gender) & (matched_data['hoitoketju'] == 'physical')], 'age'
    )
    table_stats["Digital"][gender] = calculate_statistics(
        matched_data[(matched_data['sex'] == gender) & (matched_data['hoitoketju'] == 'digi')], 'age'
    )

# Create DataFrame to display results
table_df = pd.DataFrame.from_dict(table_stats, orient='index')
table_df = table_df.applymap(lambda x: f"{x['Mean']:.2f} ± {x['SD']:.2f}" if isinstance(x, dict) else x)

# Display the final table
table_df





# Calculate statistics for matched data
def calculate_statistics(data, group_column, treatment_column):
    # Initialize a results dictionary to store statistics
    results = []

    # Process each diagnosis group
    for group, group_data in data.groupby(group_column):
        # Process each treatment group (digi and physical)
        for treatment, treatment_data in group_data.groupby(treatment_column):
            # Calculate statistics
            mean_age = treatment_data['age'].mean()
            mean_n_contacts = treatment_data['n_contacts'].mean()
            mean_n_labs = treatment_data['n_labs'].mean()
            mean_n_rtg = treatment_data['n_rtg'].mean()*100 # 
            mean_contacts_costs = treatment_data['contacts_cost'].mean()
            mean_labs_costs = treatment_data['labs_cost'].mean()
            mean_rtg_costs = treatment_data['rtg_cost'].mean()
            mean_kustannus = treatment_data['kustannus'].mean()
            percent_female = (treatment_data['sex'] == 1).mean() * 100  # Use 1 for "Nainen"
            percent_male = (treatment_data['sex'] == 0).mean() * 100  # Use 0 for "Mies"
            count_patients = len(treatment_data)

            # Append the results
            results.append({
                'diagnosis_group': group,
                'treatment': treatment,
                'mean_age': mean_age,
                'mean_n_contacts': mean_n_contacts,
                'mean_n_labs': mean_n_labs,
                'mean_n_rtg_per_100': mean_n_rtg,
                'mean_contacts_costs': mean_contacts_costs,
                'mean_labs_costs' : mean_labs_costs,
                'mean_rtg_costs' : mean_rtg_costs,
                'mean_kustannus': mean_kustannus,
                'percent_female': percent_female,
                'percent_male': percent_male,
                'count_patients': count_patients
            })

    # Convert results to a DataFrame
    stats_df = pd.DataFrame(results)
    return stats_df

# Run the function on the matched dataset
stats_summary = calculate_statistics(
    data=matched_data,
    group_column='diagnosis_group',
    treatment_column='hoitoketju'
)

stats_summary.round(2)


# ## Statistical significance

from scipy.stats import ttest_ind

# Grouping data into digital and physical groups
physical_data = matched_data[matched_data['hoitoketju'] == 'physical']
digital_data = matched_data[matched_data['hoitoketju'] == 'digi']

# Function to calculate mean costs and p-value
def calculate_cost_stats(physical_cost, digital_cost):
    # Calculate mean for both groups
    physical_mean = physical_cost.mean()
    digital_mean = digital_cost.mean()
    
    # Perform independent t-test to get p-value
    _, p_value = ttest_ind(physical_cost, digital_cost, equal_var=False)
    
    return physical_mean, digital_mean, p_value

# Initialize an empty dictionary to store the results
table_stats = {}

# Visit Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['contacts_cost'], digital_data['contacts_cost']
)
table_stats['Visit costs'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}

# Laboratory Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['labs_cost'], digital_data['labs_cost']
)
table_stats['Laboratory costs'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}

# Imaging Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['rtg_cost'], digital_data['rtg_cost']
)
table_stats['Imaging costs'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}


# Total Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['kustannus'], digital_data['kustannus']
)
table_stats['Total'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}

# Create a DataFrame to display the results
cost_table_df = pd.DataFrame.from_dict(table_stats, orient='index')

# Formatting the results to two decimal points for readability
cost_table_df = cost_table_df.applymap(lambda x: f"{x:.10e}" if isinstance(x, float) else x)

# Display the resulting table
print(cost_table_df)




import matplotlib.pyplot as plt

# Extracting relevant data from cost_table_df
categories = cost_table_df.index[:-1]  # Exclude 'Total' row
digital_costs = cost_table_df.loc[categories, 'Digital (€)'].astype(float).values
physical_costs = cost_table_df.loc[categories, 'Physical (€)'].astype(float).values

# Colors for the stacked components
digital_colors = ['#394984', '#5968b3', '#7a87cc']  # Different tones of blue for digital pathway
physical_colors = ['#05843d', '#36a165', '#68bf8d']  # Different tones of green for physical pathway

# Plotting stacked bar chart
plt.figure(figsize=(8, 8))  # Reduce the figure width to further reduce whitespace
x = [0, 0.5]  # Define bar positions closer to each other

# Plotting digital stacked bars
bottom_digital = 0
for i, cost in enumerate(digital_costs):
    plt.bar(x[0], cost, width=0.3, color=digital_colors[i], bottom=bottom_digital, label=f'Digital' if i == 0 else "")
    bottom_digital += cost

# Plotting physical stacked bars
bottom_physical = 0
for i, cost in enumerate(physical_costs):
    plt.bar(x[1], cost, width=0.3, color=physical_colors[i], bottom=bottom_physical, label=f'Physical' if i == 0 else "")
    bottom_physical += cost

plt.xticks(ticks=x, labels=['Digital Pathway', 'Physical Pathway'])
plt.ylabel('Mean Cost (€)')
plt.title('Comparison of Cost Components per Episode by Pathway')
plt.legend()
plt.show()




from scipy.stats import ttest_ind

def perform_ttest_by_group(matched_data, outcome_column, group_column, treatment_column):
    """
    Perform independent two-sample t-test for each group in the matched data.
    
    Parameters:
    -----------
    matched_data : pandas.DataFrame
        DataFrame containing matched data with group, treatment, and outcome columns.
    outcome_column : str
        Name of the outcome column to test (e.g., 'kustannus').
    group_column : str
        Name of the column indicating groups (e.g., 'diagnoosiryhmä').
    treatment_column : str
        Name of the treatment column (e.g., 'hoitoketju').
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with t-test results for each group.
    """
    results = []

    # Group by the specified group column (e.g., 'diagnoosiryhmä')
    for group, group_data in matched_data.groupby(group_column):
        print(f"Processing group: {group}")
        
        # Separate treatment and control groups
        treatment_group = group_data[group_data[treatment_column] == 'digi'][outcome_column]
        control_group = group_data[group_data[treatment_column] == 'physical'][outcome_column]

        # Check if there are enough samples in both groups
        if len(treatment_group) < 2 or len(control_group) < 2:
            print(f"Skipping group {group} due to insufficient data.")
            continue

        # Perform independent two-sample t-test
        t_stat, p_value = ttest_ind(
            treatment_group,
            control_group,
            equal_var=False  # Use Welch's t-test to account for unequal variances
        )

        # Store results
        results.append({
            group_column: group,
            'mean_digital (euro)': treatment_group.mean(),
            'mean_physical (euro)': control_group.mean(),
            't_stat': t_stat,
            'p_value': p_value,
            'n_digital': len(treatment_group),
            'n_physical': len(control_group)
        })

    # Convert results to a DataFrame for easy interpretation
    return pd.DataFrame(results)

# Example Usage
# Perform t-tests for each diagnoosiryhmä
outcome_column = 'kustannus'
group_column = 'diagnosis_group'
treatment_column = 'hoitoketju'

ttest_results = perform_ttest_by_group(
    matched_data=matched_data,
    outcome_column=outcome_column,
    group_column=group_column,
    treatment_column=treatment_column
)

# Display results
ttest_results



# Enhance results with savings calculations
def calculate_savings(ttest_results):
    ttest_results['percentage_saved'] = (
        (ttest_results['mean_physical (euro)'] - ttest_results['mean_digital (euro)']) /
        ttest_results['mean_physical (euro)'] * 100
    ).round(2)

    ttest_results['total_savings'] = (
        (ttest_results['mean_physical (euro)'] - ttest_results['mean_digital (euro)']) *
        ttest_results['n_digital']
    ).round(2)

    # Calculate overall savings and append to the DataFrame
    overall_mean_physical = (
        ttest_results['mean_physical (euro)'] * ttest_results['n_physical']
    ).sum() / ttest_results['n_physical'].sum()

    overall_mean_digital = (
        ttest_results['mean_digital (euro)'] * ttest_results['n_digital']
    ).sum() / ttest_results['n_digital'].sum()

    overall_savings = overall_mean_physical - overall_mean_digital
    overall_percentage_saved = (overall_savings / overall_mean_physical * 100).round(2)
    overall_total_savings = (
        overall_savings * ttest_results['n_digital'].sum()
    ).round(2)

    # Append overall results as a new row
    overall_summary = pd.DataFrame({
        'diagnosis_group': ['Overall'],
        'mean_digital (euro)': [overall_mean_digital],
        'mean_physical (euro)': [overall_mean_physical],
        'percentage_saved': [overall_percentage_saved],
        'total_savings': [overall_total_savings],
        'n_digital': [ttest_results['n_digital'].sum()],
        'n_physical': [ttest_results['n_physical'].sum()],
        't_stat': [None],
        'p_value': [None]
    })

    ttest_results = pd.concat([ttest_results, overall_summary], ignore_index=True)

    return ttest_results


# Apply savings calculation
business_summary = calculate_savings(ttest_results)

# Display business summary
business_summary



from scipy.stats import ttest_ind
import pandas as pd

def calculate_statistics_and_ttests(data, group_column, treatment_column, cost_columns):
    """
    Calculates statistics and performs t-tests for matched data.
    
    Parameters:
    - data: DataFrame containing matched data.
    - group_column: The column name for diagnosis groups.
    - treatment_column: The column name indicating treatment types ('digi' or 'physical').
    - cost_columns: List of column names for cost-related data.
    
    Returns:
    - stats_df: DataFrame with aggregated statistics.
    - ttest_results: DataFrame with t-test results.
    """
    # Initialize results dictionary for statistics
    stats_results = []
    ttest_results = []

    # Process each diagnosis group
    for group, group_data in data.groupby(group_column):
        # Process each treatment group (digital and physical)
        for treatment, treatment_data in group_data.groupby(treatment_column):
            # Calculate statistics
            mean_age = treatment_data['age'].mean()
            mean_n_contacts = treatment_data['n_contacts'].mean()
            mean_n_labs = treatment_data['n_labs'].mean()
            mean_n_rtg = treatment_data['n_rtg'].mean() * 100
            mean_contacts_costs = treatment_data['contacts_cost'].mean()
            mean_labs_costs = treatment_data['labs_cost'].mean()
            mean_rtg_costs = treatment_data['rtg_cost'].mean()
            mean_kustannus = treatment_data['kustannus'].mean()
            percent_female = (treatment_data['sex'] == 1).mean() * 100
            percent_male = (treatment_data['sex'] == 0).mean() * 100
            count_patients = len(treatment_data)

            # Append statistics to results
            stats_results.append({
                'diagnosis_group': group,
                'treatment': treatment,
                'mean_age': mean_age,
                'mean_n_contacts': mean_n_contacts,
                'mean_n_labs': mean_n_labs,
                'mean_n_rtg_per_100': mean_n_rtg,
                'mean_contacts_costs': mean_contacts_costs,
                'mean_labs_costs': mean_labs_costs,
                'mean_rtg_costs': mean_rtg_costs,
                'mean_kustannus': mean_kustannus,
                'percent_female': percent_female,
                'percent_male': percent_male,
                'count_patients': count_patients
            })

        # Perform t-tests for the specified cost columns
        digital_data = group_data[group_data[treatment_column] == 'digi']
        traditional_data = group_data[group_data[treatment_column] == 'physical']

        for cost_column in cost_columns:
            if cost_column in data.columns:
                digital_values = digital_data[cost_column]
                traditional_values = traditional_data[cost_column]
                
                # Perform independent t-test
                t_stat, p_value = ttest_ind(digital_values, traditional_values, equal_var=False)
                
                # Append t-test results
                ttest_results.append({
                    'diagnosis_group': group,
                    'cost_category': cost_column,
                    'mean_digital': digital_values.mean(),
                    'mean_traditional': traditional_values.mean(),
                    't_stat': t_stat,
                    'p_value': p_value
                })

    # Convert results to DataFrames
    stats_df = pd.DataFrame(stats_results)
    ttest_results_df = pd.DataFrame(ttest_results)

    return stats_df, ttest_results_df

# Specify cost columns for t-tests
cost_columns = ['contacts_cost', 'labs_cost', 'rtg_cost', 'kustannus']

# Run the integrated function on matched data
stats_summary, ttest_summary = calculate_statistics_and_ttests(
    data=matched_data,
    group_column='diagnosis_group',
    treatment_column='hoitoketju',
    cost_columns=cost_columns
)


ttest_summary.round(4)


import matplotlib.pyplot as plt

# Updating the 'Diagnosis Group' column to have proper capitalization and removing the 'Overall' row
business_summary['diagnosis_group'] = business_summary['diagnosis_group'].str.capitalize()
business_summary = business_summary[business_summary['diagnosis_group'] != 'Overall']

# Color scheme with the updated requested colors
digital_color_updated = '#394984'  # Dark blue
physical_color_updated = '#05843d'  # Green

# Creating side-by-side bar chart with the updated color scheme
plt.figure(figsize=(12, 8))
x = range(len(business_summary['diagnosis_group']))

plt.bar([i - 0.2 for i in x], business_summary['mean_digital (euro)'], width=0.4, color=digital_color_updated, label='Digital Pathway Mean (€)')
plt.bar([i + 0.2 for i in x], business_summary['mean_physical (euro)'], width=0.4, color=physical_color_updated, label='Physical Pathway Mean (€)')

plt.xticks(ticks=x, labels=business_summary['diagnosis_group'])
plt.xlabel('Diagnosis Group')
plt.ylabel('Mean Cost per Episode (€)')
plt.title('Comparison of Digital vs Physical Pathway Costs per Diagnosis Group')
plt.legend()
plt.show()


# ## Sensitivity analysis

# ### Same person sensitivity analysis


# BASED ON EACH DIAGNOSIS GROUP, same person comparison
from scipy.stats import ttest_rel

# Initialize a list to store the results for each diagnoosiryhmä
results = []

# Get the list of unique diagnoosiryhmä values
diag_groups = df_episodi['diagnosis_group'].unique()

# Loop over each diagnoosiryhmä
for diag in diag_groups:
    # Subset df_episodi for the current diagnoosiryhmä
    df_diag = df_episodi[df_episodi['diagnosis_group'] == diag]
    
    # Identify IDs with both digi and physical episodes in this diagnoosiryhmä
    id_counts = df_diag.groupby('id')['hoitoketju'].nunique()
    ids_with_both = id_counts[id_counts == 2].index  # IDs with both digi and physical episodes
    
    # Filter episodes for these IDs
    df_diag_same_person = df_diag[df_diag['id'].isin(ids_with_both)]
    
    # Separate digi and physical episodes for comparison
    digi_episodes = df_diag_same_person[df_diag_same_person['hoitoketju'] == 'digi']
    physical_episodes = df_diag_same_person[df_diag_same_person['hoitoketju'] == 'physical']
    
    # Sum kustannus (cost) per id for digi and physical episodes
    digi_sums = digi_episodes.groupby('id')['kustannus'].sum().reset_index(name='kustannus_digi')
    physical_sums = physical_episodes.groupby('id')['kustannus'].sum().reset_index(name='kustannus_physical')
    
    # Merge digi and physical sums on 'id'
    comparison = pd.merge(digi_sums, physical_sums, on='id')
    
    # Calculate cost differences
    comparison['cost_diff'] = comparison['kustannus_digi'] - comparison['kustannus_physical']
    
    # Perform a paired t-test on the cost differences
    t_stat, p_value = ttest_rel(comparison['kustannus_digi'], comparison['kustannus_physical'])
    
    # Store the results
    results.append({
        'diagnoosiryhmä': diag,
        't_stat': t_stat,
        'p_value': p_value,
        'mean_cost_digi': comparison['kustannus_digi'].mean(),
        'mean_cost_physical': comparison['kustannus_physical'].mean(),
        'num_pairs': len(comparison)
    })
    
    # Print the results for the current diagnoosiryhmä
    print(f"Diagnoosiryhmä: {diag}")
    print(f"Number of matched pairs: {len(comparison)}")
    print(f"T-Statistic: {t_stat:.3f}, P-Value: {p_value:.3f}")
    print(f"Mean Cost Digi: €{comparison['kustannus_digi'].mean():.2f}")
    print(f"Mean Cost Physical: €{comparison['kustannus_physical'].mean():.2f}")
    print()

# Convert the results list to a DataFrame for easy viewing
results_df = pd.DataFrame(results)
results_df



# ### Strata analysis

matched_data


from sklearn.linear_model import LogisticRegression

# Calculate propensity scores per diagnosis group
def calculate_propensity_scores_per_group(data, group_column, treatment_column, covariate_columns):
    """
    Calculate propensity scores for each diagnosis group separately.
    """
    data['propensity_scores'] = np.nan  # Initialize column for propensity scores

    for group in data[group_column].unique():
        group_data = data[data[group_column] == group].copy()

        # Explicitly encode treatment variable
        group_data['treatment'] = group_data[treatment_column].map({'digi': 1, 'physical': 0})

        # Fit logistic regression for propensity scores
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(group_data[covariate_columns], group_data['treatment'])

        # Predict propensity scores for this group
        data.loc[group_data.index, 'propensity_scores'] = log_reg.predict_proba(group_data[covariate_columns])[:, 1]

    return data

# Create strata per diagnosis group
def create_strata_per_group(data, group_column, propensity_score_column, n_strata=5):
    """
    Create strata for each diagnosis group separately based on propensity scores.
    """
    data['strata'] = np.nan  # Initialize column for strata

    for group in data[group_column].unique():
        group_data = data[data[group_column] == group].copy()

        # Create strata based on propensity scores
        data.loc[group_data.index, 'strata'] = pd.qcut(
            group_data[propensity_score_column], q=n_strata, labels=False
        )

    return data

# Check balance within strata for each diagnosis group
def check_balance_per_group_global_sd(data, group_column, strata_column, treatment_column, propensity_score_column):
    """
    Check balance for each diagnosis group and stratum using global SD.
    """
    balance_info = []

    for group in data[group_column].unique():
        group_data = data[data[group_column] == group]

        # Calculate the global SD of propensity scores for the entire group
        global_sd = group_data[propensity_score_column].std()

        for stratum in group_data[strata_column].dropna().unique():
            stratum_data = group_data[group_data[strata_column] == stratum]

            # Calculate mean propensity scores for treated and control groups
            treated_mean = stratum_data.loc[
                stratum_data[treatment_column] == 'digi', propensity_score_column
            ].mean()
            control_mean = stratum_data.loc[
                stratum_data[treatment_column] == 'physical', propensity_score_column
            ].mean()

            # Calculate Standardized Mean Difference (SMD) using global SD
            smd = (treated_mean - control_mean) / global_sd if global_sd > 0 else 0

            # Store results
            balance_info.append({
                "Diagnosis Group": group,
                "Stratum": stratum,
                "Treated Mean": treated_mean,
                "Control Mean": control_mean,
                "Global SD": global_sd,
                "SMD": smd
            })

    balance_info_df = pd.DataFrame(balance_info)
    return balance_info_df

covariate_columns = ['age', 'cci', 'n_laakari']  

# Calculate propensity scores
matched_data_strata = calculate_propensity_scores_per_group(
    data=matched_data,
    group_column='diagnosis_group',
    treatment_column='hoitoketju',
    covariate_columns=covariate_columns
)

# Create strata
matched_data_strata = create_strata_per_group(
    data=matched_data_strata,
    group_column='diagnosis_group',
    propensity_score_column='propensity_scores',
    n_strata=5
)

# Check balance within strata using global SD
balance_info_global_sd = check_balance_per_group_global_sd(
    data=matched_data_strata,
    group_column='diagnosis_group',
    strata_column='strata',
    treatment_column='hoitoketju',
    propensity_score_column='propensity_scores'
)

# Sort and display balance metrics
balance_info_global_sd = balance_info_global_sd.sort_values(by=["Diagnosis Group", "Stratum"]).reset_index(drop=True)
print(balance_info_global_sd)




# Identify problematic strata only for the respiratory group
problematic_strata_resp = balance_info_global_sd[
    (balance_info_global_sd["SMD"] > 0.1) & 
    (balance_info_global_sd["Diagnosis Group"] == "respiratory")
][["Diagnosis Group", "Stratum"]]

# Filter matched_data_strata to exclude problematic strata for respiratory group only
filtered_data = matched_data_strata[~(
    (matched_data_strata["diagnosis_group"] == "respiratory") &
    (matched_data_strata["strata"].isin(problematic_strata_resp["Stratum"]))
)]


# Validate the filtered dataset
print("Original data size:", matched_data_strata.shape[0])
print("Filtered data size:", filtered_data.shape[0])
print("Excluded rows:", matched_data_strata.shape[0] - filtered_data.shape[0])




# Perform t-tests for each diagnoosiryhmä
outcome_column = 'kustannus'
group_column = 'diagnosis_group'
treatment_column = 'hoitoketju'

ttest_results_strata = perform_ttest_by_group(
    matched_data=filtered_data,
    outcome_column=outcome_column,
    group_column=group_column,
    treatment_column=treatment_column
)

# Display results
ttest_results_strata


# ### Time window sensitivity analysis

# #### 7 day time window

# Assuming df_contacts_diagn is already defined
df = pd.DataFrame(df_contacts_diagn)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["diagnosis_group"] = df["diagnosis_group"].replace("nan", np.nan)

# Sort by id and datetime
df = df.sort_values(by=["id", "date"])

# Initialize the episode_number
df["episode_number"] = 0


# Function to assign episode numbers
def assign_episode_numbers(group):
    episode_number = 1
    episode_start_date = group["date"].iloc[0]
    group["episode_number"].iloc[0] = episode_number

    for i in range(1, len(group)):
        current_date = group["date"].iloc[i]
        previous_date = group["date"].iloc[i - 1]

        if (current_date - episode_start_date).days > 7 and (
            current_date - previous_date
        ).days > 7:
            episode_number += 1
            episode_start_date = current_date

        group["episode_number"].iloc[i] = episode_number

    return group


# Apply the function to each group
df = df.groupby("id", group_keys=False).apply(assign_episode_numbers)


# Function to propagate the first non-NaN diagnosis_group value
def propagate_diagnosis_group(group):
    for episode in group["episode_number"].unique():
        episode_group = group[group["episode_number"] == episode]
        # Get the first non-NaN diagnosis_group value
        non_nan_diagnosis = episode_group["diagnosis_group"].dropna()
        if not non_nan_diagnosis.empty:
            first_non_nan_diagnosis = non_nan_diagnosis.iloc[0]
            group.loc[episode_group.index, "diagnosis_group_new"] = (
                first_non_nan_diagnosis
            )
    return group


# Apply the function to propagate diagnosis_group
df = df.groupby("id", group_keys=False).apply(propagate_diagnosis_group)

# Display the DataFrame with the new column
print(df.head())

# only keep first row of each episode
df_episodi_7 = df.drop_duplicates(subset=["id", "episode_number"], keep="first")
df_episodi_7


# Drop those first two weeks
df_episodi_7 = df_episodi_7[df_episodi_7['date'] >= '2023-05-15']

# Create new columns for 'alku_pvm' and 'loppu_pvm'
df_episodi_7['alku_pvm'] = df_episodi_7['date']
df_episodi_7['loppu_pvm'] = df_episodi_7['date'] + pd.Timedelta(days=7)

df_episodi_7['episode_id'] = range(1, len(df_episodi_7) + 1) # New column with episode_id
df_episodi_7.reset_index(drop=True, inplace=True)

df_episodi_7



df_episodi_7 = df_episodi_7[
    (df_episodi_7['diagnosis_group_new'].notna())
].copy()

# Drop the 'ensi_kontakti' column (should be True for all rows)
df_episodi_7 = df_episodi_7.drop(columns=['diagnosis_group'])

# Rename the column 'diagnosis_group_new' to 'diagnosis_group'
df_episodi_7.rename(columns={"diagnosis_group_new": "diagnosis_group"}, inplace=True)


df_episodi_7




df_contacts_diagn['date'] = pd.to_datetime(df_contacts_diagn['date'], utc=False)  # Ensure timezone-naive
df_episodi_7['date'] = pd.to_datetime(df_episodi_7['date'], utc=False)  # Ensure timezone-naive

df_contacts_diagn['date'] = pd.to_datetime(df_contacts_diagn['date']) # Date as datetime, varmuuden vuoksi

# Ensure `price` column is properly aligned in df_lab and df_rtg
df_lab['price'] = df_lab['price'].fillna(0)
df_rtg['price'] = df_rtg['price'].fillna(0)

# Contacts
df_contacts_diagn_merged = df_contacts_diagn.merge(
    df_episodi_7[['id', 'episode_id', 'alku_pvm', 'loppu_pvm']], 
    on='id', 
    how='inner'
)
df_contacts_diagn_filtered = df_contacts_diagn_merged[
    (df_contacts_diagn_merged['date'] >= df_contacts_diagn_merged['alku_pvm']) &
    (df_contacts_diagn_merged['date'] <= df_contacts_diagn_merged['loppu_pvm'])
]


# Repeat the process for labs
df_lab_merged = df_lab.merge(
    df_episodi_7[['id', 'episode_id', 'alku_pvm', 'loppu_pvm']], 
    on='id', 
    how='inner'
)
df_lab_filtered = df_lab_merged[
    (df_lab_merged['date'] >= df_lab_merged['alku_pvm']) &
    (df_lab_merged['date'] <= df_lab_merged['loppu_pvm'])
]

# Repeat the process for rtg
df_rtg_merged = df_rtg.merge(
    df_episodi_7[['id', 'episode_id', 'alku_pvm', 'loppu_pvm']], 
    on='id', 
    how='inner'
)
df_rtg_filtered = df_rtg_merged[
    (df_rtg_merged['date'] >= df_rtg_merged['alku_pvm']) &
    (df_rtg_merged['date'] <= df_rtg_merged['loppu_pvm'])
]

# Aggregate costs and counts by episode_id
contacts_costs = df_contacts_diagn_filtered.groupby('episode_id')['price'].sum().rename('contacts_cost')
labs_costs = df_lab_filtered.groupby('episode_id')['price'].sum().rename('labs_cost')
rtg_costs = df_rtg_filtered.groupby('episode_id')['price'].sum().rename('rtg_cost')

# Count number of contacts, labs, and rtg per episode_id
n_contacts = df_contacts_diagn_filtered.groupby('episode_id').size().rename('n_contacts')
n_labs = df_lab_filtered.groupby('episode_id').size().rename('n_labs')
n_rtg = df_rtg_filtered.groupby('episode_id').size().rename('n_rtg')

# Extract first contact details
first_contact = df_contacts_diagn_filtered.sort_values('date').groupby('episode_id')[['profession', 'contact_mode']].first().reset_index()
first_contact['ensi_kontakti'] = first_contact['profession'] + " " + first_contact['contact_mode']

# Merge aggregated data into df_episodi
df_episodi_7 = df_episodi_7.merge(contacts_costs, on='episode_id', how='left')
df_episodi_7 = df_episodi_7.merge(labs_costs, on='episode_id', how='left')
df_episodi_7 = df_episodi_7.merge(rtg_costs, on='episode_id', how='left')
df_episodi_7 = df_episodi_7.merge(n_contacts, on='episode_id', how='left')
df_episodi_7 = df_episodi_7.merge(n_labs, on='episode_id', how='left')
df_episodi_7 = df_episodi_7.merge(n_rtg, on='episode_id', how='left')
df_episodi_7 = df_episodi_7.merge(first_contact[['episode_id', 'ensi_kontakti']], on='episode_id', how='left')

# Fill missing values in cost and count columns
df_episodi_7['contacts_cost'] = df_episodi_7['contacts_cost'].fillna(0)
df_episodi_7['labs_cost'] = df_episodi_7['labs_cost'].fillna(0)
df_episodi_7['rtg_cost'] = df_episodi_7['rtg_cost'].fillna(0)
df_episodi_7['n_contacts'] = df_episodi_7['n_contacts'].fillna(0).astype(int)
df_episodi_7['n_labs'] = df_episodi_7['n_labs'].fillna(0).astype(int)
df_episodi_7['n_rtg'] = df_episodi_7['n_rtg'].fillna(0).astype(int)

# Calculate total costs
df_episodi_7['kustannus'] = df_episodi_7['contacts_cost'] + df_episodi_7['labs_cost'] + df_episodi_7['rtg_cost']

# Display the updated dataframe
df_episodi_7



# Merge df_covariates into df_episodi based on the id
df_episodi_7 = df_episodi_7.merge(
    df_covariates,
    on='id',  # Match based on the 'id' column
    how='left'  # Use left join to keep all episodes in df_episodi
)

df_episodi_7 = df_episodi_7[
    ~((df_episodi_7['diagnosis_group'] == 'urinary') & 
      ((df_episodi_7['age'] > 65)))
]

df_episodi_7= df_episodi_7.drop_duplicates()
# Reset the episode_id to start from 1
df_episodi_7['episode_id'] = range(1, len(df_episodi_7) + 1)

df_episodi_7



# Create a new column 'hoitoketju' based on the condition in 'ensi_kontakti'
df_episodi_7.rename(columns={'digi': 'hoitoketju'}, inplace=True)

# Update the values in the 'hoitoketju' column
df_episodi_7['hoitoketju'] = df_episodi_7['hoitoketju'].apply(lambda x: 'digi' if x == 'DIGI' else 'physical' if x == 'MUU' else x)

# Filter out rows where diagnoosiryhmä is 'urinary' and age > 65
df_episodi_7 = df_episodi_7[~((df_episodi_7['diagnosis_group'] == 'urinary') & (df_episodi_7['age'] > 65))]

df_episodi_7



df_episodi_7['sex'] = df_episodi_7['sex'].map({'Mies': 0, 'Nainen': 1})

# Perform PSM for each diagnoosiryhmä
matched_data_7 = perform_psm_with_polynomial_features(
    data=df_episodi_7,  
    group_column='diagnosis_group',
    treatment_column='hoitoketju',  # Treatment indicator column
    covariate_columns=covariate_columns,
    caliper=0.2  # Adjust caliper as necessary, caliper sets the maximum allowable distance (or difference) between the propensity scores of a treated unit and a control unit to be matched
)

outcome_column = 'kustannus'
group_column = 'diagnosis_group'
treatment_column = 'hoitoketju'

ttest_results = perform_ttest_by_group(
    matched_data=matched_data_7,
    outcome_column=outcome_column,
    group_column=group_column,
    treatment_column=treatment_column
)

# Display results
ttest_results


# Enhance results with savings calculations
def calculate_savings(ttest_results):
    ttest_results['percentage_saved'] = (
        (ttest_results['mean_physical (euro)'] - ttest_results['mean_digital (euro)']) /
        ttest_results['mean_physical (euro)'] * 100
    ).round(2)

    ttest_results['total_savings'] = (
        (ttest_results['mean_physical (euro)'] - ttest_results['mean_digital (euro)']) *
        ttest_results['n_digital']
    ).round(2)

    # Calculate overall savings and append to the DataFrame
    overall_mean_physical = (
        ttest_results['mean_physical (euro)'] * ttest_results['n_physical']
    ).sum() / ttest_results['n_physical'].sum()

    overall_mean_digital = (
        ttest_results['mean_digital (euro)'] * ttest_results['n_digital']
    ).sum() / ttest_results['n_digital'].sum()

    overall_savings = overall_mean_physical - overall_mean_digital
    overall_percentage_saved = (overall_savings / overall_mean_physical * 100).round(2)
    overall_total_savings = (
        overall_savings * ttest_results['n_digital'].sum()
    ).round(2)

    # Append overall results as a new row
    overall_summary = pd.DataFrame({
        'diagnosis_group': ['Overall'],
        'mean_digital (euro)': [overall_mean_digital],
        'mean_physical (euro)': [overall_mean_physical],
        'percentage_saved': [overall_percentage_saved],
        'total_savings': [overall_total_savings],
        'n_digital': [ttest_results['n_digital'].sum()],
        'n_physical': [ttest_results['n_physical'].sum()],
        't_stat': [None],
        'p_value': [None]
    })

    ttest_results = pd.concat([ttest_results, overall_summary], ignore_index=True)

    return ttest_results


# Apply savings calculation
business_summary = calculate_savings(ttest_results)

# Display business summary
business_summary



# Grouping data into digital and physical groups
physical_data = matched_data_7[matched_data_7['hoitoketju'] == 'physical']
digital_data = matched_data_7[matched_data_7['hoitoketju'] == 'digi']

# Function to calculate mean costs and p-value
def calculate_cost_stats(physical_cost, digital_cost):
    # Calculate mean for both groups
    physical_mean = physical_cost.mean()
    digital_mean = digital_cost.mean()
    
    # Perform independent t-test to get p-value
    _, p_value = ttest_ind(physical_cost, digital_cost, equal_var=False)
    
    return physical_mean, digital_mean, p_value

# Initialize an empty dictionary to store the results
table_stats = {}

# Visit Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['contacts_cost'], digital_data['contacts_cost']
)
table_stats['Visit costs'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}

# Laboratory Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['labs_cost'], digital_data['labs_cost']
)
table_stats['Laboratory costs'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}

# Imaging Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['rtg_cost'], digital_data['rtg_cost']
)
table_stats['Imaging costs'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}


# Total Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['kustannus'], digital_data['kustannus']
)
table_stats['Total'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}

# Create a DataFrame to display the results
cost_table_df = pd.DataFrame.from_dict(table_stats, orient='index')

# Formatting the results to two decimal points for readability
cost_table_df = cost_table_df.applymap(lambda x: f"{x:.10e}" if isinstance(x, float) else x)

# Display the resulting table
cost_table_df.round(2)


# #### 30 day timewindow

df = pd.DataFrame(df_contacts_diagn)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["diagnosis_group"] = df["diagnosis_group"].replace("nan", np.nan)

# Sort by id and datetime
df = df.sort_values(by=["id", "date"])

# Initialize the episode_number
df["episode_number"] = 0


# Function to assign episode numbers
def assign_episode_numbers(group):
    episode_number = 1
    episode_start_date = group["date"].iloc[0]
    group["episode_number"].iloc[0] = episode_number

    for i in range(1, len(group)):
        current_date = group["date"].iloc[i]
        previous_date = group["date"].iloc[i - 1]

        if (current_date - episode_start_date).days > 30 and (
            current_date - previous_date
        ).days > 30:
            episode_number += 1
            episode_start_date = current_date

        group["episode_number"].iloc[i] = episode_number

    return group


# Apply the function to each group
df = df.groupby("id", group_keys=False).apply(assign_episode_numbers)


# Function to propagate the first non-NaN diagnosis_group value
def propagate_diagnosis_group(group):
    for episode in group["episode_number"].unique():
        episode_group = group[group["episode_number"] == episode]
        # Get the first non-NaN diagnosis_group value
        non_nan_diagnosis = episode_group["diagnosis_group"].dropna()
        if not non_nan_diagnosis.empty:
            first_non_nan_diagnosis = non_nan_diagnosis.iloc[0]
            group.loc[episode_group.index, "diagnosis_group_new"] = (
                first_non_nan_diagnosis
            )
    return group


# Apply the function to propagate diagnosis_group
df = df.groupby("id", group_keys=False).apply(propagate_diagnosis_group)

# Display the DataFrame with the new column
print(df.head())

# only keep first row of each episode
df_episodi_30 = df.drop_duplicates(subset=["id", "episode_number"], keep="first")



# Drop those first two weeks
df_episodi_30 = df_episodi_30[df_episodi_30['date'] >= '2023-05-15']

# Create new columns for 'alku_pvm' and 'loppu_pvm'
df_episodi_30['alku_pvm'] = df_episodi_30['date']
df_episodi_30['loppu_pvm'] = df_episodi_30['date'] + pd.Timedelta(days=30)

df_episodi_30['episode_id'] = range(1, len(df_episodi_30) + 1) # New column with episode_id
df_episodi_30.reset_index(drop=True, inplace=True)

df_episodi_30 = df_episodi_30[
    (df_episodi_30['diagnosis_group_new'].notna())
].copy()

# Drop the 'ensi_kontakti' column (should be True for all rows)
df_episodi_30 = df_episodi_30.drop(columns=['diagnosis_group'])

# Rename the column 'diagnosis_group_new' to 'diagnosis_group'
df_episodi_30.rename(columns={"diagnosis_group_new": "diagnosis_group"}, inplace=True)

df_episodi_30



df_episodi_30['date'] = pd.to_datetime(df_episodi_30['date'], utc=False)  # Ensure timezone-naive

# Ensure `price` column is properly aligned in df_lab and df_rtg
df_lab['price'] = df_lab['price'].fillna(0)
df_rtg['price'] = df_rtg['price'].fillna(0)

# Ensure `price` column is properly aligned in df_lab and df_rtg
df_lab['price'] = df_lab['price'].fillna(0)
df_rtg['price'] = df_rtg['price'].fillna(0)

# Contacts
df_contacts_diagn_merged = df_contacts_diagn.merge(
    df_episodi_30[['id', 'episode_id', 'alku_pvm', 'loppu_pvm']], 
    on='id', 
    how='inner'
)
df_contacts_diagn_filtered = df_contacts_diagn_merged[
    (df_contacts_diagn_merged['date'] >= df_contacts_diagn_merged['alku_pvm']) &
    (df_contacts_diagn_merged['date'] <= df_contacts_diagn_merged['loppu_pvm'])
]


# Repeat the process for labs
df_lab_merged = df_lab.merge(
    df_episodi_30[['id', 'episode_id', 'alku_pvm', 'loppu_pvm']], 
    on='id', 
    how='inner'
)
df_lab_filtered = df_lab_merged[
    (df_lab_merged['date'] >= df_lab_merged['alku_pvm']) &
    (df_lab_merged['date'] <= df_lab_merged['loppu_pvm'])
]

# Repeat the process for rtg
df_rtg_merged = df_rtg.merge(
    df_episodi_30[['id', 'episode_id', 'alku_pvm', 'loppu_pvm']], 
    on='id', 
    how='inner'
)
df_rtg_filtered = df_rtg_merged[
    (df_rtg_merged['date'] >= df_rtg_merged['alku_pvm']) &
    (df_rtg_merged['date'] <= df_rtg_merged['loppu_pvm'])
]

# Aggregate costs and counts by episode_id
contacts_costs = df_contacts_diagn_filtered.groupby('episode_id')['price'].sum().rename('contacts_cost')
labs_costs = df_lab_filtered.groupby('episode_id')['price'].sum().rename('labs_cost')
rtg_costs = df_rtg_filtered.groupby('episode_id')['price'].sum().rename('rtg_cost')

# Count number of contacts, labs, and rtg per episode_id
n_contacts = df_contacts_diagn_filtered.groupby('episode_id').size().rename('n_contacts')
n_labs = df_lab_filtered.groupby('episode_id').size().rename('n_labs')
n_rtg = df_rtg_filtered.groupby('episode_id').size().rename('n_rtg')

# Extract first contact details
first_contact = df_contacts_diagn_filtered.sort_values('date').groupby('episode_id')[['profession', 'contact_mode']].first().reset_index()
first_contact['ensi_kontakti'] = first_contact['profession'] + " " + first_contact['contact_mode']

# Merge aggregated data into df_episodi
df_episodi_30 = df_episodi_30.merge(contacts_costs, on='episode_id', how='left')
df_episodi_30 = df_episodi_30.merge(labs_costs, on='episode_id', how='left')
df_episodi_30 = df_episodi_30.merge(rtg_costs, on='episode_id', how='left')
df_episodi_30 = df_episodi_30.merge(n_contacts, on='episode_id', how='left')
df_episodi_30 = df_episodi_30.merge(n_labs, on='episode_id', how='left')
df_episodi_30 = df_episodi_30.merge(n_rtg, on='episode_id', how='left')
df_episodi_30 = df_episodi_30.merge(first_contact[['episode_id', 'ensi_kontakti']], on='episode_id', how='left')

# Fill missing values in cost and count columns
df_episodi_30['contacts_cost'] = df_episodi_30['contacts_cost'].fillna(0)
df_episodi_30['labs_cost'] = df_episodi_30['labs_cost'].fillna(0)
df_episodi_30['rtg_cost'] = df_episodi_30['rtg_cost'].fillna(0)
df_episodi_30['n_contacts'] = df_episodi_30['n_contacts'].astype(int)
df_episodi_30['n_labs'] = df_episodi_30['n_labs'].fillna(0).astype(int)
df_episodi_30['n_rtg'] = df_episodi_30['n_rtg'].fillna(0).astype(int)

# Calculate total costs
df_episodi_30['kustannus'] = df_episodi_30['contacts_cost'] + df_episodi_30['labs_cost'] + df_episodi_30['rtg_cost']

# Display the updated dataframe
df_episodi_30




# Merge df_covariates into df_episodi based on the id
df_episodi_30 = df_episodi_30.merge(
    df_covariates,
    on='id',  # Match based on the 'id' column
    how='left'  # Use left join to keep all episodes in df_episodi
)

df_episodi_30 = df_episodi_30[
    ~((df_episodi_30['diagnosis_group'] == 'urinary') & 
      ((df_episodi_30['age'] > 65)))
]

df_episodi_30= df_episodi_30.drop_duplicates()
# Reset the episode_id to start from 1
df_episodi_30['episode_id'] = range(1, len(df_episodi_30) + 1)

df_episodi_30




# Create a new column 'hoitoketju' based on the condition in 'ensi_kontakti'
df_episodi_30.rename(columns={'digi': 'hoitoketju'}, inplace=True)

# Update the values in the 'hoitoketju' column
df_episodi_30['hoitoketju'] = df_episodi_30['hoitoketju'].apply(lambda x: 'digi' if x == 'DIGI' else 'physical' if x == 'MUU' else x)

# Filter out rows where diagnoosiryhmä is 'urinary' and age > 65
df_episodi_30 = df_episodi_30[~((df_episodi_30['diagnosis_group'] == 'urinary') & (df_episodi_30['age'] > 65))]




df_episodi_30['sex'] = df_episodi_30['sex'].map({'Mies': 0, 'Nainen': 1})

# Perform PSM for each diagnoosiryhmä
matched_data_30 = perform_psm_with_polynomial_features(
    data=df_episodi_30, 
    group_column='diagnosis_group',
    treatment_column='hoitoketju',  # Treatment indicator column
    covariate_columns=covariate_columns,
    caliper=0.2  # Adjust caliper as necessary, caliper sets the maximum allowable distance (or difference) between the propensity scores of a treated unit and a control unit to be matched
)

outcome_column = 'kustannus'
group_column = 'diagnosis_group'
treatment_column = 'hoitoketju'

ttest_results = perform_ttest_by_group(
    matched_data=matched_data_30,
    outcome_column=outcome_column,
    group_column=group_column,
    treatment_column=treatment_column
)

# Display results
ttest_results



# Enhance results with savings calculations
def calculate_savings(ttest_results):
    ttest_results['percentage_saved'] = (
        (ttest_results['mean_physical (euro)'] - ttest_results['mean_digital (euro)']) /
        ttest_results['mean_physical (euro)'] * 100
    ).round(2)

    ttest_results['total_savings'] = (
        (ttest_results['mean_physical (euro)'] - ttest_results['mean_digital (euro)']) *
        ttest_results['n_digital']
    ).round(2)

    # Calculate overall savings and append to the DataFrame
    overall_mean_physical = (
        ttest_results['mean_physical (euro)'] * ttest_results['n_physical']
    ).sum() / ttest_results['n_physical'].sum()

    overall_mean_digital = (
        ttest_results['mean_digital (euro)'] * ttest_results['n_digital']
    ).sum() / ttest_results['n_digital'].sum()

    overall_savings = overall_mean_physical - overall_mean_digital
    overall_percentage_saved = (overall_savings / overall_mean_physical * 100).round(2)
    overall_total_savings = (
        overall_savings * ttest_results['n_digital'].sum()
    ).round(2)

    # Append overall results as a new row
    overall_summary = pd.DataFrame({
        'diagnosis_group': ['Overall'],
        'mean_digital (euro)': [overall_mean_digital],
        'mean_physical (euro)': [overall_mean_physical],
        'percentage_saved': [overall_percentage_saved],
        'total_savings': [overall_total_savings],
        'n_digital': [ttest_results['n_digital'].sum()],
        'n_physical': [ttest_results['n_physical'].sum()],
        't_stat': [None],
        'p_value': [None]
    })

    ttest_results = pd.concat([ttest_results, overall_summary], ignore_index=True)

    return ttest_results


# Apply savings calculation
business_summary = calculate_savings(ttest_results)

# Display business summary
business_summary



# Grouping data into digital and physical groups
physical_data = matched_data_30[matched_data_30['hoitoketju'] == 'physical']
digital_data = matched_data_30[matched_data_30['hoitoketju'] == 'digi']

# Function to calculate mean costs and p-value
def calculate_cost_stats(physical_cost, digital_cost):
    # Calculate mean for both groups
    physical_mean = physical_cost.mean()
    digital_mean = digital_cost.mean()
    
    # Perform independent t-test to get p-value
    _, p_value = ttest_ind(physical_cost, digital_cost, equal_var=False)
    
    return physical_mean, digital_mean, p_value

# Initialize an empty dictionary to store the results
table_stats = {}

# Visit Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['contacts_cost'], digital_data['contacts_cost']
)
table_stats['Visit costs'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}

# Laboratory Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['labs_cost'], digital_data['labs_cost']
)
table_stats['Laboratory costs'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}

# Imaging Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['rtg_cost'], digital_data['rtg_cost']
)
table_stats['Imaging costs'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}


# Total Costs
physical_mean, digital_mean, p_value = calculate_cost_stats(
    physical_data['kustannus'], digital_data['kustannus']
)
table_stats['Total'] = {
    'Physical (€)': physical_mean,
    'Digital (€)': digital_mean,
    'P-value': p_value
}

# Create a DataFrame to display the results
cost_table_df = pd.DataFrame.from_dict(table_stats, orient='index')

# Formatting the results to two decimal points for readability
cost_table_df = cost_table_df.applymap(lambda x: f"{x:.10e}" if isinstance(x, float) else x)

# Display the resulting table
cost_table_df.round(2)


# ### Strata sensitivity analysis


from sklearn.linear_model import LogisticRegression

# Calculate propensity scores per diagnosis group
def calculate_propensity_scores_per_group(data, group_column, treatment_column, covariate_columns):
    """
    Calculate propensity scores for each diagnosis group separately.
    """
    data['propensity_scores'] = np.nan  # Initialize column for propensity scores

    for group in data[group_column].unique():
        group_data = data[data[group_column] == group].copy()

        # Explicitly encode treatment variable
        group_data['treatment'] = group_data[treatment_column].map({'digi': 1, 'physical': 0})

        # Fit logistic regression for propensity scores
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(group_data[covariate_columns], group_data['treatment'])

        # Predict propensity scores for this group
        data.loc[group_data.index, 'propensity_scores'] = log_reg.predict_proba(group_data[covariate_columns])[:, 1]

    return data

# Create strata per diagnosis group
def create_strata_per_group(data, group_column, propensity_score_column, n_strata=5):
    """
    Create strata for each diagnosis group separately based on propensity scores.
    """
    data['strata'] = np.nan  # Initialize column for strata

    for group in data[group_column].unique():
        group_data = data[data[group_column] == group].copy()

        # Create strata based on propensity scores
        data.loc[group_data.index, 'strata'] = pd.qcut(
            group_data[propensity_score_column], q=n_strata, labels=False
        )

    return data

# Check balance within strata for each diagnosis group
def check_balance_per_group_global_sd(data, group_column, strata_column, treatment_column, propensity_score_column):
    """
    Check balance for each diagnosis group and stratum using global SD.
    """
    balance_info = []

    for group in data[group_column].unique():
        group_data = data[data[group_column] == group]

        # Calculate the global SD of propensity scores for the entire group
        global_sd = group_data[propensity_score_column].std()

        for stratum in group_data[strata_column].dropna().unique():
            stratum_data = group_data[group_data[strata_column] == stratum]

            # Calculate mean propensity scores for treated and control groups
            treated_mean = stratum_data.loc[
                stratum_data[treatment_column] == 'digi', propensity_score_column
            ].mean()
            control_mean = stratum_data.loc[
                stratum_data[treatment_column] == 'physical', propensity_score_column
            ].mean()

            # Calculate Standardized Mean Difference (SMD) using global SD
            smd = (treated_mean - control_mean) / global_sd if global_sd > 0 else 0

            # Store results
            balance_info.append({
                "Diagnosis Group": group,
                "Stratum": stratum,
                "Treated Mean": treated_mean,
                "Control Mean": control_mean,
                "Global SD": global_sd,
                "SMD": smd
            })

    balance_info_df = pd.DataFrame(balance_info)
    return balance_info_df

covariate_columns = ['age', 'cci', 'n_laakari']  

# Calculate propensity scores
matched_data_strata = calculate_propensity_scores_per_group(
    data=matched_data,
    group_column='diagnosis_group',
    treatment_column='hoitoketju',
    covariate_columns=covariate_columns
)

matched_data_strata = create_strata_per_group(
    data=matched_data_strata,
    group_column='diagnosis_group',
    propensity_score_column='propensity_scores',
    n_strata=5
)

# Check balance within strata using global SD
balance_info_global_sd = check_balance_per_group_global_sd(
    data=matched_data_strata,
    group_column='diagnosis_group',
    strata_column='strata',
    treatment_column='hoitoketju',
    propensity_score_column='propensity_scores'
)

# Sort and display balance metrics
balance_info_global_sd = balance_info_global_sd.sort_values(by=["Diagnosis Group", "Stratum"]).reset_index(drop=True)
print(balance_info_global_sd)



# Identify problematic strata with SMD > 0.1
problematic_strata = balance_info_global_sd[balance_info_global_sd["SMD"] > 0.1][["Diagnosis Group", "Stratum"]]

# Merge problematic strata to flag rows in matched_data
matched_data_strata = matched_data_strata.merge(
    problematic_strata,
    how="left",
    left_on=["diagnosis_group", "strata"],
    right_on=["Diagnosis Group", "Stratum"],
    indicator=True
)

# Retain only rows NOT in problematic strata
filtered_data = matched_data_strata[matched_data_strata["_merge"] == "left_only"].drop(
    columns=["Diagnosis Group", "Stratum", "_merge"]
)

# Validate the filtered dataset
print("Original data size:", matched_data_strata.shape[0])
print("Filtered data size:", filtered_data.shape[0])
print("Excluded rows:", matched_data_strata.shape[0] - filtered_data.shape[0])

# Display the filtered DataFrame
print(filtered_data)




# Perform t-tests for each diagnoosiryhmä
outcome_column = 'kustannus'
group_column = 'diagnosis_group'
treatment_column = 'hoitoketju'

ttest_results_strata = perform_ttest_by_group(
    matched_data=filtered_data,
    outcome_column=outcome_column,
    group_column=group_column,
    treatment_column=treatment_column
)

# Display results
ttest_results_strata


# ### Inverse Probability of Treatment Weighing (IPTW)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

def calculate_iptw_weights_with_polynomial_features(data, group_column, treatment_column, degree=3):
    """
    Calculate IPTW weights for each diagnosis group, incorporating polynomial features.
    """
    # Initialize weights and propensity scores columns
    data['iptw_weights'] = np.nan
    data['propensity_score'] = np.nan

    for group in data[group_column].unique():
        print(f"Processing group: {group}")
        # Subset data by diagnosis group
        group_data = data[data[group_column] == group].copy()

        # Encode treatment variable (1 = digi, 0 = physical)
        group_data['treatment'] = group_data[treatment_column].map({'digi': 1, 'physical': 0})

        # Explicitly encode 'sex' to ensure Female = 1 and Male = 0
        group_data['sex'] = group_data['sex'].map({'Nainen': 1, 'Mies': 0})

        # Generate polynomial features for 'age' and 'n_laakari'
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        age_poly = poly.fit_transform(group_data[['age']])
        n_laakari_poly = poly.fit_transform(group_data[['n_laakari']])

        # Create dataframes for polynomial features with appropriate column names
        age_poly_df = pd.DataFrame(age_poly, columns=[f'age^{i+1}' for i in range(age_poly.shape[1])], index=group_data.index)
        n_laakari_poly_df = pd.DataFrame(n_laakari_poly, columns=[f'n_laakari^{i+1}' for i in range(n_laakari_poly.shape[1])], index=group_data.index)

        # Combine polynomial features with other covariates
        covariates = pd.concat([group_data[['sex', 'cci']], age_poly_df, n_laakari_poly_df], axis=1)

        # Fit logistic regression to estimate propensity scores
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(covariates, group_data['treatment'])


        # Predict propensity scores
        propensity_scores = log_reg.predict_proba(covariates)[:, 1]

        # Trim propensity scores to the range [0.01, 0.99]
        propensity_scores = np.clip(propensity_scores, 0.01, 0.99)

        # Assign trimmed propensity scores back to the group_data DataFrame
        group_data['propensity_score'] = propensity_scores

        # Calculate IPTW weights
        group_data['iptw_weights'] = np.where(
            group_data['treatment'] == 1,  # Treated (digi)
            1 / group_data['propensity_score'],  # Weight for treated
            1 / (1 - group_data['propensity_score'])  # Weight for control (physical)
            )

        # Assign weights and scores back to the main dataframe
        data.loc[group_data.index, 'iptw_weights'] = group_data['iptw_weights']
        data.loc[group_data.index, 'propensity_score'] = group_data['propensity_score']

    return data

# Applying the function on your dataset
df_episodi = calculate_iptw_weights_with_polynomial_features(
    data=df_episodi,
    group_column='diagnosis_group',
    treatment_column='hoitoketju',
    degree=3  # Polynomial degree for 'age' and 'n_laakari'
)


# Calculate weighted means for total costs (kustannus) by diagnosis group and pathway
weighted_results = df_episodi.groupby(['diagnosis_group', 'hoitoketju']).apply(
    lambda x: np.average(x['kustannus'], weights=x['iptw_weights'])
).reset_index(name='weighted_mean_cost')


import statsmodels.api as sm

# Initialize dictionary to store regression results by diagnosis group
regression_results = {}

# Perform weighted regression for each diagnosis group
for group in df_episodi['diagnosis_group'].unique():
    group_data = df_episodi[df_episodi['diagnosis_group'] == group].copy()

    # Add intercept for the regression model
    group_data['intercept'] = 1
    
    # Ensure `hoitoketju` is converted to numeric (e.g., binary: 1 for "digi" and 0 for "physical")
    group_data['hoitoketju_numeric'] = group_data['hoitoketju'].map({'digi': 1, 'physical': 0})

    # Weighted regression
    model = sm.WLS(
        group_data['kustannus'],  # Outcome
        group_data[['intercept', 'hoitoketju_numeric']],  # Predictors
        weights=group_data['iptw_weights']  # Weights
    )
    results = model.fit()

    # Store results
    regression_results[group] = results.summary()
    print(f"Regression results for {group}:\n", results.summary())


# ### Calipers


calipers = [0.01, 0.2, 1.0]
psm_results = []
ttest_results = []

for caliper in calipers:
    # Perform PSM for each caliper value
    matched_data = perform_psm_with_polynomial_features(
        data=df_episodi,
        group_column='diagnosis_group',
        treatment_column='hoitoketju',
        covariate_columns=covariate_columns,
        caliper=caliper
    )
    # Store the matched data
    psm_results.append(matched_data)
    
    # Perform t-test on the matched data
    ttest_result = perform_ttest_by_group(
        matched_data=matched_data,
        outcome_column='kustannus',  
        group_column='diagnosis_group',  
        treatment_column='hoitoketju' 
    )
    
    # Add a column for the caliper value for tracking
    ttest_result['caliper'] = caliper
    
    # Store the t-test results
    ttest_results.append(ttest_result)

# Combine the t-test results for all calipers into a single DataFrame
combined_ttest_results = pd.concat(ttest_results, ignore_index=True)
combined_ttest_results = combined_ttest_results.sort_values(by=['diagnosis_group', 'caliper'], ignore_index=True)

combined_ttest_results


