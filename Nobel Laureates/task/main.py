import pandas as pd
import os
import requests
import sys
import re
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'Nobel_laureates.json' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/m6ld4vaq2sz3ovd/nobel_laureates.json?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/Nobel_laureates.json', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")

    # write your code here
df = pd.read_json('../Data/Nobel_laureates.json')

df = df.dropna(subset=['gender'])

def extract_country(place):
    if pd.isna(place) or ',' not in place:
        return None
    parts = place.split(',')
    return parts[-1].strip()

# Apply the function to the place_of_birth column
df['extracted_country'] = df['place_of_birth'].apply(extract_country)

# Fill missing values in born_in with the extracted country
df['born_in'] = df['born_in'].where(df['born_in'] != '', df['extracted_country'])

# Drop rows where born_in is still empty
df.dropna(subset=['born_in'], inplace=True)

# Standardize country names
df['born_in'] = df['born_in'].replace({
    'US': 'USA',
    'United States': 'USA',
    'U.S.': 'USA',
    'United Kingdom': 'UK'
})

# Reset the DataFrame index
df.reset_index(drop=True, inplace=True)

def extract_year_of_birth(dob):
    if pd.isna(dob):
        return None
    year_search = re.search(r'\b(1[89]\d{2}|20[01]\d)\b', dob)
    if year_search:
        return int(year_search.group(0))
    return None

# Apply the function to create a new column for the year of birth
df['year_born'] = df['date_of_birth'].apply(extract_year_of_birth)

# Calculate the age of winning the Nobel Prize
df['age_of_winning'] = df.apply(lambda row: (row['year'] - row['year_born']) if not pd.isna(row['year_born']) else None,
                                axis=1)

# Remove entries with missing 'year_born' or 'age_of_winning'
df.dropna(subset=['year_born', 'age_of_winning'], inplace=True)

# Convert 'year_born' and 'age_of_winning' to integers (cleaning NaN and converting float to int)
df['year_born'] = df['year_born'].astype(int)
df['age_of_winning'] = df['age_of_winning'].astype(int)

# Output the lists of year born and age of winning
year_born_list = df['year_born'].tolist()
age_of_winning_list = df['age_of_winning'].tolist()

country_counts = df['born_in'].value_counts()

# Re-code small categories into 'Other countries'
threshold = 25
other_countries = country_counts[country_counts < threshold].sum()
country_counts = country_counts[country_counts >= threshold]
if other_countries > 0:
    country_counts['Other countries'] = other_countries

# Explode larger slices
explode = [0.08 if count > 100 else 0 for count in country_counts]
colors = ['blue', 'orange', 'red', 'yellow', 'green', 'pink', 'brown', 'cyan', 'purple']
# Colors and figure size as per example
def custom_autopct(pct):
    total = sum(country_counts)
    value = int(round(pct/100.*total))
    return f"{pct:.2f}%\n({value})"

# Plotting the pie chart with a function for autopct
#  plt.figure(figsize=(12, 12))
#  plt.pie(country_counts, labels=country_counts.index, autopct=custom_autopct, startangle=90,
#        colors=colors[:len(country_counts)], explode=explode)
#  plt.axis('equal')

df.dropna(subset=['category'], inplace=True)

# Create a pivot table to count laureates by gender and category
category_gender = df.pivot_table(index='category', columns='gender', aggfunc='size', fill_value=0)
category_gender = category_gender.iloc[1:]

# Plotting the grouped bar chart
# plt.figure(figsize=(10, 10))
# bar_width = 0.4
# categories = category_gender.index
# index = np.arange(len(categories))
#
# fig, ax = plt.subplots()
# bars1 = ax.bar(index - bar_width/2, category_gender['male'], width=bar_width, label='Males', color='blue')
# bars2 = ax.bar(index + bar_width/2, category_gender['female'], width=bar_width, label='Females', color='crimson')
#
# ax.set_xlabel('Category', fontsize=14)
# ax.set_ylabel('Counts', fontsize=14)
# ax.set_title('Count of Male and Female Nobel Laureates by Category', fontsize=20)
# ax.set_xticks(index)
# ax.set_xticklabels(categories)
# ax.legend()


grouped = df.groupby('category')['age_of_winning'].apply(list).reset_index(name='Ages')

# Prepare the data for box plot
ages = grouped['Ages'].values.tolist()
categories = grouped['category'].tolist()
categories.pop(0)

categories.append('All categories')


plt.figure(figsize=(10, 10))
box = plt.boxplot(ages, labels=categories, patch_artist=True, meanline=False, showmeans=True)

# Customizing the plot appearance
plt.xlabel('Category', fontsize=14)
plt.ylabel('Age of obtaining Nobel Prize', fontsize=14)
plt.title('Distribution of Ages by Category', fontsize=20)

plt.show()
