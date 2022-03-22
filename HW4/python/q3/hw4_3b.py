import pandas as pd
import numpy as np
from hw4_3a import iso_code_to_int

covid_data = pd.read_csv("HW4/python/dataset/owid-covid-data.csv")
country_labels = np.unique(covid_data.iso_code)
deaths_dict = {}
hash_table = {}


for code in country_labels:
    max_num_deaths = covid_data.total_deaths_per_million.loc[covid_data.iso_code == code].max()
    hash_table[iso_code_to_int(code)] = code
    deaths_dict[iso_code_to_int(code)] = max_num_deaths

deaths_data = np.array(list(deaths_dict.items()))

first_q = np.nanquantile(deaths_data[:,1], .25)
third_q = np.nanquantile(deaths_data[:,1], .75)

q1, q3 = [], []

for key in deaths_dict:
    if deaths_dict[key] <= first_q:
        q1.append(hash_table[key])
    if deaths_dict[key] >= third_q:
        q3.append(hash_table[key])

print(f"q1: {q1}")
print(f"q3: {q3}")

