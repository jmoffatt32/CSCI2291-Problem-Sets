import pandas as pd
import numpy as np
from hw4_4a import clean_string

def iso_code_to_int(code):
    out = 0
    for i, c in enumerate(code):
        num = ord(c)
        power = len(code) - (i + 1)
        out += num * (256 ** (power))
    return out

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

geographic = pd.read_csv("HW4/python/dataset/countries_codes_and_coordinates.csv")
geographic["Alpha-3 code"] = geographic["Alpha-3 code"].apply(clean_string)
geographic["Latitude (average)"] = geographic["Latitude (average)"].apply(clean_string)
geographic = geographic.set_index("Alpha-3 code")

def code_to_lat(code):
    try:
        return float(geographic.loc[code, "Latitude (average)"])
    except:
        return np.nan

lat_low_mortality, lat_high_mortality = [], []

for code in q1:
    lat_low_mortality.append(code_to_lat(code))

for code in q3:
    lat_high_mortality.append(code_to_lat(code))

print(f"Low Mortality: {lat_low_mortality}")
print(f"High Mortality: {lat_high_mortality}")



        
    



