import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

covid_data = pd.read_csv("HW4/python/dataset/owid-covid-data.csv")
country_labels = np.unique(covid_data.iso_code)
country_data = np.zeros(shape=(len(country_labels),2))

for i in range(0, len(country_labels)):
    code = country_labels[i]
    vax_dataset = covid_data.people_fully_vaccinated_per_hundred.loc[covid_data.iso_code == code]
    deaths_dataset = covid_data.total_deaths_per_million.loc[covid_data.iso_code == code]
    country_data[i] = [vax_dataset.max(), deaths_dataset.max()]

country_data = country_data[np.logical_not(np.isnan(country_data[:,0]))]
country_data = country_data[np.logical_not(np.isnan(country_data[:,1]))]

print(f"{'Correlation Coefficent:' :<25}{round(np.corrcoef(country_data, rowvar=False)[0][1], 3)}")

plt.scatter(country_data[:,0], country_data[:,1])
plt.title("Vaccination Rate vs. Total Deaths per Million (by country)")
plt.xlabel("Total Percentage of People Fully Vaccinated")
plt.ylabel("Total Number of Deaths per Million People")
plt.xlim([0,100])
plt.ylim(bottom=0)
plt.show()