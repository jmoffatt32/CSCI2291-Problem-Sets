import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

covid_data = pd.read_csv("HW4/python/dataset/owid-covid-data.csv")
country_labels = np.unique(covid_data.iso_code)
correlations_array = np.zeros(len(country_labels))

for i in range(0, len(country_labels)):
    code = country_labels[i]
    vax_dataset = covid_data.people_fully_vaccinated_per_hundred.loc[covid_data.iso_code == code]
    deaths_dataset = covid_data.new_deaths_smoothed_per_million.loc[covid_data.iso_code == code]
    country_data = pd.DataFrame({"people_fully_vaccinated_per_hundred" : vax_dataset, "new_deaths_smoothed_per_million" : deaths_dataset})
    country_data = country_data.loc[np.logical_not(np.isnan(vax_dataset))]
    country_data = country_data.loc[np.logical_not(np.isnan(deaths_dataset))]
    coerr_coef = country_data['people_fully_vaccinated_per_hundred'].corr(country_data['new_deaths_smoothed_per_million'])
    correlations_array[i] = coerr_coef

correlations_array = correlations_array[np.logical_not(np.isnan(correlations_array))]
print(np.median(correlations_array))

plt.boxplot(correlations_array, notch = True)
plt.title("Correlation Coefficents Between Vaccination Rate and \nDeath Rate per Million")
plt.ylim([-1,1])
plt.show()
