import pandas as pd
from pprint import pprint
from matplotlib import pyplot as plt

covid_variants = pd.read_csv("HW3/python/datasets/covid-variants.csv")
us_variants = covid_variants.loc[covid_variants.location == "United States"]

plt.boxplot([us_variants.num_sequences, us_variants.perc_sequences, us_variants.num_sequences_total])
plt.show()