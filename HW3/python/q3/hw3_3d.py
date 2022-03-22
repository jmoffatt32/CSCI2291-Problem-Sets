import pandas as pd
import numpy as np
import scipy.stats as stat

covid_variants = pd.read_csv("HW3/python/datasets/covid-variants.csv")
us_variants = covid_variants.loc[covid_variants.location == "United States"]
canada_variants = covid_variants.loc[covid_variants.location == "Canada"]

us_perc_sequences = us_variants.perc_sequences.to_numpy()
canada_perc_sequences = canada_variants.perc_sequences.to_numpy()

normalized_us =  (us_perc_sequences - np.mean(us_perc_sequences)) / np.std(us_perc_sequences)
normalized_canada = (canada_perc_sequences - np.mean(canada_perc_sequences)) / np.std(canada_perc_sequences)

t_test = stat.ttest_ind(normalized_us, normalized_canada)

print(t_test)