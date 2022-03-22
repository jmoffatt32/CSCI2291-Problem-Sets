import pandas as pd

covid_variants = pd.read_csv("HW3/python/datasets/covid-variants.csv")
us_variants = covid_variants.loc[covid_variants.location == "United States"]
canada_variants = covid_variants.loc[covid_variants.location == "Canada"]

us_perc_sequences = us_variants.perc_sequences
canada_perc_sequences = canada_variants.perc_sequences

us_mean = us_perc_sequences.mean()
canada_mean = canada_perc_sequences.mean()

print(f"{'US Mean' :<15}{us_mean}")
print(f"{'Canada Mean' :<15}{canada_mean}")
