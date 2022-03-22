import pandas as pd

covid_variants = pd.read_csv("HW3/python/datasets/covid-variants.csv")
us_variants = covid_variants.loc[covid_variants.location == "United States"]
weeks_list = us_variants.date.unique()

count = 0

for week in range(0, len(weeks_list)):
    if weeks_list[week][0:4] == '2021':
        num_sequences_total = us_variants.num_sequences_total.loc[us_variants.date == weeks_list[week]].mean()
        count += num_sequences_total

print(f"Total number of US records in 2021: {int(count)}")


