import pandas as pd

def clean_string(string):
    return string.replace("\"", "").strip()

geographic = pd.read_csv("HW4/python/dataset/countries_codes_and_coordinates.csv")
geographic["Alpha-3 code"] = geographic["Alpha-3 code"].apply(clean_string)
geographic["Latitude (average)"] = geographic["Latitude (average)"].apply(clean_string)