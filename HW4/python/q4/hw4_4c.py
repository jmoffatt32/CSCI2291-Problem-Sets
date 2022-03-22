from hw4_4b import lat_high_mortality, lat_low_mortality
import numpy as np
import matplotlib.pyplot as plt

low_array = np.array([lat for lat in lat_low_mortality if not np.isnan(lat)])
high_array = np.array([lat for lat in lat_high_mortality if not np.isnan(lat)])

low_median = np.median(low_array)
high_median = np.median(high_array)

print(f"{'Low Median:' :<25}{low_median}")
print(f"{'High Median:' :<25}{high_median}")

plt.boxplot(low_array, notch = True, positions=[0], labels=["Bottom Quartile of \nMax Deaths per Million"])
plt.boxplot(high_array, notch = True, positions=[1], labels=["Top Quartile of \nMax Deaths per Million"])
plt.title("Average Latitudes of Countries in the \nTop and Bottom Quartiles of \nCOVID-19 Deaths per Million People")
plt.ylabel("Average Latitude")
plt.show()
