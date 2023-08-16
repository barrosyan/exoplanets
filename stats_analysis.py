filename = 'PS_2023.08.14_09.36.52.csv'

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, f_oneway
from scipy.stats import kruskal
from scipy.stats import shapiro

exoplanets = pd.read_csv(filename, skiprows=96)
print(exoplanets.head(10))

print(exoplanets.columns)

print(exoplanets.info())

print(exoplanets.describe())

#Analysis of Exoplanet Discoveries by Discovery Method:

#**Hypothesis:** The methods of exoplanet discovery exhibit significant differences in terms of the number of discoveries.

#**Statistical Method:** Chi-Square Test or ANOVA to compare the frequencies of discoveries by method.

contingency_table = pd.crosstab(exoplanets['discoverymethod'], columns='count')
unique_methods = exoplanets['discoverymethod'].unique()
for method in unique_methods:
  print(method)

chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

radial_velocity = exoplanets['discoverymethod'] == 'Radial Velocity'
imaging = exoplanets['discoverymethod'] == 'Imaging'
eclipse_timing_variations = exoplanets['discoverymethod'] == 'Eclipse Timing Variations'
transit = exoplanets['discoverymethod'] == 'Transit'
astrometry = exoplanets['discoverymethod'] == 'Astrometry'
disk_kinematics = exoplanets['discoverymethod'] == 'Disk Kinematics'
microlensing = exoplanets['discoverymethod'] == 'Microlensing'
orbital_brightness_modulation = exoplanets['discoverymethod'] == 'Orbital Brightness Modulation'
pulsation_timing_variations = exoplanets['discoverymethod'] == 'Pulsation Timing Variations'
transit_timing_variations = exoplanets['discoverymethod'] == 'Transit Timing Variations'
pulsar_timing = exoplanets['discoverymethod'] == 'Pulsar Timing'

anova_results = f_oneway(exoplanets[radial_velocity].count(), exoplanets[imaging].count(), exoplanets[eclipse_timing_variations].count(),
                         exoplanets[transit].count(), exoplanets[astrometry].count(), exoplanets[disk_kinematics].count(),
                         exoplanets[microlensing].count(), exoplanets[orbital_brightness_modulation].count(),
                         exoplanets[pulsation_timing_variations].count(), exoplanets[transit_timing_variations].count(),
                         exoplanets[pulsar_timing].count())

print("Chi-Square Test Results:")
print(f"Chi-Square Statistic: {chi2_stat}")
print(f"P-Value: {p_val}")

print("\nANOVA Test Results:")
print(f"F-Statistic: {anova_results.statistic}")
print(f"P-Value: {anova_results.pvalue}")

# LOOKING TO THE DISTRIBUTION OF THIS DATA TO AVALUATE THIS CONFLICT BETWEEN ANOVA TOO SMALL AND CHI-SQUARE TOO HIGH
num_methods = len(unique_methods)
fig, axes = plt.subplots(nrows=num_methods, ncols=1, figsize=(10, 3*num_methods))

for idx, method in enumerate(unique_methods):
    ax = axes[idx]
    ax.hist(exoplanets[exoplanets['discoverymethod'] == method].count(), alpha=0.5, label=method)

    ax.set_xlabel('Count of Discoveries')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Discoveries by Method: {method}')
    ax.legend()

plt.tight_layout()

plt.show()

num_methods = len(unique_methods)
fig, axes = plt.subplots(nrows=num_methods, ncols=1, figsize=(8, 3*num_methods))  # Adjust the figsize here

for idx, method in enumerate(unique_methods):
    ax = axes[idx]
    sns.kdeplot(exoplanets[exoplanets['discoverymethod'] == method].count(), ax=ax, label=method)

    ax.set_xlabel('Count of Discoveries')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of Discoveries by Method: {method}')
    ax.legend()

plt.tight_layout()

plt.show()

method_data = []
for method in unique_methods:
    method_data.append(exoplanets[exoplanets['discoverymethod'] == method].count())

kruskal_stat, p_value = kruskal(*method_data)

print("Kruskal-Wallis Test Results:")
print("Kruskal-Wallis Statistic:", kruskal_stat)
print("P-Value:", p_value)

#I initially conducted the chi-square and ANOVA tests and observed discrepancies between their outcomes. Consequently, I visualized the distributions of the labels and noted that they follow non-parametric distributions. Thus, I re-conducted the second test, replacing ANOVA with Kruskal-Wallis and re-evaluated the results.

#This can be interpreted as an indication that exoplanet discovery methods are indeed yielding different outcomes in terms of discovery counts, providing support for the analysis that the non-parametric distributions exhibit significant differences.

#Distribution of Exoplanet Masses:

#**Hypothesis:** The distribution of exoplanet masses follows a normal distribution.

#**Statistical Method:** Normality test, such as the Shapiro-Wilk Test.

statistic, p_value = shapiro(exoplanets['pl_bmassj'])

print("Shapiro-Wilk Test Results:")
print("Test Statistic", statistic)
print("P-Value", p_value)

alpha = 0.05

if p_value > alpha:
  print("The data follows a normal distribution (fail to reject null hypothesis).")
else:
  print("The data does not follows a normal distribution (reject null hypothesis).")

stats.probplot(exoplanets['pl_bmassj'], dist="norm", plot=plt)
plt.title("Q-Q Plot of Exoplanet Masses")
plt.xlabel("Theorethical Quantiles")
plt.ylabel("Sample Quantiles")
plt.show()

#In summary, the analysis of exoplanet masses revealed that the Shapiro-Wilk test yielded a p-value of 1.0, suggesting no significant departure from normality. However, the Q-Q plot displayed an upward exponential curvature at the end, indicating deviations from normality, particularly in the tails. This suggests that while the Shapiro-Wilk test did not reject normality, visual inspection of the Q-Q plot hints at the presence of heavy tails in the distribution. These heavy tails could imply the presence of outliers or rare events in the exoplanet mass data. Researchers should consider exploring outlier identification, data transformation, or robust statistical methods to handle the potential influence of these outliers and non-normal distribution characteristics in further analyses.

#Relationship between Exoplanet Mass and Host Star Mass:

#**Hypothesis:** There is a positive correlation between the exoplanet mass and the mass of the host star.

#**Statistical Method:** Correlation test, such as the Pearson correlation coefficient.

correlation_coefficient = exoplanets['pl_bmassj'].corr(exoplanets['st_mass'])
plt.figure(figsize=(8, 6))
sns.scatterplot(data=exoplanets, x='st_mass', y='pl_bmassj')
plt.title("Correlation Between Exoplanet Mass and Host Star Mass")
plt.xlabel("Host Star Mass")
plt.ylabel("Exoplanet Mass")
plt.show()

print("Pearson Correlation Coefficient:", correlation_coefficient)

Based on the analysis of the correlation between exoplanet mass and host star mass, we observed a Pearson correlation coefficient of approximately 0.26. This positive value indicates a weak correlation between the two variables. Although the relationship is not strong, the presence of a positive correlation suggests that, in general, exoplanets with larger masses tend to orbit host stars with larger masses. However, it's important to note that other factors may influence this relationship and that correlation does not necessarily imply a cause-and-effect relationship between exoplanet and star masses.

#Comparison of Exoplanet Masses between Systems with 1 and 2 Host Stars:

**Hypothesis:** The masses of exoplanets differ between systems with different numbers of stars.

**Statistical Method:** Student's t-test.

from scipy.stats import ttest_ind

group_1 = exoplanets[exoplanets['sy_snum'] == 1]['pl_bmassj'].dropna()
group_2 = exoplanets[exoplanets['sy_snum'] == 2]['pl_bmassj'].dropna()

t_statistic, p_value = ttest_ind(group_1, group_2)

print("T-Statistic:", t_statistic)
print("P-Value:", p_value)

After conducting the Student's t-test to compare exoplanet masses between systems with 1 and 2 host stars, we observed a t-statistic value of approximately -1.57 and a p-value of about 0.12. The p-value is greater than the usual significance level of 0.05, indicating that there is not enough evidence to reject the null hypothesis that the exoplanet masses are equal between the two groups. Therefore, I did not find statistically significant differences in exoplanet masses between systems with different numbers of host stars.

#Kepler's Law: Relationship between Semi-Major Axis and Orbital Period:

**Hypothesis:** The relationship between semi-major axis and orbital period follows Kepler's law.

**Statistical Method:** Visual analysis with a scatter plot.

valid_data = exoplanets[['pl_orbsmax', 'pl_orbper']].dropna()
filtered_data = valid_data[valid_data['pl_orbsmax'] < 100]

plt.figure(figsize=(10, 6))
plt.scatter(filtered_data['pl_orbsmax'], filtered_data['pl_orbper'], alpha=0.5)
plt.xlabel("Semi-Major Axis (AU)")
plt.ylabel("Orbital Period (days)")
plt.title("Kepler's Law: Semi-Major Axis vs. Orbital Period (Semi-Major Axis < 1000)")
plt.show()

from scipy.optimize import curve_fit

def polynomial_func(x, a, b, c):
  return a* x**2 + b * x + c

x = filtered_data['pl_orbsmax']
y = filtered_data['pl_orbper']

linear_fit = np.polyfit(x, y, 1)
linear_func = np.poly1d(linear_fit)

popt, _ = curve_fit(polynomial_func, x, y)

plt.figure(figsize=(10,6))
plt.scatter(x, y, alpha=0.5, label='Data')
plt.plot(x, linear_func(x), color='red', label='Linear Fit')
plt.plot(x, polynomial_func(x, *popt), color='green', label='Polynomial Fit')
plt.xlabel("Semi-Major Axis (AU)")
plt.ylabel("Orbital Period (days)")
plt.title("Kepler's Law: Semi-Major Axis vs. Orbital Period (Semi-Major Axis < 100)")
plt.legend()
plt.show()


Based on the results of the analysis of the relationship between Semi-Major Axis and Orbital Period of exoplanets, we observe a strong positive correlation between these two variables. By fitting a polynomial curve to the data, we find that this curve fits well with the experimental data points, indicating a non-linear relationship between Semi-Major Axis and Orbital Period. This is consistent with Kepler's Law, which describes the relationship between the orbital parameters of a planetary system. Therefore, we can conclude that the results support the idea that Kepler's Law provides an accurate description of the relationship between Semi-Major Axis and Orbital Period of the analyzed exoplanets.

#Distribution of Visual and Infrared Magnitudes:

**Hypothesis:** The distributions of visual magnitude, infrared magnitudes, and Gaia magnitude are different.

**Statistical Method:** Kolmogorov-Smirnov Test.

from scipy.stats import ks_2samp

visual_mag = exoplanets['sy_vmag'].dropna()
infrared_mag = exoplanets['sy_kmag'].dropna()
gaia_mag = exoplanets['sy_gaiamag'].dropna()

ks_visual_infrared, p_visual_infrared = ks_2samp(visual_mag, infrared_mag)
ks_visual_gaia, p_visual_gaia = ks_2samp(visual_mag, gaia_mag)
ks_infrared_gaia, p_infrared_gaia = ks_2samp(infrared_mag, gaia_mag)

print("KS Test Results:")
print(f"Visual vs Infrared - KS Statistic: {ks_visual_infrared}, P-Value: {p_visual_infrared}")
print(f"Visual vs Gaia - KS Statistic: {ks_visual_gaia}, P-Value: {p_visual_gaia}")
print(f"Infrared vs Gaia - KS Statistic: {ks_infrared_gaia}, P-Value: {p_infrared_gaia}")

plt.figure(figsize=(10, 6))
sns.histplot(visual_mag, label='Visual Magnitude', color='blue', alpha=0.5)
sns.histplot(infrared_mag, label='Infrared Magnitude', color='orange', alpha=0.5)
sns.histplot(gaia_mag, label='Gaia Magnitude', color='green', alpha=0.5)

plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.title('Distribution of Visual, Infrared, and Gaia Magnitudes')
plt.legend()
plt.show()

These results indicate that magnitudes measured in different wavelength ranges have statistically significant differences among them. This could be related to varying sensitivities of measurement instruments in each wavelength range, light absorption by interstellar medium, and other variables affecting observations at different wavelengths. Therefore, when comparing magnitudes across different wavelength ranges, it's important to consider potential sources of variation that could contribute to these differences.

#Relationship between Distance and Visual Magnitude:

**Hypothesis:** There is a relationship between the distance and the visual magnitude of host stars.

**Statistical Method:** Regression analysis.

from scipy.stats import linregress

alldata = exoplanets[['sy_dist', 'sy_vmag']].dropna()
distance = alldata['sy_dist']
visual_mag = alldata['sy_vmag']

regression_result = linregress(distance, visual_mag)
slope = regression_result.slope
intercept = regression_result.intercept

plt.figure(figsize=(10,6))
sns.scatterplot(x=distance, y=visual_mag, alpha=0.5, label='Data')
plt.plot(distance, slope * distance + intercept, color='red', label='Regression Line')
plt.xlabel('Distance (parsecs)')
plt.ylabel('Visual Magnitude')
plt.title('Relationship between Distance and Visual Magnitude')
plt.legend()
plt.show()

print("Regression Slope:", slope)
print("Regression Intercept:", intercept)
print("R-squared:", regression_result.rvalue ** 2)
print("P-Value:", regression_result.pvalue)

Overall, the results suggest that there is a statistically significant but relatively weak positive relationship between distance and visual magnitude of host stars. The R-squared value indicates that other factors not included in the analysis may also contribute to the variability in visual magnitude.
#Temporal Analysis of Exoplanet Discoveries:

**Hypothesis:** There is a temporal trend in the discoveries of exoplanets over the years.

**Statistical Method:** Time series analysis or bar chart by year.

discoveries_by_year = exoplanets['disc_year'].value_counts().sort_index()

plt.figure(figsize=(10,6))
plt.bar(discoveries_by_year.index, discoveries_by_year.values, color='blue')
plt.xlabel("Year")
plt.ylabel("Number of Discoveries")
plt.title("Temporal Analysis of Exoplanet Discoveries")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

The significant increases in exoplanet discoveries in 2014 and 2016 could be attributed to various factors, including advancements in observation techniques, improvements in data analysis methods, and the launch of new space telescopes or missions that were particularly effective at detecting exoplanets during those years. Additionally, collaborative efforts among different research groups, increased funding, and dedicated exoplanet discovery missions could have also played a role in boosting the number of discoveries during those specific years. It would be beneficial to investigate historical records, scientific publications, and announcements related to exoplanet research during those years to gain a better understanding of the specific factors that contributed to the observed increases.

#Comparison of Stellar Parameters Among Discovery Methods:

**Hypothesis:** There is a difference in stellar properties among different discovery methods.

**Statistical Method:** Student's t-test or ANOVA.

stellar_params = exoplanets[['st_teff', 'st_rad', 'st_mass', 'discoverymethod']]
stellar_params = stellar_params.dropna()

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18))

# Plot Stellar Effective Temperature
sns.boxplot(ax=axes[0], x='discoverymethod', y='st_teff', data=stellar_params)
axes[0].set_xlabel('Discovery Method')
axes[0].set_ylabel('Stellar Effective Temperature (K)')
axes[0].set_title('Comparison of Stellar Effective Temperature Among Discovery Methods')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

# Plot Stellar Radius
sns.boxplot(ax=axes[1], x='discoverymethod', y='st_rad', data=stellar_params)
axes[1].set_xlabel('Discovery Method')
axes[1].set_ylabel('Stellar Radius (Solar Radii)')
axes[1].set_title('Comparison of Stellar Radius Among Discovery Methods')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

# Plot Stellar Mass
sns.boxplot(ax=axes[2], x='discoverymethod', y='st_mass', data=stellar_params)
axes[2].set_xlabel('Discovery Method')
axes[2].set_ylabel('Stellar Mass (Solar Masses)')
axes[2].set_title('Comparison of Stellar Mass Among Discovery Methods')
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()

#Dimensionality Reduction and Clustering

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

selected_columns = ['pl_orbper', 'pl_rade', 'pl_radj', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist', 'st_spectype']

X = exoplanets[selected_columns].dropna()
X

numerical_columns = ['pl_orbper', 'pl_rade', 'pl_radj', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist']
X_numerical = X[numerical_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numerical)

label_encoder = LabelEncoder()
X['st_spectype_encoded'] = label_encoder.fit_transform(X['st_spectype'])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

tsne = TSNE(n_components=2, perplexity=30, random_state=50)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=X['st_spectype_encoded'], cmap='viridis')
plt.title('PCA')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=X['st_spectype_encoded'], cmap='viridis')
plt.title('t-SNE')
plt.colorbar()

plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans

num_clusters = 3

kmeans = KMeans(n_clusters=num_clusters, random_state=42)

kmeans.fit(X_scaled)

cluster_labels = kmeans.labels_

plt.figure(figsize=(30,20))
plt.subplot(1,2,1)
for label in range(121):
    plt.scatter(X_pca[kmeans.labels_ == label, 0], X_pca[kmeans.labels_ == label, 1], label=f'Cluster {label}')
plt.title('PCA')
plt.tight_layout()

plt.subplot(1,2,2)
for label in range(121):
    plt.scatter(X_tsne[kmeans.labels_ == label, 0], X_tsne[kmeans.labels_ == label, 1], label=f'Cluster {label}')
plt.title('t-SNE')
plt.tight_layout()

plt.show()

labels = ['pl_orbper', 'ttv_flag', 'pl_rade', 'pl_orbsmax', 'st_mass', 'st_rad', 'pl_name']
data = exoplanets[labels]

data.dropna(inplace=True)

data.info()

# Scatter Plot
plt.figure(figsize=(12, 6))

# pl_orbper vs pl_orbsmax
plt.subplot(1, 2, 1)
plt.scatter(data['pl_orbper'], data['pl_orbsmax'], alpha=0.5)
plt.xlabel('Orbital Period (days)')
plt.ylabel('Orbital Distance (AU)')
plt.title('Period x Orbital Distance')

# Plotando st_mass vs st_rad
plt.subplot(1, 2, 2)
plt.scatter(data['st_mass'], data['st_rad'], alpha=0.5)
plt.xlabel('Star Mass (M_sun)')
plt.ylabel('Star Radium (R_sun)')
plt.title('Mass x Radium')

plt.tight_layout()
plt.show()

# HISTOGRAM

plt.figure(figsize=(12, 6))

# Histograma de pl_rade
plt.subplot(1, 2, 1)
plt.hist(data['pl_rade'], bins=20, edgecolor='k')
plt.xlabel('Exoplanet Radium (R_earth)')
plt.ylabel('Frequency')
plt.title('Exoplanet Radium Histogram')

# Histograma de st_mass
plt.subplot(1, 2, 2)
plt.hist(data['st_mass'], bins=20, edgecolor='k')
plt.xlabel('Star Mass (M_sun)')
plt.ylabel('Frequency')
plt.title('Star Mass Histogram')

plt.tight_layout()
plt.show()

Predicting Stellar Parameters: Predicting host star parameters using exoplanet data could be beneficial when detailed stellar observations are limited. NASA could appreciate the potential to estimate key star properties indirectly.

# *'pl_orbper' (Orbital Period):*

**Application:**

Determine the planet's year length and its orbital distance from the host star.

**Importance:**

Helps understand the climate, potential habitability, and orbital dynamics of the planet.

# *'pl_rade' (Planet Radius relative to Earth):*

**Application:**

Classify the planet's size compared to Earth.

**Importance:**

 Provides insights into the planet's composition, internal structure, and potential habitability.

# *'pl_bmasse' or 'pl_bmassj' (Planet Mass relative to Earth or Jupiter):*

**Application:**

Assess the planet's mass and compare it with known planets.

**Importance:**

Helps determine the planet's composition, type (terrestrial or gas giant), and influences its characteristics.

# *'pl_insol' (Insolation Flux Received by the Planet):*

**Application:**

Estimate the level of solar radiation received by the planet.

**Importance:**

Crucial for determining whether the surface temperature is suitable for the presence of liquid water.

# *'st_teff' (Effective Temperature of the Host Star):*

**Application:**

Evaluate the level of heat emitted by the star.

**Importance:**

Determines the average temperature of the habitable zone and affects the planet's habitability.

# *'st_rad' (Radius of the Host Star):*

**Application:**

Define the size of the star.

**Importance:**

Affects the habitable zone and the gravitational influence exerted on the planet.

# *'st_mass' (Mass of the Host Star):*

**Application:**

Classify the star relative to others.

**Importance:**

Influences stellar evolution and the formation of the planetary system.

# *'st_met' (Metallicity of the Host Star):*

**Application:**

Investigate the star's chemical composition.

**Importance:**
May be related to the formation and evolution of planetary systems.

# *'st_logg' (Logarithm of the Host Star's Surface Gravity):*

**Application:**

Estimate the star's mass and size.

**Importance:**

Helps determine the star's size and therefore the characteristics of the habitable zone.

# *'sy_dist' (Distance to the Planetary System):*

**Application:**

Assess the proximity of the planet to Earth.

**Importance:**

Affects the reception of solar radiation and influences climate and habitability conditions.

# *'sy_vmag', 'sy_kmag', 'sy_gaiamag' (Apparent Magnitudes of the Planetary System):*

**Application:**

Evaluate the observed brightness of the host star.

**Importance:**

Aids in estimating the system's luminosity and in astronomical observations.

# *'pl_orbeccen' (Orbital Eccentricity of the Planet):*

**Application:**

Analyze the shape of the planet's orbit.

**Importance:**
Can indicate the presence of elliptical orbits and their impact on climate conditions.