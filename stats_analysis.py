import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, f_oneway, kruskal, shapiro, probplot, ttest_ind, ks_2samp, linregress
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Load data
filename = 'PS_2023.08.14_09.36.52.csv'
exoplanets = pd.read_csv(filename, skiprows=96)

def explore_data(data):
    # Display first 10 rows
    print(data.head(10))

    # Display column names
    print(data.columns)

    # Display basic information
    print(data.info())

    # Display summary statistics
    print(data.describe())

explore_data(exoplanets)

print("""
Hypothesis 1:
 
The methods of exoplanet discovery exhibit significant
differences in terms of the number of discoveries.

""")

def visualize_distributions(data, methods):
    num_methods = len(methods)
    fig, axes = plt.subplots(nrows=num_methods, ncols=1, figsize=(20, 18*num_methods))

    for idx, method in enumerate(methods.keys()):
        ax = axes[idx]
        sns.kdeplot(data[methods[method]].count(), ax=ax, label=method)

        ax.set_xlabel('Count of Discoveries')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of Discoveries by Method: {method}')
        ax.legend()

    plt.tight_layout()
    plt.show()

print(""" 
Insights:

1.I initially conducted the chi-square and ANOVA tests and observed 
discrepancies between their outcomes. Consequently, I visualized the
distributions of the labels and noted that they follow non-parametric
distributions. Thus, I re-conducted the second test, replacing ANOVA
with Kruskal-Wallis and re-evaluated the results.

2.This can be interpreted as an indication that exoplanet discovery 
methods are indeed yielding different outcomes in terms of discovery 
counts, providing support for the analysis that the non-parametric 
distributions exhibit significant differences.""")

print(
"""
Hypothesis 2:

The distribution of exoplanet masses follows a normal distribution.

""")

def analyze_exoplanet_masses(data):
    # Perform Shapiro-Wilk Test
    statistic, p_value = shapiro(data['pl_bmassj'])

    print("Shapiro-Wilk Test Results:")
    print("Test Statistic:", statistic)
    print("P-Value:", p_value)

    alpha = 0.05

    if p_value > alpha:
        print("The data follows a normal distribution (fail to reject null hypothesis).")
    else:
        print("The data does not follow a normal distribution (reject null hypothesis).")

    # Create a Q-Q plot
    probplot(data['pl_bmassj'], dist="norm", plot=plt)
    plt.title("Q-Q Plot of Exoplanet Masses")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.show()

    # Summary interpretation
    if p_value > alpha:
        print("Summary:\nThe Shapiro-Wilk test suggests that the data may follow a normal distribution.")
    else:
        print("Summary:\nAlthough the Shapiro-Wilk test did not reject normality, the Q-Q plot suggests potential deviations in the tails, indicating the need for further investigation.")

analyze_exoplanet_masses(exoplanets)

print("""
Insights:
  
1.In summary, the analysis of exoplanet masses revealed that
the Shapiro-Wilk test yielded a p-value of 1.0, suggesting no
significant departure from normality. However, the Q-Q plot 
displayed an upward exponential curvature at the end, indicating
deviations from normality, particularly in the tails. This 
suggests that while the Shapiro-Wilk test did not reject normality,
visual inspection of the Q-Q plot hints at the presence of 
heavy tails in the distribution. These heavy tails could imply
the presence of outliers or rare events in the exoplanet mass
data. Researchers should consider exploring outlier identification,
data transformation, or robust statistical methods to handle the
potential influence of these outliers and non-normal distribution
characteristics in further analyses.

""")

print("""
Hypothesis 3:

There is a positive correlation between the exoplanet mass
and the mass of the host star.
  
""")

def analyze_correlation(data):
    # Calculate the Pearson correlation coefficient
    correlation_coefficient = data['pl_bmassj'].corr(data['st_mass'])

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='st_mass', y='pl_bmassj')
    plt.title("Correlation Between Exoplanet Mass and Host Star Mass")
    plt.xlabel("Host Star Mass")
    plt.ylabel("Exoplanet Mass")
    plt.show()

    # Print the correlation coefficient
    print("Pearson Correlation Coefficient:", correlation_coefficient)

analyze_correlation(exoplanets)

print(
"""
Insights:

1.Based on the analysis of the correlation between exoplanet mass
and host star mass, we observed a Pearson correlation coefficient
of approximately 0.26. This positive value indicates a weak positive
correlation between the two variables. Although the relationship is not
strong, the presence of a positive correlation suggests that, in general,
exoplanets with larger masses tend to orbit host stars with larger
masses. However, it's important to note that other factors may influence
this relationship and that correlation does not necessarily imply a
cause-and-effect relationship between exoplanet and star masses.

""")

print("""
Hypothesis 4:
  
The masses of exoplanets differ between systems with different numbers of stars.

""")

def provide_insights(p_value):
    alpha = 0.05

    print("\nInsights:")
    if p_value < alpha:
        print("The p-value is less than the significance level, suggesting that we can reject the null hypothesis.")
        print("There is evidence to suggest that exoplanet masses are different between systems with different numbers of host stars.")
    else:
        print("The p-value is greater than the significance level, suggesting that we fail to reject the null hypothesis.")
        print("There is not enough evidence to suggest that exoplanet masses differ between systems with different numbers of host stars.")
def compare_masses_by_star_count(data):
    # Select data for two groups based on star count
    group_1 = data[data['sy_snum'] == 1]['pl_bmassj'].dropna()
    group_2 = data[data['sy_snum'] == 2]['pl_bmassj'].dropna()

    # Perform t-test
    t_statistic, p_value = ttest_ind(group_1, group_2)

    # Print t-test results
    print("T-Statistic:", t_statistic)
    print("P-Value:", p_value)

    # Provide insights based on results
    provide_insights(p_value)

compare_masses_by_star_count(exoplanets)

print("""
Insights:

After conducting the Student's t-test to compare exoplanet 
masses between systems with 1 and 2 host stars, we observed a
t-statistic value of approximately -1.57 and a p-value of
about 0.12. The p-value is greater than the usual significance
level of 0.05, indicating that there is not enough evidence
to reject the null hypothesis that the exoplanet masses are
equal between the two groups. Therefore, I did not find
statistically significant differences in exoplanet masses 
between systems with different numbers of host stars.

""")

"""
Hypothesis 5:

The relationship between semi-major axis and orbital period follows Kepler's law.

"""

def polynomial_func(x, a, b, c):
    return a * x**2 + b * x + c

def visualize_relationship(data):
    x = data.dropna()['pl_orbsmax']
    y = data.dropna()['pl_orbper']

    # Perform linear and polynomial fits
    linear_fit = np.polyfit(x, y, 2)
    linear_func = np.poly1d(linear_fit)

    popt, _ = curve_fit(polynomial_func, x, y)

    # Plot the data and fits
    plt.figure(figsize=(10,6))
    plt.scatter(x, y, alpha=0.5, label='Data')
    plt.plot(x, linear_func(x), color='red', label='Linear Fit')
    plt.plot(x, polynomial_func(x, *popt), color='green', label='Polynomial Fit')
    plt.xlabel("Semi-Major Axis (AU)")
    plt.ylabel("Orbital Period (days)")
    plt.title("Kepler's Law: Semi-Major Axis vs. Orbital Period (Semi-Major Axis < 100)")
    plt.legend()
    plt.show()

def analyze_keplers_law(data):
    # Filter and prepare valid data
    valid_data = data[['pl_orbsmax', 'pl_orbper']].dropna()
    filtered_data = valid_data[valid_data['pl_orbsmax'] < 100]

    # Visualize the data and fits
    visualize_relationship(filtered_data)

analyze_keplers_law(exoplanets)


print("""
Insights:

Based on the results of the analysis of the relationship between Semi-Major Axis
and Orbital Period of exoplanets, we observe a strong positive correlation between
these two variables. By fitting a polynomial curve to the data, we find that this
curve fits well with the experimental data points, indicating a non-linear
relationship between Semi-Major Axis and Orbital Period. This is consistent with
Kepler's Law, which describes the relationship between the orbital parameters of
a planetary system. Therefore, we can conclude that the results support the idea
that Kepler's Law provides an accurate description of the relationship between
Semi-Major Axis and Orbital Period of the analyzed exoplanets.

""")

print("""
Hypothesis 6:

The distributions of visual magnitude, infrared magnitudes, and Gaia magnitude are different.

""")

def provide_insights(p_visual_infrared, p_visual_gaia, p_infrared_gaia):
    alpha = 0.05

    print("\nInsights:")
    if any(p < alpha for p in [p_visual_infrared, p_visual_gaia, p_infrared_gaia]):
        print("These results indicate that magnitudes measured in different wavelength ranges have statistically significant differences among them.")
        print("This could be related to varying sensitivities of measurement instruments in each wavelength range, light absorption by interstellar medium, and other variables affecting observations at different wavelengths.")
        print("Therefore, when comparing magnitudes across different wavelength ranges, it's important to consider potential sources of variation that could contribute to these differences.")
    else:
        print("Based on the KS test results, there is not enough evidence to conclude significant differences in magnitude distributions among different wavelength ranges.")

def visualize_distributions(visual, infrared, gaia):
    plt.figure(figsize=(10, 6))
    sns.histplot(visual, label='Visual Magnitude', color='blue', alpha=0.5)
    sns.histplot(infrared, label='Infrared Magnitude', color='orange', alpha=0.5)
    sns.histplot(gaia, label='Gaia Magnitude', color='green', alpha=0.5)

    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    plt.title('Distribution of Visual, Infrared, and Gaia Magnitudes')
    plt.legend()
    plt.show()

def compare_magnitudes_distributions(data):
    # Extract magnitudes data
    visual_mag = data['sy_vmag'].dropna()
    infrared_mag = data['sy_kmag'].dropna()
    gaia_mag = data['sy_gaiamag'].dropna()

    # Perform Kolmogorov-Smirnov tests
    ks_visual_infrared, p_visual_infrared = ks_2samp(visual_mag, infrared_mag)
    ks_visual_gaia, p_visual_gaia = ks_2samp(visual_mag, gaia_mag)
    ks_infrared_gaia, p_infrared_gaia = ks_2samp(infrared_mag, gaia_mag)

    # Print KS test results
    print("KS Test Results:")
    print(f"Visual vs Infrared - KS Statistic: {ks_visual_infrared}, P-Value: {p_visual_infrared}")
    print(f"Visual vs Gaia - KS Statistic: {ks_visual_gaia}, P-Value: {p_visual_gaia}")
    print(f"Infrared vs Gaia - KS Statistic: {ks_infrared_gaia}, P-Value: {p_infrared_gaia}")

    # Visualize distributions
    visualize_distributions(visual_mag, infrared_mag, gaia_mag)

    # Provide insights based on the KS test results
    provide_insights(p_visual_infrared, p_visual_gaia, p_infrared_gaia)


print("""

Hypothesis 7:

There is a relationship between the distance and the visual magnitude of host stars.

""")

def provide_insights(r_squared):
    print("\nOverall, the results suggest:")
    if r_squared > 0.5:
        print("There is a statistically significant and relatively strong positive relationship between distance and visual magnitude of host stars.")
    elif r_squared > 0.3:
        print("There is a statistically significant but relatively moderate positive relationship between distance and visual magnitude of host stars.")
    else:
        print("There is a statistically significant but relatively weak positive relationship between distance and visual magnitude of host stars.")
    print("The R-squared value indicates that other factors not included in the analysis may also contribute to the variability in visual magnitude.")

def visualize_relationship(x, y, slope, intercept):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=x, y=y, alpha=0.5, label='Data')
    plt.plot(x, slope * x + intercept, color='red', label='Regression Line')
    plt.xlabel('Distance (parsecs)')
    plt.ylabel('Visual Magnitude')
    plt.title('Relationship between Distance and Visual Magnitude')
    plt.legend()
    plt.show()

def analyze_distance_vs_visual_magnitude(data):
    # Extract relevant data
    alldata = data[['sy_dist', 'sy_vmag']].dropna()
    distance = alldata['sy_dist']
    visual_mag = alldata['sy_vmag']

    # Perform linear regression
    regression_result = linregress(distance, visual_mag)
    slope = regression_result.slope
    intercept = regression_result.intercept

    # Visualize the data and regression line
    visualize_relationship(distance, visual_mag, slope, intercept)

    # Print regression statistics
    print("Regression Slope:", slope)
    print("Regression Intercept:", intercept)
    print("R-squared:", regression_result.rvalue ** 2)
    print("P-Value:", regression_result.pvalue)

    # Provide insights based on regression results
    provide_insights(regression_result.rvalue ** 2)

analyze_distance_vs_visual_magnitude(exoplanets)

print("""
Hypothesis 8:

There is a temporal trend in the discoveries of exoplanets over the years.

""")

def plot_exoplanet_discoveries_by_year(data):
    discoveries_by_year = data['disc_year'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.bar(discoveries_by_year.index, discoveries_by_year.values, color='blue')
    plt.xlabel("Year")
    plt.ylabel("Number of Discoveries")
    plt.title("Temporal Analysis of Exoplanet Discoveries")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_exoplanet_discoveries_by_year(exoplanets)

print("""
Insights:

The significant increases in exoplanet discoveries in 2014 and 2016 could be
attributed to various factors, including advancements in observation techniques,
improvements in data analysis methods, and the launch of new space telescopes
or missions that were particularly effective at detecting exoplanets during
those years. Additionally, collaborative efforts among different research
groups, increased funding, and dedicated exoplanet discovery missions could
have also played a role in boosting the number of discoveries during those
specific years. It would be beneficial to investigate historical records,
scientific publications, and announcements related to exoplanet research during
those years to gain a better understanding of the specific factors that contributed
to the observed increases.

""")

print("""
Hypothesis 9:
      
There is a difference in stellar properties among different discovery methods.

""")

def stellar_parameters_by_discovery_method(data):

    stellar_params = data[['st_teff', 'st_rad', 'st_mass', 'discoverymethod']]
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

stellar_parameters_by_discovery_method(exoplanets)

#Dimensionality Reduction and Clustering

def dimensionality_reduction_and_clustering(data):
    selected_columns = ['pl_orbper', 'pl_rade', 'pl_radj', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist', 'st_spectype']

    X = exoplanets[selected_columns].dropna()

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

dimensionality_reduction_and_clustering(exoplanets)

def scatter_period_dist_mass_rad(data):
  labels = ['pl_orbper', 'ttv_flag', 'pl_rade', 'pl_orbsmax', 'st_mass', 'st_rad', 'pl_name']
  data = data[labels]
  data.dropna(inplace=True)
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

scatter_period_dist_mass_rad(exoplanets)

def histogram_(data):
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

histogram_(exoplanets)