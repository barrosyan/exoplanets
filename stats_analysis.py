import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, f_oneway, kruskal, shapiro, probplot, ttest_ind, ks_2samp, linregress
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from sklearn.utils import shuffle
from kerastuner.tuners import RandomSearch
from sklearn.metrics import mean_squared_error
from lime.lime_tabular import LimeTabularExplainer
import shap

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

print('--------------------------------')
print('Data Analysis Section')

print("""
Hypothesis 1:
 
The methods of exoplanet discovery exhibit significant
differences in terms of the number of discoveries.
""")

def visualize_distributions(data, methods):
    num_methods = len(methods)
    fig, axes = plt.subplots(nrows=num_methods, ncols=1, figsize=(10, 3*num_methods))

    for idx, method in enumerate(methods.keys()):
        ax = axes[idx]
        sns.kdeplot(data[methods[method]].count(), ax=ax, label=method)

        ax.set_xlabel('Count of Discoveries')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of Discoveries by Method: {method}')
        ax.legend()

    plt.tight_layout()
    plt.show()

def analyze_discovery_methods(data):
    # Create a contingency table for discovery methods
    contingency_table = pd.crosstab(data['discoverymethod'], columns='count')

    # Perform Chi-Square Test
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
    print("Chi-Square Test Results:")
    print(f"Chi-Square Statistic: {chi2_stat}")
    print(f"P-Value: {p_val}")

    # Create boolean masks for each discovery method
    methods = {
        'Radial Velocity': data['discoverymethod'] == 'Radial Velocity',
        'Imaging': data['discoverymethod'] == 'Imaging',
        'Eclipse Timing Variations': data['discoverymethod'] == 'Eclipse Timing Variations',
        'Transit': data['discoverymethod'] == 'Transit',
        'Astrometry': data['discoverymethod'] == 'Astrometry',
        'Disk Kinematics': data['discoverymethod'] == 'Disk Kinematics',
        'Microlensing': data['discoverymethod'] == 'Microlensing',
        'Orbital Brightness Modulation': data['discoverymethod'] == 'Orbital Brightness Modulation',
        'Pulsation Timing Variations': data['discoverymethod'] == 'Pulsation Timing Variations',
        'Transit Timing Variations': data['discoverymethod'] == 'Transit Timing Variations',
        'Pulsar Timing': data['discoverymethod'] == 'Pulsar Timing'
    }

    # Perform ANOVA
    anova_results = f_oneway(*[data[mask].count() for mask in methods.values()])
    print("\nANOVA Test Results:")
    print(f"F-Statistic: {anova_results.statistic}")
    print(f"P-Value: {anova_results.pvalue}")

    visualize_distributions(data, methods)

analyze_discovery_methods(exoplanets)

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
distributions exhibit significant differences.
""")

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

print('-------------------------------------------------')
print('Machine Learning Section')

selected_columns = [
    'sy_snum', 'sy_pnum', 'pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_bmasse',
    'pl_bmassj', 'pl_orbeccen', 'pl_insol', 'st_teff', 'st_rad', 'st_mass',
    'st_met', 'st_logg', 'ra', 'dec', 'sy_dist', 'sy_vmag', 'sy_kmag', 'sy_gaiamag'
]
data = exoplanets[selected_columns].dropna()

def corr_vis(data):
    # Visualize correlation
    correlation_matrix = data.corr()
    plt.figure(figsize=(30, 20))
    sns.set(font_scale=1.2)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matrix Correlation Diagram')
    plt.show()

corr_vis(data)

input_data = data[['pl_orbsmax', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_met', 'dec']].values
output_data = data['pl_orbper'].values

def build_model(input_data, output_data):
    # Standardize input data
    scaler = StandardScaler()
    normalized_input_data = scaler.fit_transform(input_data)

    # Build NN architecture
    n_input = input_data.shape[1]
    n_output = 1
    n_hidden = [10, 10]
    activation = 'relu'
    initializer = 'he_normal'

    input_layer = tf.keras.layers.Input(shape=(n_input,))
    hidden_layer = input_layer
    for units in n_hidden:
        hidden_layer = tf.keras.layers.Dense(units, activation=activation, kernel_initializer=initializer)(hidden_layer)
    output_layer = tf.keras.layers.Dense(n_output, kernel_initializer=initializer)(hidden_layer)
    neural_net = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    predicted_orbper = neural_net(input_layer)

    # Train model
    model = tf.keras.models.Model(inputs=input_layer, outputs=predicted_orbper)
    model.compile(optimizer='adam', loss='mse')
    model.fit(normalized_input_data, output_data, epochs=20000, verbose=1)
    
    # Visualize loss function
    plt.plot(model.history.history['loss'])
    plt.title('Loss Function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Searching for Best Hyperparameter
    def build_model(hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(n_input,)))

        # Tune the number of hidden layers and units per layer
        for i in range(hp.Int('num_hidden_layers', min_value=1, max_value=5)):
            model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), min_value=10, max_value=100, step=10),
                                           activation='relu'))

        model.add(tf.keras.layers.Dense(n_output, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,
        directory='tuner_results',
        project_name='exoplanets')

    tuner.search(normalized_input_data, output_data, epochs=100, validation_split=0.2)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Hyperparameters:", best_hps.values)

    final_model = tuner.hypermodel.build(best_hps)
    final_model.fit(normalized_input_data, output_data, epochs=100, validation_split=0.2)

    # Checking Loss Function for new Model
    plt.plot(final_model.history.history['loss'])
    plt.title('Loss Function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Visualizing Best Model Architecture
    final_model.summary()

    return final_model, normalized_input_data

final_model, normalized_input_data = build_model(input_data, output_data)

def vis_pred(final_model, normalized_input_data):
    # Visualize correlation between real and predicted values
    predicted_values = final_model.predict(normalized_input_data)
    plt.scatter(output_data, predicted_values)
    plt.xlabel('Real')
    plt.ylabel('Predicted')
    plt.title('Real Values vs. Predicted Values')
    plt.show()

    # Visualize predictions
    plt.plot(output_data, label='Real')
    plt.plot(predicted_values, label='Predicted')
    plt.title('Real Values vs. Predicted Values')
    plt.legend()
    plt.show()

# Analyze sensitivity
labels = ['Semimajor Axis', 'Planet Radius', 'Planet Mass', 'Stellar Effective Temperature', 'Stellar Metallicity', 'Declination']
example_idx = 0
example_input = normalized_input_data[example_idx]

def analyze_sensitivity(model, input_data, example_idx, labels):
    example_input = input_data[example_idx]
    for i, label in enumerate(labels):
        feature_idx = i
        vary_range = np.linspace(0.8 * example_input[feature_idx], 1.2 * example_input[feature_idx], num=50)
        variable = label
        predictions = []

        for value in vary_range:
            modified_input = example_input.copy()
            modified_input[feature_idx] = value
            prediction = model.predict(np.array([modified_input]))
            predictions.append(prediction[0][0])

        plt.plot(vary_range, predictions)
        plt.xlabel(f'Variation on {variable}')
        plt.ylabel('Orbit Prediction')
        plt.title('Sensitivity of Variable on Prediction')
        plt.show()

analyze_sensitivity(final_model, normalized_input_data, example_idx, labels)

def permutation_imp(model):
    # Permutation Importance
    n_permutations = 30
    importances = []

    for _ in range(n_permutations):
        shuffled_output = shuffle(output_data, random_state=0)
        model.fit(normalized_input_data, shuffled_output, epochs=100, verbose=0)
        shuffled_predictions = model.predict(normalized_input_data)
        importance = mean_squared_error(output_data, shuffled_predictions)
        importances.append(importance)

    average_importance = np.mean(importances)
    print("Mean Influence of Variables:")
    print(average_importance)
    if average_importance > 10000.:
        print("High Significancy on those Variables")
    else:
        print("Low Significancy on those Variables")

permutation_imp(final_model)

def explainer_ind(normalized_input_data, model):
    # LimeTabularExplainer for Individual Instances
    explainer = LimeTabularExplainer(normalized_input_data, mode="regression")

    num_entries = 10
    feature_values = {}

    for example_idx in range(num_entries):
        explanation = explainer.explain_instance(normalized_input_data[example_idx], model.predict)
        feature_values[example_idx] = explanation.local_exp[1]  # Obtém os pesos das características

    print("feature_values =", feature_values)

    feature_mapping = {
        0: "pl_orbsmax",
        1: "pl_rade",
        2: "pl_bmasse",
        3: "st_teff",
        4: "st_met",
        5: "dec"
    }

    physical_mapping = {
        "pl_orbsmax": "Semimajor Axis",
        "pl_rade": "Planet Radius",
        "pl_bmasse": "Planet Mass",
        "st_teff": "Stellar Effective Temperature",
        "st_met": "Stellar Metallicity",
        "dec": "Declination"
    }

    for example_idx, values in feature_values.items():
        print(f"Entry {example_idx} Analysis:")
    
        for idx, value in values:
            feature_name = feature_mapping[idx]
            physical_name = physical_mapping[feature_name]
        
            influence = "Positive" if value > 0 else "Negative"
        
            print(f"{physical_name}: {influence} Impact: {value}")
    
    print("------------------------------------")

explainer_ind(normalized_input_data, final_model):

def analyze_errors(normalized_input_data=normalized_input_data, final_model=final_model):
    # Analyzing Prediction Errors
    predicted_values = final_model.predict(normalized_input_data)
    errors = predicted_values - output_data

    plt.scatter(output_data, errors[0])
    plt.xlabel('Real Value')
    plt.ylabel('Error')
    plt.title('Prediction Errors')
    plt.show()

analyze_errors()

def checking_relevance(input_data=normalized_input_data, output_data=output_data):
    # Correlation Matrix Heatmap
    correlation_matrix = np.corrcoef(input_data.T, output_data)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, xticklabels=['pl_orbsmax', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_met', 'dec', 'pl_orbper'], yticklabels=['pl_orbsmax', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_met', 'dec', 'pl_orbper'])
    plt.title('Correlation Matrix Heatmap')
    plt.show()

    # Feature Influences for Entry 0
    feature_names = [
        'Stellar Metallicity', 
        'Planet Mass', 
        'Stellar Effective Temperature', 
        'Declination', 
        'Planet Radius', 
        'Semimajor Axis'
        ]
    influences_entry_0 = [
        -41.35655194605194, 
        -37.049456362808904, 
        35.86883501867438, 
        35.454882955404784, 
        -11.29755847376939, 
        0.08430690613074271
        ]

    plt.barh(feature_names, influences_entry_0, color=['g' if i > 0 else 'r' for i in influences_entry_0])
    plt.xlabel('Impact Value')
    plt.ylabel('Features')
    plt.title('Feature Influences for Entry 0')
    plt.show()



def curve_fitting(normalized_input_data=normalized_input_data):
    # Curve Fitting - Exponential Function
    def exponential_function(x, a, b):
        return a * np.exp(b * x)

    params, _ = curve_fit(exponential_function, normalized_input_data[:, 0], output_data)

    plt.scatter(normalized_input_data[:, 0], output_data, label='Real Data')
    plt.plot(normalized_input_data[:, 0], exponential_function(normalized_input_data[:, 0], *params), label='Exponential Adjust', color='red')
    plt.xlabel('Input Feature')
    plt.ylabel('Output Value')
    plt.title('Exponential Function Adjusted')
    plt.legend()
    plt.show()

    print("Adjusted Equation Coefficients:", params)

curve_fitting()

def new_model(final_model=final_model):
    # New Model with Modified Architecture
    hidden_layer = tf.keras.layers.Dense(10, activation='relu')(final_model.input)
    output_layer = tf.keras.layers.Dense(1)(hidden_layer)
    new_model = tf.keras.models.Model(inputs=final_model.input, outputs=output_layer)
    new_model.compile(optimizer='adam', loss='mse')

    intermediate_layer_model = tf.keras.models.Model(inputs=new_model.input, outputs=hidden_layer)
    intermediate_output = intermediate_layer_model.predict(input_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(intermediate_output[:10], cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel("Hidden Layer Units")
    plt.ylabel("Input Samples")
    plt.title("Intermediate Layer Activations")
    plt.show()

    # Heatmap of Intermediate Activations
    plt.figure(figsize=(10, 6))
    sns.heatmap(intermediate_output, cmap='viridis', cbar=True)
    plt.xlabel("Hidden Layer Units")
    plt.ylabel("Input Samples")
    plt.title("Heatmap of Intermediate Activations")
    plt.show()

    # Combined Data and Correlation Heatmap
    combined_data = np.hstack((normalized_input_data, intermediate_output))
    all_feature_names = feature_names + ['Activation_' + str(i) for i in range(intermediate_output.shape[1])]
    combined_df = pd.DataFrame(data=combined_data, columns=all_feature_names)
    correlations = combined_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap between Input Variables and Intermediate Activations")
    plt.show()

def explainingai(model=final_model)

    explainer = shap.Explainer(model, normalized_input_data)
    shap_values = explainer(normalized_input_data)

    shap.summary_plot(shap_values, normalized_input_data)

    # SHAP (SHapley Additive exPlanations) Visualization
    explainer = shap.Explainer(final_model, normalized_input_data)
    shap_values = explainer(normalized_input_data)
    shap.summary_plot(shap_values, normalized_input_data)

    checking_relevance()