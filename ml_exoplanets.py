import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from kerastuner.tuners import RandomSearch
from sklearn.metrics import mean_squared_error
from lime.lime_tabular import LimeTabularExplainer
from scipy.optimize import curve_fit
import shap

# Load Data
filename = 'PS_2023.08.14_09.36.52.csv'
exoplanets = pd.read_csv(filename, skiprows=96)

# Select specific columns
selected_columns = [
    'sy_snum', 'sy_pnum', 'pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_bmasse',
    'pl_bmassj', 'pl_orbeccen', 'pl_insol', 'st_teff', 'st_rad', 'st_mass',
    'st_met', 'st_logg', 'ra', 'dec', 'sy_dist', 'sy_vmag', 'sy_kmag', 'sy_gaiamag'
]
data = exoplanets[selected_columns].dropna()

# Visualize correlation
correlation_matrix = data.corr()
plt.figure(figsize=(30, 20))
sns.set(font_scale=1.2)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matrix Correlation Diagram')
plt.show()

# Prepare input and output data
input_data = data[['pl_orbsmax', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_met', 'dec']].values
output_data = data['pl_orbper'].values

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

# Analyzing Prediction Errors
predicted_values = final_model.predict(normalized_input_data)
errors = predicted_values - output_data

plt.scatter(output_data, errors[0])
plt.xlabel('Real Value')
plt.ylabel('Error')
plt.title('Prediction Errors')
plt.show()

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

explainer = shap.Explainer(model, normalized_input_data)
shap_values = explainer(normalized_input_data)

shap.summary_plot(shap_values, normalized_input_data)

# SHAP (SHapley Additive exPlanations) Visualization
explainer = shap.Explainer(final_model, normalized_input_data)
shap_values = explainer(normalized_input_data)
shap.summary_plot(shap_values, normalized_input_data)