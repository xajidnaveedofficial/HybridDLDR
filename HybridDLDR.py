
import pandas as pd
df_data = pd.read_csv(data_URL)

df_data.head()

#pip install rdkit-pypi shap tensorflow

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data_path = data_URL
data = pd.read_csv(data_path)

# Extract SMILES strings and gene expression data
smiles = data['SMILES']
gene_expression = data.drop(columns=['SAMPLE_ID', 'TCGA_DESC', 'DRUG_NAME', 'DRUG_ID', 'SMILES', 'IC50'])
ic50 = data['IC50']

data = pd.DataFrame({
    'smiles': smiles,
    'IC50': ic50
})

!pip install keras-tuner

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, BatchNormalization, Conv1D, MaxPooling1D, Flatten, LSTM, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # Import ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

# Load data (Example arrays, replace with actual data)
data = pd.DataFrame({
    'smiles': smiles,
    'IC50': ic50
})

# Ensure that the lengths of all lists are the same
assert len(smiles) == len(gene_expression) == len(ic50), "All arrays must be of the same length"

# Preprocessing
def preprocess_gene_expression(gene_expression):
    gene_expression = np.array(gene_expression)
    scaler = StandardScaler()
    gene_expression = scaler.fit_transform(gene_expression)
    return gene_expression

# Convert SMILES to Morgan fingerprints
def smiles_to_morgan_fp(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    else:
        return np.zeros((n_bits,))

# Process data
X_smiles = np.array([smiles_to_morgan_fp(s) for s in smiles])
X_gene = preprocess_gene_expression(gene_expression)
X = np.hstack((X_smiles, X_gene))
y = np.array(ic50)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define custom Transformer encoder layer
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.05):  # Increased dropout for regularization
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="LeakyReLU")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

# Define the input for molecular descriptors (CNN model)
morgan_input = Input(shape=(1024, 1))
x1 = Conv1D(16, kernel_size=3, activation='LeakyReLU')(morgan_input)
x1 = MaxPooling1D(pool_size=2)(x1)
x1 = Flatten()(x1)
x1 = Dense(32, activation='LeakyReLU')(x1)
x1 = Dropout(0.05)(x1)  # Increased dropout for regularization

# Define the input for gene expression data (LSTM model)
gene_input = Input(shape=(X_train.shape[1] - 1024, 1))
x2 = LSTM(16, return_sequences=True)(gene_input)
x2 = LSTM(16)(x2)
x2 = Dense(32, activation='LeakyReLU')(x2)
x2 = Dropout(0.05)(x2)  # Increased dropout for regularization

# Define the input for gene expression data (Transformer model)
gene_input_transformer = Input(shape=(X_train.shape[1] - 1024, 1))
x3 = transformer_encoder(gene_input_transformer, head_size=32, num_heads=4, ff_dim=32, dropout=0.05)  # Increased dropout
x3 = GlobalAveragePooling1D()(x3)
x3 = Dense(32, activation='LeakyReLU')(x3)
x3 = Dropout(0.05)(x3)  # Increased dropout for regularization

# Concatenate the outputs of the three networks
concatenated = Concatenate()([x1, x2, x3])
x = Dense(32, activation='LeakyReLU')(concatenated)
x = Dropout(0.05)(x)  # Increased dropout for regularization
output = Dense(1)(x)

# Define the model
model = Model(inputs=[morgan_input, gene_input, gene_input_transformer], outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)  # Allow lower minimum learning rate
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

# Reshape data for CNN, LSTM, and Transformer inputs
X_train_smiles = X_train[:, :1024].reshape(-1, 1024, 1)
X_train_gene = X_train[:, 1024:].reshape(-1, X_train.shape[1] - 1024, 1)
X_test_smiles = X_test[:, :1024].reshape(-1, 1024, 1)
X_test_gene = X_test[:, 1024:].reshape(-1, X_test.shape[1] - 1024, 1)

# Train the model
history = model.fit([X_train_smiles, X_train_gene, X_train_gene], y_train, epochs=100, batch_size=2,  # Increased epochs, decreased batch size
                    validation_split=0.2, callbacks=[early_stopping, reduce_lr, checkpoint], verbose=1)

# Load the best model
model.load_weights('best_model.h5')

# Evaluate the model
y_pred = model.predict([X_test_smiles, X_test_gene, X_test_gene])
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse}")
print(f"Test MAE: {mae}")
print(f"Test R²: {r2}")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Attention, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import LeakyReLU
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
# Evaluate the model
model.evaluate([X_test_smiles, X_test_gene, X_test_gene], y_test, verbose=1)

# Make predictions on the test set
y_pred_ic50 = model.predict([X_test_smiles, X_test_gene, X_test_gene])

# Calculate additional metrics
mse = mean_squared_error(y_test, y_pred_ic50)
mae = mean_absolute_error(y_test, y_pred_ic50)
r2 = r2_score(y_test, y_pred_ic50)
correlation, _ = pearsonr(y_test, y_pred_ic50.flatten())

print(f"Test IC50 Prediction MSE: {mse:.4f}")
print(f"Test IC50 Prediction MAE: {mae:.4f}")
print(f"Test IC50 Prediction R^2: {r2:.4f}")
print(f"Test IC50 Prediction Correlation: {correlation:.4f}")

# Create a DataFrame for IC50 prediction results
ic50_results = pd.DataFrame({
    'Real IC50': y_test,
    'Predicted IC50': y_pred_ic50.flatten()
})

print(ic50_results)

# Scatter plot of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_ic50, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual IC50')
plt.ylabel('Predicted IC50')
plt.title('Actual vs Predicted IC50')
plt.show()

# Line plot of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual IC50')
plt.plot(y_pred_ic50, label='Predicted IC50', alpha=0.7)
plt.xlabel('Samples')
plt.ylabel('IC50')
plt.title('True vs Predicted IC50')
plt.legend()
plt.show()

# Evaluate the model on the test set
test_results = model.evaluate([X_test_smiles, X_test_gene, X_test_gene], y_test, verbose=1)

print(f"Test IC50 Prediction MSE: {test_results[0]:.4f}")
print(f"Test IC50 Prediction MAE: {test_results[1]:.4f}")

# Make predictions on the test set
y_pred_ic50 = model.predict([X_test_smiles, X_test_gene, X_test_gene])

# Calculate additional metrics
mse = mean_squared_error(y_test, y_pred_ic50)
mae = mean_absolute_error(y_test, y_pred_ic50)
r2 = r2_score(y_test, y_pred_ic50)
correlation, _ = pearsonr(y_test, y_pred_ic50.flatten())

print(f"Test IC50 Prediction R^2: {r2:.4f}")
print(f"Test IC50 Prediction Correlation: {correlation:.4f}")

# Create a DataFrame for IC50 prediction results
ic50_results = pd.DataFrame({
    'Real IC50': y_test,
    'Predicted IC50': y_pred_ic50.flatten()
})

# Display the first few rows of the IC50 prediction results
print("IC50 Prediction Results:")
print(ic50_results.head())

# Save the results to a CSV file if needed
ic50_results.to_csv('ic50_prediction_results.csv', index=False)

# Plot predicted vs actual IC50 values
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_ic50, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual IC50')
plt.ylabel('Predicted IC50')
plt.title('Predicted vs Actual IC50')
plt.show()

# Plot true vs predicted values for IC50
plt.figure(figsize=(8, 8))
plt.plot(y_test, label='True Values')
plt.plot(y_pred_ic50, label='Predicted Values')
plt.xlabel('Samples')
plt.ylabel('IC50')
plt.title('True vs Predicted IC50 Values')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Evaluation metrics
metrics = ['Test MSE', 'Test MAE', 'Test R²']
values = [0.9464386666024427, 0.7553990537821689, 0.8697512592331484]

# Creating a bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, values, color=['blue', 'green', 'red'])

# Adding the value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom')

# Setting title and labels
plt.title('Model Evaluation Metrics on Test Data')
plt.xlabel('Metrics')
plt.ylabel('Values')

# Display the plot
plt.ylim(0, 1.2)  # Setting the y-axis limit for better visualization
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Evaluation metrics for your model
your_model_metrics = [0.9525443059039462, 0.7599131435152942, 0.8689110021106745]

# Evaluation metrics for existing models
existing_model_1_metrics = [1.2, 0.85, 0.80]
existing_model_2_metrics = [1.1, 0.80, 0.82]

# Labels for the metrics
metrics = ['Test MSE', 'Test MAE', 'Test R²']

# X axis positions for each group of bars
x = np.arange(len(metrics))

# Width of the bars
width = 0.25

# Creating the bar chart
fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width, your_model_metrics, width, label='Your Model')
bars2 = ax.bar(x, existing_model_1_metrics, width, label='Existing Model 1')
bars3 = ax.bar(x + width, existing_model_2_metrics, width, label='Existing Model 2')

# Adding value labels on top of the bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom')

# Setting title and labels
ax.set_title('Comparison of Model Evaluation Metrics on Test Data')
ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Display the plot
plt.ylim(0, 1.4)  # Setting the y-axis limit for better visualization
plt.show()

import matplotlib.pyplot as plt

# True and predicted IC50 values
true_ic50 = [1.957118, 1.527675, 5.787082, 5.441075, -3.653334]
predicted_ic50 = [2.855036, 3.144621, 5.951722, 4.794811, -1.709804]

# Creating the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(true_ic50, predicted_ic50, color='blue', label='Predicted IC50')

# Plotting a line for perfect prediction
plt.plot([-4, 6], [-4, 6], color='red', linestyle='--', label='Perfect Prediction')

# Setting title and labels
plt.title('True IC50 vs Predicted IC50')
plt.xlabel('True IC50')
plt.ylabel('Predicted IC50')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# True and predicted IC50 values
true_ic50 = [1.957118, 1.527675, 5.787082, 5.441075, -3.653334]
predicted_ic50 = [2.390063, 2.484954, 5.888204, 4.861550, -1.714458]

# Indices for the x-axis
indices = np.arange(len(true_ic50))

# Creating the line plot
plt.figure(figsize=(10, 6))
plt.plot(indices, true_ic50, marker='o', linestyle='-', color='blue', label='True IC50')
plt.plot(indices, predicted_ic50, marker='x', linestyle='--', color='green', label='Predicted IC50')

# Setting title and labels
plt.title('True IC50 vs Predicted IC50')
plt.xlabel('Index')
plt.ylabel('IC50 Value')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()

# Evaluate the model
y_pred = model.predict([X_test_smiles, X_test_gene, X_test_gene])
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# Plot training & validation MAE values
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Scatter plot of real vs predicted IC50 values
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Real IC50')
plt.ylabel('Predicted IC50')
plt.title('Real vs Predicted IC50')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.show()

# Residuals plot
residuals = y_test - y_pred.flatten()
plt.figure(figsize=(8, 8))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Predicted IC50')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.axhline(0, color='r', linestyle='--')
plt.show()

# Convert predictions and true values to a DataFrame for easy comparison
results = pd.DataFrame({'True IC50': y_test, 'Predicted IC50': y_pred.flatten()})
print(results.head())