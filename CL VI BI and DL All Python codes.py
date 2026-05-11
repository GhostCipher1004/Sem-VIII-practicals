# BI 1 Import Data from different Sources such as (Excel, Sql Server, Oracle etc.) and load in targeted system.

import pandas as pd
import sqlite3

# --- Source 1: Excel ---
excel_data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Salary': [50000, 60000, 70000, 55000]
})
excel_data.to_excel('sample_data.xlsx', index=False)
df_excel = pd.read_excel('sample_data.xlsx')

# --- Source 2: CSV ---
excel_data.to_csv('sample_data.csv', index=False)
df_csv = pd.read_csv('sample_data.csv')

# --- Source 3: JSON ---
excel_data.to_json('sample_data.json', orient='records')
df_json = pd.read_json('sample_data.json')

# --- Source 4: MySQL ---
# conn = pymssql.connect(
#     server='localhost',
#     user='root',
#     password='root',
#     database='test'
# )
# cursor = conn.cursor()

# query = "SELECT * FROM users"
# cursor.execute(query)
# df_sql = pd.read_sql(query, conn)
# conn.close()

print(f"""
------ EXCEL -----
{df_excel}

------ CSV -----
{df_csv}

------ JSON -----
{df_json}
""")

# ------ SQL -----
# {df_sql}
# """)

# --- Load into Target System (SQLite Database) ---
conn = sqlite3.connect('target_database.db')
df_excel.to_sql('employees', conn, if_exists='replace', index=False)
print("\nData loaded into SQLite database successfully!")

# Verify loaded data
result = pd.read_sql('SELECT * FROM employees', conn)
print("\nData in Target DB:\n", result)
conn.close()

# BI 2 Data Visualization from Extraction Transformation and Loading (ETL) Process

import pandas as pd
import matplotlib.pyplot as plt

# --- Extract ---
data = pd.DataFrame({
    'Department': ['IT', 'HR', 'Sales', 'Marketing', 'Finance'],
    'Employees': [50, 30, 40, 25, 35],
    'Revenue': [500000, 200000, 800000, 350000, 600000],
    'Satisfaction': [4.2, 3.8, 4.0, 4.5, 3.9]
})
print("Extracted Data:\n", data)

# --- Transform ---
data['Revenue_Per_Employee'] = data['Revenue'] / data['Employees']
print("\nTransformed Data:\n", data)

# --- Load (Save to CSV) ---
data.to_csv('etl_output.csv', index=False)
print("\nData saved to etl_output.csv")

# --- Visualize ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('ETL Data Visualization', fontsize=16)

# Bar chart - Revenue by Department
axes[0, 0].bar(data['Department'], data['Revenue'], color='steelblue')
axes[0, 0].set_title('Revenue by Department')
axes[0, 0].set_ylabel('Revenue')

# Pie chart - Employee Distribution
axes[0, 1].pie(data['Employees'], labels=data['Department'], autopct='%1.1f%%')
axes[0, 1].set_title('Employee Distribution')

# Line chart - Satisfaction Score
axes[1, 0].plot(data['Department'], data['Satisfaction'], marker='o', color='green')
axes[1, 0].set_title('Satisfaction Score')
axes[1, 0].set_ylabel('Score')

# Bar chart - Revenue Per Employee
axes[1, 1].barh(data['Department'], data['Revenue_Per_Employee'], color='coral')
axes[1, 1].set_title('Revenue Per Employee')

plt.tight_layout()
plt.savefig('etl_visualization.png')
plt.show()
print("Visualization saved as etl_visualization.png")

# BI 3 Perform the Extraction Transformation and Loading (ETL) process to construct the database in the Sql server / Power BI.

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# --- Extract ---
data = pd.DataFrame({
    'Department': ['IT', 'HR', 'Sales', 'Marketing', 'Finance'],
    'Employees': [50, 30, 40, 25, 35],
    'Revenue': [500000, 200000, 800000, 350000, 600000],
    'Satisfaction': [4.2, 3.8, 4.0, 4.5, 3.9]
})

print("Extracted Data:\n", data)

# --- Transform ---
data['Revenue_Per_Employee'] = data['Revenue'] / data['Employees']

print("\nTransformed Data:\n", data)

# --- Load into Database ---
conn = sqlite3.connect('company.db')

data.to_sql(
    'department_data',
    conn,
    if_exists='replace',
    index=False
)

print("\nData loaded into database successfully!")

# Verify database data
result = pd.read_sql(
    "SELECT * FROM department_data",
    conn
)

print("\nData in Database:\n", result)

conn.close()

# --- Visualize ---
plt.bar(data['Department'], data['Revenue'])

plt.title("Revenue by Department")
plt.xlabel("Department")
plt.ylabel("Revenue")

plt.show()

# BI 4 Data Analysis and Visualization using Advanced Excel.
# TO BE DONE IN EXCEL


# BI 5 Perform the data classification algorithm using any Classification algorithm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

# Train Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and Evaluate
y_pred = clf.predict(X_test)
print("Decision Tree Classification Results")
print("=" * 40)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=data.target_names)}")

#######DL#######
# DL 1 Predict house prices in the USA using Linear Regression.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load USA Housing dataset from GitHub
url = "https://raw.githubusercontent.com/huzaifsayed/Linear-Regression-Model-for-House-Price-Prediction/master/USA_Housing.csv"
df = pd.read_csv(url)

# Select numerical features used to predict price
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Area Population']]
y = df['Price']  # Target: house price

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set and evaluate with R² score
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# DL 2 Multiclass CNN classifier on MNIST with confusion matrix.

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Reshape and normalize images
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Convert labels to categorical format
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Build CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train_cat,
          epochs=5,
          batch_size=64,
          validation_split=0.1)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"\nTest Accuracy: {acc:.4f}")

# Predict classes
y_pred = np.argmax(model.predict(X_test), axis=1)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=range(10))

disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# DL 3 LSTM on IMDB reviews to predict positive/negative sentiment.

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Load IMDB dataset (already tokenized)
max_features = 10000
maxlen = 200
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)

# Pad sequences
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

# Build LSTM model
model = keras.Sequential([
    layers.Embedding(max_features, 64, input_length=maxlen),
    layers.LSTM(64, dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

# Predict on sample
sample = X_test[:5]
preds = model.predict(sample)
for i, p in enumerate(preds):
    print(f"Review {i + 1}: {'Positive' if p > 0.5 else 'Negative'} (score: {p[0]:.4f})")

# DL 4 CNN for Image Classification on CIFAR-10 with hyperparameter tuning.

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize images
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_split=0.1)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# Plot accuracy graph
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend()
plt.title("Accuracy Curve")
plt.show()

# DL 5 Sentiment Analysis on a social network graph using RNN.

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Small dataset of social‑media style posts with positive/negative labels
texts = ["I love this product!", "Terrible experience.", "Amazing quality!",
         "Waste of money.", "Highly recommend it!", "Very disappointed.",
         "Best purchase ever!", "Horrible service.", "Absolutely fantastic!", "Not worth it."]
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 1=pos, 0=neg

# Define who is connected to whom (edges in graph)
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)]

# Convert text to sequences using a simple tokenizer
tok = Tokenizer(num_words=500)
tok.fit_on_texts(texts)
X = pad_sequences(tok.texts_to_sequences(texts), maxlen=8)

# Build RNN model for sentiment
model = Sequential([
    Embedding(500, 16, input_length=8),  # Small vocab, 8‑word sequences
    SimpleRNN(32),  # Learn sentence‑level patterns
    Dense(1, activation='sigmoid')  # Binary sentiment
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, labels, epochs=30, verbose=0)

# Predict sentiment for each node (post)
predicted = (model.predict(X) > 0.5).astype(int).flatten()

# Build network graph structure from edges
G = nx.Graph()
G.add_edges_from(edges)

# Color nodes: green=pos, red=neg; create custom labels
colors = ['#2ecc71' if p == 1 else '#e74c3c' for p in predicted]
label_map = {i: f"N{i}\n{'POS' if p else 'NEG'}" for i, p in enumerate(predicted)}

# Draw the sentiment‑annotated network
nx.draw_networkx(G, nx.spring_layout(G, seed=42), labels=label_map,
                 node_color=colors, node_size=1000, font_color='white', font_size=8)
plt.title("Sentiment Analysis on Network Graph (RNN)")
plt.axis('off')
plt.show()
