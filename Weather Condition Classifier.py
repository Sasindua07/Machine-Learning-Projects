import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = '/content/daily_data.csv'
weather_data = pd.read_csv(file_path)

# Encode categorical variables 'city_id', 'sunrise', and 'sunset'
label_encoders = {}
for column in ['city_id', 'sunrise', 'sunset']:
    le = LabelEncoder()
    weather_data[column] = le.fit_transform(weather_data[column])
    label_encoders[column] = le

# Fill missing values in 'condition_text' with a placeholder 'missing'
weather_data['condition_text'].fillna('missing', inplace=True)

# Prepare features and target variable
X = weather_data.drop(columns=['condition_text', 'sunrise', 'sunset'])
y = weather_data['condition_text']

# Encode the target variable 'condition_text'
le_condition = LabelEncoder()
y = le_condition.fit_transform(y)

# Select rows where 'condition_text' is not 'missing'
initial_data = weather_data[weather_data['condition_text'] != 'missing']

# Separate features and target variable for the initial dataset
X_initial = initial_data.drop(columns=['condition_text', 'day_id'])
y_initial = initial_data['condition_text']

# Encode the target variable for the initial dataset
y_initial_encoded = le_condition.fit_transform(y_initial)

# Split the initial dataset into training and testing sets
X_train_initial, X_test_initial, y_train_initial, y_test_initial = train_test_split(X_initial, y_initial_encoded, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model using the initial dataset
model_initial = RandomForestClassifier(random_state=42)
model_initial.fit(X_train_initial, y_train_initial)

# Predict on the test set using the initial dataset
y_pred_initial = model_initial.predict(X_test_initial)

# Decode the predicted labels for the test set
y_test_initial_decoded = le_condition.inverse_transform(y_test_initial)
y_pred_initial_decoded = le_condition.inverse_transform(y_pred_initial)

# Train the RandomForestClassifier model on the full initial dataset
model_initial.fit(X_initial, y_initial_encoded)

# Prepare the full dataset for prediction
X_full = weather_data.drop(columns=['condition_text', 'day_id'])

# Predict missing 'condition_text' values in the entire dataset
y_full_pred_encoded = model_initial.predict(X_full)

# Decode the predicted labels for the entire dataset
y_full_pred_decoded = le_condition.inverse_transform(y_full_pred_encoded)

# Create a mapping from day_id to the predicted condition_text
day_id_to_condition = dict(zip(weather_data['day_id'], y_full_pred_decoded))

# Load the submission data
submission_data = pd.read_csv('/content/submission.csv')

# Update the condition_text in the submission file only for missing values
submission_data['condition_text'] = submission_data.apply(
    lambda row: day_id_to_condition[row['day_id']] if pd.isnull(row['condition_text']) else row['condition_text'],
    axis=1
)

# Save the updated submission dataframe to a new CSV file
submission_data.to_csv('/content/submission.csv', index=False)

# Display the first few rows of the updated submission file
print(submission_data.head())
