# Simplified model training script
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading data...")
# Load dataset
df = pd.read_csv("student_lifestyle_dataset.csv")

# Create Letter Grade column
def gpa_to_letter(gpa):
    if gpa >= 3.7:
        return 'A'
    elif gpa >= 3.3:
        return 'A-'
    elif gpa >= 3.0:
        return 'B+'
    elif gpa >= 2.7:
        return 'B'
    elif gpa >= 2.3:
        return 'B-'
    elif gpa >= 2.0:
        return 'C+'
    elif gpa >= 1.7:
        return 'C'
    elif gpa >= 1.3:
        return 'C-'
    elif gpa >= 1.0:
        return 'D+'
    elif gpa >= 0.7:
        return 'D'
    else:
        return 'F'

# Apply letter grade conversion
df['Letter_Grade'] = df['GPA'].apply(gpa_to_letter)

# Group letter grades for better prediction
def group_letter_grade(letter):
    if letter in ['A', 'A-']:
        return 'A'
    elif letter in ['B+', 'B', 'B-']:
        return 'B'
    elif letter in ['C+', 'C', 'C-']:
        return 'C'
    else:
        return 'D/F'

# Apply grouping
df['Grouped_Letter_Grade'] = df['Letter_Grade'].apply(group_letter_grade)

# Define features and targets
X = df.drop(columns=['Student_ID', 'GPA', 'Letter_Grade', 'Grouped_Letter_Grade', 'Stress_Level'])
y = df[['Grouped_Letter_Grade', 'Stress_Level']]

print("Engineering features...")
# Create interaction features
# Study-Sleep interaction (quality study time)
X['Study_Sleep_Interaction'] = X['Study_Hours_Per_Day'] * X['Sleep_Hours_Per_Day']

# Social-to-Study ratio (balance indicator)
X['Social_Study_Ratio'] = X['Social_Hours_Per_Day'] / (X['Study_Hours_Per_Day'] + 1e-5)

# Total active time (energy expenditure)
X['Total_Activity_Hours'] = (
    X['Study_Hours_Per_Day'] + 
    X['Extracurricular_Hours_Per_Day'] + 
    X['Physical_Activity_Hours_Per_Day']
)

# Study efficiency (sleep-adjusted study potency)
X['Study_Efficiency'] = X['Study_Hours_Per_Day'] * (X['Sleep_Hours_Per_Day'] / 8)

# Balance metric (proper mix of activities)
X['Life_Balance'] = (
    (X['Sleep_Hours_Per_Day'] / 24) * 
    (X['Study_Hours_Per_Day'] / 24) * 
    (X['Physical_Activity_Hours_Per_Day'] / 24)
) * 100  # Scale up for readability

# Encode targets
grade_encoder = LabelEncoder()
stress_encoder = LabelEncoder()

grade_labels = grade_encoder.fit_transform(y['Grouped_Letter_Grade'])
stress_labels = stress_encoder.fit_transform(y['Stress_Level'])

# Create encoded dataframe
y_encoded = pd.DataFrame({
    'Grouped_Letter_Grade': grade_labels,
    'Stress_Level': stress_labels
})

# Store mapping for later interpretation
grade_mapping = {i: label for i, label in enumerate(grade_encoder.classes_)}
stress_mapping = {i: label for i, label in enumerate(stress_encoder.classes_)}

print("Grade mapping:", grade_mapping)
print("Stress level mapping:", stress_mapping)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, 
    stratify=y_encoded['Grouped_Letter_Grade']
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Random Forest model...")
# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,  # More trees for better accuracy
    max_depth=12,      # Slightly deeper trees
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,         # Use all cores
    class_weight='balanced'
)

# Create multi-output classifier
multi_rf = MultiOutputClassifier(rf_model, n_jobs=-1)

# Train model
multi_rf.fit(X_train_scaled, y_train)

# Evaluate model
print("\nEvaluating model...")
y_pred = multi_rf.predict(X_test_scaled)

# Grade accuracy
grade_acc = accuracy_score(y_test['Grouped_Letter_Grade'], y_pred[:, 0])
print(f"Letter Grade Accuracy: {grade_acc:.4f}")
print("\nLetter Grade Report:")
print(classification_report(y_test['Grouped_Letter_Grade'], y_pred[:, 0], 
                           target_names=grade_encoder.classes_, zero_division=0))

# Stress level accuracy
stress_acc = accuracy_score(y_test['Stress_Level'], y_pred[:, 1])
print(f"\nStress Level Accuracy: {stress_acc:.4f}")
print("\nStress Level Report:")
print(classification_report(y_test['Stress_Level'], y_pred[:, 1], 
                           target_names=stress_encoder.classes_, zero_division=0))

print(f"\nOverall Model Accuracy: {(grade_acc + stress_acc) / 2:.4f}")

# Create model wrapper class
class StudentPredictionModel:
    def __init__(self, model, scaler, grade_encoder, stress_encoder, feature_names):
        self.model = model
        self.scaler = scaler
        self.grade_encoder = grade_encoder
        self.stress_encoder = stress_encoder
        self.grade_mapping = {i: label for i, label in enumerate(grade_encoder.classes_)}
        self.stress_mapping = {i: label for i, label in enumerate(stress_encoder.classes_)}
        self.feature_names = feature_names
    
    def predict(self, X_input):
        """
        Makes predictions from raw input features and returns human-readable labels
        
        Args:
            X_input: DataFrame with the base features (no need for engineered features)
        
        Returns:
            List of predictions with human-readable labels [letter_grade, stress_level]
        """
        # Engineer features if they're not already present
        X_processed = X_input.copy()
        
        # Create missing features
        if 'Study_Sleep_Interaction' not in X_processed.columns:
            # Study-Sleep interaction
            X_processed['Study_Sleep_Interaction'] = X_processed['Study_Hours_Per_Day'] * X_processed['Sleep_Hours_Per_Day']
            
            # Social-to-Study ratio
            X_processed['Social_Study_Ratio'] = X_processed['Social_Hours_Per_Day'] / (X_processed['Study_Hours_Per_Day'] + 1e-5)
            
            # Total active time
            X_processed['Total_Activity_Hours'] = (
                X_processed['Study_Hours_Per_Day'] + 
                X_processed['Extracurricular_Hours_Per_Day'] + 
                X_processed['Physical_Activity_Hours_Per_Day']
            )
            
            # Study efficiency
            X_processed['Study_Efficiency'] = X_processed['Study_Hours_Per_Day'] * (X_processed['Sleep_Hours_Per_Day'] / 8)
            
            # Balance metric
            X_processed['Life_Balance'] = (
                (X_processed['Sleep_Hours_Per_Day'] / 24) * 
                (X_processed['Study_Hours_Per_Day'] / 24) * 
                (X_processed['Physical_Activity_Hours_Per_Day'] / 24)
            ) * 100
        
        # Ensure columns are in the right order
        X_processed = X_processed[self.feature_names]
        
        # Scale the features
        X_scaled = self.scaler.transform(X_processed)
        
        # Make prediction
        y_encoded = self.model.predict(X_scaled)
        
        # Decode predictions
        grade_idx = int(y_encoded[0][0])
        stress_idx = int(y_encoded[0][1])
        
        letter_grade = self.grade_mapping[grade_idx]
        stress_level = self.stress_mapping[stress_idx]
        
        return [letter_grade, stress_level]

# Create wrapper model
print("\nCreating model wrapper...")
wrapped_model = StudentPredictionModel(
    multi_rf, 
    scaler,
    grade_encoder,
    stress_encoder,
    list(X.columns)
)

# Save the model
print("Saving model...")
joblib.dump(wrapped_model, "stacked_multioutput_predictor.pkl")
print("Model saved successfully!")

# Test the saved model
print("\nTesting saved model with sample input...")
sample_student = pd.DataFrame({
    'Study_Hours_Per_Day': [7.0],
    'Extracurricular_Hours_Per_Day': [2.5],
    'Sleep_Hours_Per_Day': [7.5],
    'Social_Hours_Per_Day': [3.0],
    'Physical_Activity_Hours_Per_Day': [1.5]
})

# Load model and predict
loaded_model = joblib.load("stacked_multioutput_predictor.pkl")
prediction = loaded_model.predict(sample_student)

print(f"Sample Student Input:")
print(sample_student.iloc[0][['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day']])
print(f"\nPredicted Letter Grade: {prediction[0]}")
print(f"Predicted Stress Level: {prediction[1]}")

print("\nModel ready for deployment in the app.py application!")
