import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import warnings

# Load dataset
data = pd.read_csv("Data.csv")
warnings.filterwarnings('ignore')

# Streamlit UI Title
st.title('🏋️‍♂️ Fitness Tracker Analysis: Heart Rate, Sleep, and Daily Activity Monitoring')

# Sidebar Navigation
st.sidebar.title('📌 Navigation')
page = st.sidebar.radio('Go to:', ['Dataset', 'Heart Rate Analysis', 'Classification', 'Recommendations'])

# Dataset Preview
if page == 'Dataset':
    st.subheader('📊 Dataset Preview')
    st.write(data.head())

# Time Series Analysis - Heart Rate Visualization with Filters
if page == 'Heart Rate Analysis':
    st.subheader('📈 Heart Rate Time Series Analysis')
    data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S')
    data = data.sort_values(by='Time')
    
    # Filtering Data Points
    sample_size = st.slider("Select number of sample data points:", 10, 50, 20)
    sample_data = data.sample(sample_size)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=sample_data['Time'], y=sample_data['HeartRate (bpm)'], marker='o', color='blue')
    ax.set_xlabel('Time')
    ax.set_ylabel('Heart Rate (bpm)')
    ax.set_title('Heart Rate Trend (Filtered)')
    ax.grid(True)
    st.pyplot(fig)
    
# Categorizing Age Groups
st.subheader('Age Group Categorization: Healthy vs. Unhealthy')
conditions = [
    (data['Age'] <= 30) & (data['HeartRate (bpm)'] < 100) & (data['SleepDuration (hours)'] >= 7),
    (data['Age'] > 30) & (data['HeartRate (bpm)'] < 85) & (data['SleepDuration (hours)'] >= 6)
]
labels = ['Healthy', 'Unhealthy']
data['Health_Status'] = np.select(conditions, labels, default='Unhealthy')
st.write(data[['Age', 'HeartRate (bpm)', 'SleepDuration (hours)', 'Health_Status']].head(10))

# Classification Model (Decision Tree)
if page == 'Classification':
    st.subheader('🤖 User Classification - Decision Tree')
    X = data[['TotalSteps', 'HeartRate (bpm)', 'SleepDuration (hours)', 'StressLevel', 'Hydration (L)']]
    y = data['Health_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write('**Accuracy:**', accuracy_score(y_test, y_pred))
    st.write('**Precision:**', precision_score(y_test, y_pred, average='weighted'))
    st.write('**Recall:**', recall_score(y_test, y_pred, average='weighted'))
    st.write('**F1 Score:**', f1_score(y_test, y_pred, average='weighted'))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax_cm)
    st.pyplot(fig_cm)

# Health Alerts & Recommendations
if page == 'Recommendations':
    st.subheader('🚨 Health Alerts & Personalized Recommendations')
    recommendations = [
        "Stay active and maintain a balanced workout routine!",
        "Try meditating or yoga to reduce stress levels.",
        "Ensure you drink enough water daily to stay hydrated.",
        "Maintain a consistent sleep schedule for better recovery.",
        "Reduce screen time before bed for quality sleep.",
        "Incorporate stretching and mobility exercises in your routine.",
        "Monitor your heart rate trends to detect early health warnings.",
        "Eat a balanced diet rich in proteins, vitamins, and minerals.",
        "Consider reducing caffeine intake if sleep is an issue.",
        "Take breaks between workouts to allow muscle recovery."
    ]
    
    for index, row in data.iterrows():
        advice = np.random.choice(recommendations)
        status = "✅ Keep up the good work maintaining your fitness routine!" if row['Health_Status'] == 'Healthy' else f"🚨 {advice}"
        st.write(f"**User {row['ID']}**: {status}")


