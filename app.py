import streamlit as st  
import joblib
import numpy as np

# Load the trained model
model = joblib.load('student_risk_model.pkl')

# Define the feature names exactly as they were used during training
feature_names = [
    'Marital status',
    'Application mode',
    'Application order',
    'Course',
    'Daytime/evening attendance',
    'Previous qualification',
    'Previous qualification (grade)',
    'Nationality',
    'Mother\'s qualification',
    'Father\'s qualification',
    'Mother\'s occupation',
    'Father\'s occupation',
    'Admission grade',
    'Displaced',
    'Educational special needs',
    'Debtor',
    'Tuition fees up to date',
    'Gender',
    'Scholarship holder',
    'Age at enrollment',
    'International',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]

# Explanations for features (add tooltips to make it easier to understand)
descriptions = {
    'Marital status': '1 – single, 2 – married, 3 – widower, 4 – divorced, 5 – facto union, 6 – legally separated',
    'Application mode': '1 - 1st phase - general contingent, 2 - Ordinance No. 612/93, 3 - Azores/ Madeira special contingent, 4 - Economically challenged contingent, 5 - Other contingents, 6 - Transfer from other institution',
    'Application order': 'Between 0 – first choice and 9 – last choice',
    'Course': 'Course code related to the program the student is enrolled in.',
    'Daytime/evening attendance': '1 – daytime, 0 – evening',
    'Previous qualification': 'The level of education achieved before enrolling in the current program.',
    'Previous qualification (grade)': 'Grade achieved in the student\'s previous qualification (0-200)',
    'Nationality': 'Nationality of the student (e.g., 1 - Portuguese, 2 - German, 3 - Spanish, etc.)',
    'Mother\'s qualification': 'The highest level of education achieved by the mother.',
    'Father\'s qualification': 'The highest level of education achieved by the father.',
    'Mother\'s occupation': 'The occupation of the mother (job classification code).',
    'Father\'s occupation': 'The occupation of the father (job classification code).',
    'Admission grade': 'Grade achieved during the admission process (0-200)',
    'Displaced': 'Whether the student is displaced (1 - yes, 0 - no)',
    'Educational special needs': 'Whether the student has special educational needs (1 - yes, 0 - no)',
    'Debtor': 'Whether the student has unpaid debts (1 - yes, 0 - no)',
    'Tuition fees up to date': 'Whether the student\'s tuition fees are paid (1 - yes, 0 - no)',
    'Gender': '1 – male, 0 – female',
    'Scholarship holder': 'Whether the student is a scholarship holder (1 - yes, 0 - no)',
    'Age at enrollment': 'Age of student at the time of enrollment.',
    'International': 'Whether the student is an international student (1 - yes, 0 - no)',
    'Curricular units 1st sem (credited)': 'Number of curricular units credited in the 1st semester',
    'Curricular units 1st sem (enrolled)': 'Number of curricular units enrolled in the 1st semester',
    'Curricular units 1st sem (evaluations)': 'Number of evaluations taken for curricular units in the 1st semester',
    'Curricular units 1st sem (approved)': 'Number of curricular units approved in the 1st semester',
    'Curricular units 1st sem (grade)': 'Grade average in the 1st semester (0-20)',
    'Curricular units 1st sem (without evaluations)': 'Number of curricular units without evaluations in the 1st semester',
    'Curricular units 2nd sem (credited)': 'Number of curricular units credited in the 2nd semester',
    'Curricular units 2nd sem (enrolled)': 'Number of curricular units enrolled in the 2nd semester',
    'Curricular units 2nd sem (evaluations)': 'Number of evaluations taken for curricular units in the 2nd semester',
    'Curricular units 2nd sem (approved)': 'Number of curricular units approved in the 2nd semester',
    'Curricular units 2nd sem (grade)': 'Grade average in the 2nd semester (0-20)',
    'Curricular units 2nd sem (without evaluations)': 'Number of curricular units without evaluations in the 2nd semester',
    'Unemployment rate': 'Unemployment rate in % at the time of enrollment.',
    'Inflation rate': 'Inflation rate in % at the time of enrollment.',
    'GDP': 'Gross Domestic Product of the country at the time of enrollment.'
}

# Create input fields for all features with tooltips
st.title("Student At-Risk Prediction Tool")

input_data = []
valid = True
errors = []

for feature in feature_names:
    tooltip = descriptions.get(feature, None)
    
    if feature in descriptions and isinstance(tooltip, str):
        # For categorical and binary variables, show a dropdown with clearly explained options
        if feature == 'Gender':
            options = [(0, '0 - Female'), (1, '1 - Male')]
            value = st.selectbox(f"{feature}:", options=[option[1] for option in options], help=tooltip)
            selected_value = next(option[0] for option in options if option[1] == value)
        
        elif feature == 'Marital status':
            options = [(1, '1 - Single'), (2, '2 - Married'), (3, '3 - Widower'), 
                       (4, '4 - Divorced'), (5, '5 - Facto union'), (6, '6 - Legally separated')]
            value = st.selectbox(f"{feature}:", options=[option[1] for option in options], help=tooltip)
            selected_value = next(option[0] for option in options if option[1] == value)

        elif feature in ['Displaced', 'Educational special needs', 'Debtor', 
                         'Tuition fees up to date', 'Scholarship holder', 'International']:
            options = [(0, '0 - No'), (1, '1 - Yes')]
            value = st.selectbox(f"{feature}:", options=[option[1] for option in options], help=tooltip)
            selected_value = next(option[0] for option in options if option[1] == value)

        else:
            # For integer inputs (remove decimal places)
            value = st.number_input(f"Enter {feature}:", help=tooltip, min_value=0, format='%d')
            selected_value = int(value)  # Ensures integer value
    else:
        value = st.number_input(f"Enter {feature}:", min_value=0, format='%d')
        selected_value = int(value)  # Ensures integer value
    
    input_data.append(selected_value)

if st.button("Predict"):
    # Validation logic
    valid = True
    errors = []

    for feature, value in zip(feature_names, input_data):
        if feature == 'Previous qualification (grade)' or feature == 'Admission grade':
            if value < 0 or value > 200:
                valid = False
                errors.append(f"{feature} should be between 0 and 200.")
        elif feature == 'Age at enrollment':
            if value < 0 or value > 100:
                valid = False
                errors.append(f"{feature} should be between 0 and 100.")
        elif feature in ['Daytime/evening attendance', 'Displaced', 'Educational special needs', 
                         'Debtor', 'Tuition fees up to date', 'Gender', 
                         'Scholarship holder', 'International']:
            if value not in [0, 1]:
                valid = False
                errors.append(f"{feature} should be 0 or 1.")
        elif feature == 'Marital status':
            if value not in [1, 2, 3, 4, 5, 6]:
                valid = False
                errors.append(f"{feature} should be one of the following values: 1, 2, 3, 4, 5, 6.")
        elif feature == 'Application order':
            if value < 0 or value > 9:
                valid = False
                errors.append(f"{feature} should be between 0 and 9.")
        elif feature.startswith('Curricular units') and 'grade' in feature:
            if value < 0 or value > 20:
                valid = False
                errors.append(f"{feature} should be between 0 and 20.")

    if valid:
        input_data = np.array(input_data).reshape(1, -1)
        # Make the prediction
        prediction = model.predict(input_data)
        # Display the result
        if prediction[0] == 1:
            st.write("The student is at risk.")
        else:
            st.write("The student is not at risk.")
    else:
        for error in errors:
            st.error(error)