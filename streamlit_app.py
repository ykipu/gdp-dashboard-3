import streamlit as st

# Set the title of the web app
st.title('Medical Recommendation System')

# Input fields for user symptoms
st.header('Enter your symptoms')
symptoms = st.text_input('Symptoms (comma-separated)', '')

# Example symptom-diagnosis mapping
symptom_diagnosis = {
    'fever': 'You might have the flu or a common cold. Drink plenty of fluids and rest. If fever persists, see a doctor.',
    'headache': 'It could be a tension headache or migraine. Try to rest in a dark, quiet room and stay hydrated.',
    'sore throat': 'It might be a sore throat due to a viral infection. Gargle with warm salt water and stay hydrated.',
    'cough': 'You could have a respiratory infection. Drink warm fluids and rest. If cough persists, see a doctor.'
}

# Function to provide recommendations based on symptoms
def get_recommendation(symptoms):
    recommendations = []
    symptom_list = symptoms.lower().split(',')
    for symptom in symptom_list:
        symptom = symptom.strip()
        if symptom in symptom_diagnosis:
            recommendations.append(symptom_diagnosis[symptom])
        else:
            recommendations.append(f'No recommendation available for symptom: {symptom}')
    return recommendations

# Display recommendations
if symptoms:
    st.header('Recommendations')
    recommendations = get_recommendation(symptoms)
    for rec in recommendations:
        st.write(rec)

