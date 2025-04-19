# app.py
import streamlit as st 
import pandas as pd 
import numpy as np
import tensorflow as tf
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Load model and label encoder
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('./models/disease_prediction_model.h5')
    with open('./models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder


# Load dataset
@st.cache_data
def load_data():
    training_data = pd.read_csv('./data/Training.csv')
    symptom_severity = pd.read_csv('./data/Symptom-severity.csv')
    medications = pd.read_csv('./data/medications.csv')
    diets = pd.read_csv('./data/diets.csv')
    precautions = pd.read_csv('./data/precautions_df.csv')
    descriptions = pd.read_csv('./data/description.csv')
    workouts = pd.read_csv('./data/workout_df.csv')

    return {
        'trainig_data': training_data,
        'symptom_severity': symptom_severity,
        'medications': medications,
        'diets': diets,
        'precautions': precautions,
        'descriptions': descriptions,
        'workouts': workouts
    }


def predict_disease(symptoms, model, label_encoder, all_symptoms):
    input_data = np.zeros(len(all_symptoms))

    for symptom in symptoms:
        if symptom in all_symptoms:
            input_data[all_symptoms.get_loc(symptom)] = 1
    
    input_data = input_data.reshape(1, -1)

    prediction = model.predict(input_data)
    disease_index = np.argmax(prediction, axis=1)[0]
    disease = label_encoder.inverse_transform([disease_index])[0]
    return disease, prediction[0][disease_index]


def get_recommendations(disease, data):
    meds = data['medications'][data['medications']['Disease'] == disease]
    diet = data['diets'][data['diets']['Disease'] == disease]
    precaution = data['precautions'][data['precautions']['Disease'] == disease]
    description = data['descriptions'][data['descriptions']['Disease'] == disease]
    workout = data['workouts'][data['workouts']['disease'] == disease]

    # Perbaikan untuk medications
    med_list = []
    if not meds.empty:
        med_str = meds.iloc[0, 1]
        if isinstance(med_str, str):
            # Hapus karakter ['...'] dan pisahkan berdasarkan koma
            med_str = med_str.strip('[]\'\"')
            # Pisahkan string dan bersihkan setiap item
            med_items = [item.strip().strip('\'\"') for item in med_str.split(',')]
            med_list = [item for item in med_items if item]
    
    # Perbaikan untuk diet
    diet_list = []
    if not diet.empty:
        diet_str = diet.iloc[0, 1]
        if isinstance(diet_str, str):
            # Hapus karakter ['...'] dan pisahkan berdasarkan koma
            diet_str = diet_str.strip('[]\'\"')
            # Pisahkan string dan bersihkan setiap item
            diet_items = [item.strip().strip('\'\"') for item in diet_str.split(',')]
            diet_list = [item for item in diet_items if item]
    
    return {
        'medications': med_list,
        'diet': diet_list,
        'precautions': precaution.iloc[0, 1:].dropna().tolist() if not precaution.empty else [],
        'description': description.iloc[0, 1] if not description.empty else "No description available",
        'workout': workout.iloc[0, 1:].dropna().tolist()[-1] if not workout.empty else []
    }


def main():
    st.set_page_config(
        page_title="Medicine Recommendation System",
        page_icon='💊',
        layout='wide'
    )

    st.title('Medicine Recommendation System 💊')
    st.write('Select your symptoms and get personalized medicine recommendations')

    model, label_encoder = load_model()
    data = load_data()

    all_symptoms = data['trainig_data'].drop('prognosis', axis=1).columns

    st.sidebar.header('Select Your Symptoms')

    selected_symptoms = st.sidebar.multiselect(
        "Choose all symptoms you are experiencing:",
        options=all_symptoms,
        help="You can select multiple symptoms"
    )

    if st.sidebar.button('Get Recommendations'):
        if len(selected_symptoms) > 0:
            disease, confidence = predict_disease(selected_symptoms, model, label_encoder, all_symptoms)

            recommendations = get_recommendations(disease, data)

            st.header(f'Diagnosis: {disease}')
            st.subheader(f'Confidence: {confidence:.2f}')

            st.subheader('Disease Description')
            st.write(recommendations['description'])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader('Recommended Medications')
                if recommendations['medications']:
                    for med in recommendations['medications']:
                        st.write(f"- {med}")
                else:
                    st.write('No specific medications found.')
            
            with col2:
                st.subheader('Recommended Diet')
                if recommendations['diet']:
                    for diet in recommendations['diet']:
                        st.write(f'- {diet}')
                else:
                    st.write('No specific diet recommendations found.')
            
            with col3:
                st.subheader('Precautions')
                if recommendations['precautions']:
                    for precaution in recommendations['precautions']:
                        st.write(f'- {precaution}')
                else:
                    st.write('No specific precautions found.')
            
            st.subheader('Recommended Workouts')
            if recommendations['workout']:
                st.write(f'- {recommendations["workout"]}')
            else:
                st.write('No specific workout recommendations found.')

            # Plotly visualization for symptom severity
            st.subheader('Symptom Severity Analysis')

            severity_data = data['symptom_severity']
            severity_data.columns = ['Symptom', 'weight']
            selected_severity = severity_data[severity_data['Symptom'].isin(selected_symptoms)]

            if not selected_severity.empty:
                # Create interactive Plotly bar chart
                fig = px.bar(
                    selected_severity, 
                    x='Symptom', 
                    y='weight',
                    title='Severity of Selected Symptoms',
                    color='weight',
                    labels={'weight': 'Severity Score', 'Symptom': 'Symptom Name'},
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a radar chart for symptom comparison
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatterpolar(
                    r=selected_severity['weight'],
                    theta=selected_severity['Symptom'],
                    fill='toself',
                    name='Symptom Severity'
                ))
                
                fig2.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, selected_severity['weight'].max() + 1]
                        )),
                    title="Symptom Severity Radar Chart"
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.write('No severity data available for selected symptoms.')
            
            # Add a pie chart for recommendations distribution
            categories = ['Medications', 'Diet', 'Precautions', 'Workout']
            counts = [
                len(recommendations['medications']), 
                len(recommendations['diet']), 
                len(recommendations['precautions']), 
                len(recommendations['workout'])
            ]
            
            if sum(counts) > 0:
                fig3 = px.pie(
                    values=counts,
                    names=categories,
                    title='Distribution of Recommendations',
                    color_discrete_sequence=px.colors.sequential.Plasma
                )
                st.plotly_chart(fig3, use_container_width=True)
        
        else:
            st.warning('Please select at least one symptom.')
    
    st.sidebar.header('About')
    st.sidebar.info(
        'This application uses a deep learning model to predict diseases based on symptoms '
        'and provides personalized medicine recommendations. The model was trained on a dataset '
        'containing various diseases and their associated symptoms.'
    )

if __name__ == "__main__":
    main()