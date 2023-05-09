import streamlit as st
from joblib import dump, load
import numpy as np


st.title("prediction diabète")

#X=df[['BMI','Smoker','GenHlth']]
taille = float(st.text_input('Rentrez votre taille (m)', '1'))
poids= float(st.text_input('Rentrez votre poids ', '0'))
fumeur=float(st.text_input('fumeur ? 0/1 ', '0'))
gn= float(st.text_input('GenHlth 1->5 1 meilleure valeur ', '0'))
IMC=poids/(taille**2)
obs=np.array([IMC,fumeur,gn])


if st.button('send'):
        with open('xgb_model.joblib', 'rb') as file:
            model = load(file)

        #params = model.get_xgb_params()
        #model.set_params(**params)
        pred=model.predict_proba(obs.reshape(1, -1))
        st.write(pred)
        
