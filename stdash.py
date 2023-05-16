import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import requests
import pickle
import sklearn
import datetime
from datetime import date, timedelta
import shap
from PIL import Image

from lightgbm import LGBMClassifier


path1 = 'https://raw.githubusercontent.com/bouzaieni/p7/main/data_orig.csv'
path2 = 'https://raw.githubusercontent.com/bouzaieni/p7/main/data_api.csv'



data_orig = pd.read_csv(path1, encoding='unicode_escape')#.sample(1000, random_state=42)


data_orig = data_orig.loc[:, ~data_orig.columns.str.match ('index')]
data_orig = data_orig.loc[:, ~data_orig.columns.str.match ('Unnamed: 0')]

data_api = pd.read_csv (path2, index_col=0)#.sample(1000, random_state=42)
df_copie = data_api.copy()

features = data_api.columns[: -1] 
features1 = data_orig.columns[: -1].to_list()
ID_de_client = data_api.index.sort_values()
seuil = 0.57



model = pickle.load(open('lgb.pkl', 'rb'))


img = Image.open(r'logo.png')






def load_infos_gen(data):
    lst_infos = [data.shape[0],
                 round(data["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data["AMT_CREDIT"].mean(), 2)]

    nb_credits = lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]

    #gender = data.CODE_GENDER.value_counts()

    return nb_credits, rev_moy, credits_moy#, gender



st.set_page_config(layout="wide")



def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("http://clipart-library.com/images/kT858BAXc.jpg");#https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 


# Cacher le bouton en haut à droite
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# Suppression des marges par défaut
padding = 1
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)




st.sidebar.image(img, caption=" Dashboard - Aide à la décision",
                 use_column_width='always')


st.sidebar.header('Sélectionner le client:')
selected_credit = st.sidebar.selectbox('', ID_de_client)


client_data = st.sidebar.checkbox('Informations client')
client_pred_score = st.sidebar.checkbox('Etude de la demande de crédit')


client_analysis = st.sidebar.checkbox('Graphiques')


features_importance = st.sidebar.checkbox('Facteurs d\'influences')


stats_generales = st.sidebar.checkbox('Stats générales')



html_titre = "<h1><CENTER>Home Credit Default Risk</CENTER></h1>"
st.markdown(html_titre, unsafe_allow_html=True)




if client_data:
    st.write(f'### Client numéro : {selected_credit}')
    features1.remove('TARGET')
    selections_client0 = st.multiselect('Ajouter ou supprimer une caractéristique:', data_orig[features1].columns.tolist(),
    data_orig[features1].columns.tolist()[1:10])
    st.dataframe(data_orig.loc[selected_credit, selections_client0])
    
    
if client_pred_score:
    st.write('### Etude de la demande de prêt')
    #url_api_model_result = 'http://127.0.0.1:5002/credit_client'
    url_api_model_result = 'https://bouzaieni07api.herokuapp.com/credit_client'
    get_request = requests.get(url=url_api_model_result, params={'index': selected_credit})
    prediction_value = get_request.json()['credit_score_pred']
    st.write(f'Client numéro: {selected_credit}')
    today = date.today()
    st.write('Date:', today)
    st.write(f'Risque de remboursement: {prediction_value*100:.2f} %')
    if prediction_value < seuil :
        st.write('Décision : demande acceptée!')
    else:
        if prediction_value > seuil and prediction_value <= 0.65:
            st.write(' Décision : demande refusée --> contacter votre conseiller')
        else:
            st.write('Décision : demande refusée!')

    fig_gauge = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = float(f'{prediction_value*100:.1f}'),
    mode = "gauge+number+delta",
    title = {'text': "Score(%)"},
    delta = {'reference': seuil*100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
    gauge = {'axis': {'range': [0, 100]},
             'bar': {'color': 'darkblue'},
             'steps' : [
                 {'range': [0, 30], 'color': "darkgreen"},
                 {'range': [30, (seuil*100)], 'color': "lightgreen"},
                 {'range': [(seuil*100),53], 'color': "orange"},
                 {'range': [53, 100], 'color':"red"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': seuil*100}}))
    st.plotly_chart(fig_gauge)
    


if client_analysis:
    st.write('### Graphiques')
    
    selected_anaysis_gh = st.selectbox('Sélectionner un graphique', ['Boxplot', 'Histogram'])
    if selected_anaysis_gh == 'Boxplot':
        selections_analysis = st.multiselect('Ajouter ou supprimer une caractéristique::', data_api[features].columns.tolist(),
        data_api[features].columns.tolist()[0:5])
        data_api['data_origin'] = 'Tous les clients'
        cdf = pd.concat([data_api[selections_analysis + ['data_origin']] ])
        df_loan = pd.DataFrame([data_api.loc[selected_credit, features].tolist()], columns=features)
        cdf = pd.melt(cdf, id_vars='data_origin', var_name='Features')
        df_loan = pd.melt(df_loan[selections_analysis], var_name='Features')
        df_loan['data_origin'] = 'client selectionné'


        figure_boxplot = plt.figure(figsize=(4,2))
        ax = sns.boxplot(x = 'Features', y = 'value', hue='data_origin', data=cdf , showfliers=False, palette = 'tab10')
        sns.stripplot(x = 'Features', y = 'value', data = df_loan, hue = 'data_origin', palette=['yellow'], s=8, linewidth=1.5, edgecolor='black')
        plt.xticks(fontsize=3, rotation=45)
        plt.yticks(fontsize=3)
        plt.ylabel('Valeur')
        leg = plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        leg.legendHandles[-1].set_linewidth(1.5)
        leg.legendHandles[-1].set_edgecolor('black')
       
        fig = px.box(cdf, x = 'Features', y = 'value', points = False)
        fig.update_traces(quartilemethod="linear", jitter=0, col=1)
        

        st.pyplot(figure_boxplot, clear_figure=True)
        
    if selected_anaysis_gh == 'Histogram':
        selections_analysis = st.multiselect('Ajouter ou supprimer une caractéristique::', data_api[features].columns.tolist(),
        data_api[features].columns.tolist()[0:5])
        data_api['data_origin'] = 'Tous les clients'
        cdf = pd.concat([data_api[selections_analysis + ['data_origin']] ])
        df_loan = pd.DataFrame([data_api.loc[selected_credit, features].tolist()], columns=features)
        cdf = pd.melt(cdf, id_vars='data_origin', var_name='Features')
        df_loan = pd.melt(df_loan[selections_analysis], var_name='Features')
        df_loan['data_origin'] = 'ID_prêt_client_selectionné'


        figure_boxplot = plt.figure(figsize=(4,2))
        ax = sns.histplot(x = 'Features', y = 'value', hue='data_origin', data=cdf, palette = 'tab10')
        #sns.stripplot(x = 'Features', y = 'value', data = df_loan, hue = 'data_origin', palette=['yellow'], s=8, linewidth=1.5, edgecolor='black')
        plt.xticks(fontsize=6, rotation=45)
        plt.yticks(fontsize=6)
        plt.ylabel('Valeur normalisée')
        leg = plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        fig = px.histogram(cdf, x = 'Features', y = 'value')
       
        st.plotly_chart(fig)
   



if features_importance:
    st.write('### Variables importantes dans la décision de prêt')
    st.write(f'Client: {selected_credit}')
    df_copie_no = df_copie.loc[:, ~df_copie.columns.str.match('TARGET')]
    bar_explainer = shap.Explainer(model, df_copie_no)
    bar_values = bar_explainer(df_copie_no, check_additivity=False)
    explainer = shap.TreeExplainer(model)
    X_test_courant = df_copie_no[df_copie_no.index == int(selected_credit)]
    X_test_courant_array = X_test_courant.values.reshape(1, -1)
    shap_values_courant = explainer.shap_values(X_test_courant_array)
    
    df_copie_noo = df_copie_no.reset_index()
    indice_cl = df_copie_noo.loc[df_copie_noo['index'] == selected_credit].index.item()
    plt.clf()
    shap.plots.bar(bar_values[indice_cl], max_display=40)
    fig_shap1 = plt.gcf()
    st.pyplot(fig_shap1)
    
    plt.clf()
    shap.decision_plot(explainer.expected_value[1], shap_values_courant[1], X_test_courant)
    fig_shap2 = plt.gcf()
    st.pyplot(fig_shap2)
    
    
    
    
    
    
if stats_generales:
	st.write('### Informations générales:')
	st.write('Dimension des données: ' + str(data_orig.shape[0]) + ' lignes et ' + str(data_orig.shape[1]) + ' colonnes')
	selections = st.multiselect('Ajouter ou supprimer une caractéristique:', data_orig.columns.tolist(),
 	data_orig.columns.tolist()[0:10])
	st.dataframe(data_orig.loc[:,selections].sort_index())
	colors = ['red', 'blue']

	fig = go.Figure(data=[go.Pie(labels=[ "Default", "No Default"],
                             values=[8.3 , 91.7])])
	fig.update_traces(hoverinfo='label+percent', textinfo='label+percent', textfont_size=20,pull=[0, 0.1],
                  marker=dict(colors=colors, line=dict(color='#000000', width=1)))

	st.write("### Pourcentage de 'TARGET'")
	st.plotly_chart(fig)




	colors = ['red', 'pink']
	fig = go.Figure(data=[go.Pie(labels=[ "Hommes", "Femmes"],
                             values=[65 , 35])])
	fig.update_traces(hoverinfo='label+percent', textinfo='label+percent', textfont_size=20,pull=[0, 0.05],
                  marker=dict(colors=colors, line=dict(color='#000000', width=1)))
	st.write("### Pourcentage par  sexe")
	st.plotly_chart(fig)



	fig = go.Figure(data=[go.Pie(labels=[ "Secondary", "Gigher education", "Incomplete higher", "Lower secondary", "Academic degree"],
				values=data_orig['NAME_EDUCATION_TYPE'].value_counts())])
	fig.update_traces(hoverinfo='label+percent', textinfo='label+percent', textfont_size=20,pull=[0, 0.05],
                  marker=dict(line=dict(color='#000000', width=1)))
	st.write("### Pourcentage par  type d\'education")
	st.plotly_chart(fig)



