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



df_test = pd.read_csv(path1, encoding='unicode_escape')#.sample(1000, random_state=42)


df_test = df_test.loc[:, ~df_test.columns.str.match ('index')]
df_test = df_test.loc[:, ~df_test.columns.str.match ('Unnamed: 0')]

df_test_normalize = pd.read_csv (path2, index_col=0)#.sample(1000, random_state=42)
df_copie = df_test_normalize.copy()

features = df_test_normalize.columns[: -1] 
features1 = df_test.columns[: -1].to_list()
ID_de_client = df_test_normalize.index.sort_values()
treshold = 0.50



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



### Display the image with streamlit ###
st.sidebar.image(img, caption=" Dashboard - Aide à la décision",
                 use_column_width='always')

### Add column for user input ###
st.sidebar.header('Sélectionner le client:')
selected_credit = st.sidebar.selectbox('', ID_de_client)
### Add checkbox for displaying different client informations ###

client_data = st.sidebar.checkbox('Informations client')
client_pred_score = st.sidebar.checkbox('Etude de la demande de crédit')

### Add checkbox for displaying client data analysis ###
client_analysis = st.sidebar.checkbox('Graphiques')

### Add checkbox for displaying client data analysis ###
features_importance = st.sidebar.checkbox('Facteurs d\'influences')


stats_generales = st.sidebar.checkbox('Stats générales')


# ------------ Main display, part by part -----------------
## Generic title ##

html_titre = "<h1><CENTER>Home Credit Default Risk</CENTER></h1>"
st.markdown(html_titre, unsafe_allow_html=True)





## Display selected client data (checkbox condition: 'Données client') ##
if client_data:
    st.write(f'### Client numéro : {selected_credit}')
    features1.remove('TARGET')
    selections_client0 = st.multiselect('Ajouter ou supprimer une caractéristique:', df_test[features1].columns.tolist(),
    df_test[features1].columns.tolist()[1:10])
    st.dataframe(df_test.loc[selected_credit, selections_client0])
    
    
## Display loan answer regarding model probability calcul (path through API Flask to get the result / checbox condition : 'Résultat de la demande de prêt') ##
if client_pred_score:
    st.write('### Etude de la demande de prêt')
    ### careful the url of the API should be change for serial deployment!! ###
    #url_api_model_result = 'https://mankarali07api.herokuapp.com/scores'
    url_api_model_result = 'http://127.0.0.1:5002/credit_client'
    ### Be careful to the params, with must have a dict with index / ID loan value. It is how it is implemented in our API ###
    get_request = requests.get(url=url_api_model_result, params={'index': selected_credit})
    ### We get  the prediction information from the json format of the API model ###
    prediction_value = get_request.json()['credit_score_pred']
    ### We get the answer regardin loan acceptation ###
    #answer_value = bool(get_request.json()['Answer'])
    ### Display results ###
    st.write(f'Client numéro: {selected_credit}')
    today = date.today()
    #default_date_yesterday = today - timedelta(days=1)
    #date_today = st.date_input("Date of completion", today)
    st.write('Date:', today)
    #st.markdown(""":red['Date:'], today""")
    st.write(f'Risque de remboursement: {prediction_value*100:.2f} %')
    if prediction_value < treshold :
        st.write('Décision : demande acceptée!')
    else:
        #### add condition in function of the value of the prediction, if over the treshold but near should be discussed ####
        if prediction_value > treshold and prediction_value <= 0.53:
            st.write(' Décision : demande refusée --> contacter votre conseiller')
        else:
            st.write('Décision : demande refusée!')
    ### add gauge for the prediction value with plotly library ###
    fig_gauge = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = float(f'{prediction_value*100:.1f}'),
    mode = "gauge+number+delta",
    title = {'text': "Score(%)"},
    delta = {'reference': treshold*100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
    gauge = {'axis': {'range': [0, 100]},
             'bar': {'color': 'darkblue'},
             'steps' : [
                 {'range': [0, 30], 'color': "darkgreen"},
                 {'range': [30, (treshold*100)], 'color': "lightgreen"},
                 {'range': [(treshold*100),53], 'color': "orange"},
                 {'range': [53, 100], 'color':"red"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': treshold*100}}))
    st.plotly_chart(fig_gauge)
    


if client_analysis:
    st.write('### Graphiques')
    
    ### Univariate analysis choose type of plot (boxplot or histogram/bargraph) ###
    
    #### select between boxplot or histogram/barplot distributions for univariate analysis ####
    selected_anaysis_gh = st.selectbox('Sélectionner un graphique', ['Boxplot', 'Histogram'])
    if selected_anaysis_gh == 'Boxplot':
        ##### Add the possibility to display several features on the same plot #####
        selections_analysis = st.multiselect('Ajouter ou supprimer une caractéristique::', df_test_normalize[features].columns.tolist(),
        df_test_normalize[features].columns.tolist()[0:5])
        ##### display boxplot #####
        ###### create in each df a columns to identifie them and use hue parameters ######
        df_test_normalize['data_origin'] = 'Tous les clients'
        
        ###### concatenate two df before drawing boxplot ######
        cdf = pd.concat([df_test_normalize[selections_analysis + ['data_origin']] ])
        
        ###### Create DataFrame from the selected client loan ID series ######
        df_loan = pd.DataFrame([df_test_normalize.loc[selected_credit, features].tolist()], columns=features)
        ###### using melt mehtod to adapt our concatenate dataframe to the format that we want (for displaying several features) with Seaborn ######
        cdf = pd.melt(cdf, id_vars='data_origin', var_name='Features')
        df_loan = pd.melt(df_loan[selections_analysis], var_name='Features')
        df_loan['data_origin'] = 'client selectionné'

        ###### plotting figure ######
        figure_boxplot = plt.figure(figsize=(4,2))
        ax = sns.boxplot(x = 'Features', y = 'value', hue='data_origin', data=cdf , showfliers=False, palette = 'tab10')
        sns.stripplot(x = 'Features', y = 'value', data = df_loan, hue = 'data_origin', palette=['yellow'], s=8, linewidth=1.5, edgecolor='black')
        plt.xticks(fontsize=3, rotation=45)
        plt.yticks(fontsize=3)
        plt.ylabel('Valeur')
        leg = plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ###### modify legend object for selected client loan ID to match graph style ######
        leg.legendHandles[-1].set_linewidth(1.5)
        leg.legendHandles[-1].set_edgecolor('black')
       
        fig = px.box(cdf, x = 'Features', y = 'value', points = False)
        fig.update_traces(quartilemethod="linear", jitter=0, col=1)
        
        #st.plotly_chart(fig)
        st.pyplot(figure_boxplot, clear_figure=True)
        
    if selected_anaysis_gh == 'Histogram':
        ##### Add the possibility to display several features on the same plot #####
        selections_analysis = st.multiselect('Ajouter ou supprimer une caractéristique::', df_test_normalize[features].columns.tolist(),
        df_test_normalize[features].columns.tolist()[0:5])
        ##### display boxplot #####
        ###### create in each df a columns to identifie them and use hue parameters ######
        df_test_normalize['data_origin'] = 'Tous les clients'
        
        ###### concatenate two df before drawing boxplot ######
        cdf = pd.concat([df_test_normalize[selections_analysis + ['data_origin']] ])
        
        ###### Create DataFrame from the selected client loan ID series ######
        df_loan = pd.DataFrame([df_test_normalize.loc[selected_credit, features].tolist()], columns=features)
        ###### using melt mehtod to adapt our concatenate dataframe to the format that we want (for displaying several features) with Seaborn ######
        cdf = pd.melt(cdf, id_vars='data_origin', var_name='Features')
        df_loan = pd.melt(df_loan[selections_analysis], var_name='Features')
        df_loan['data_origin'] = 'ID_prêt_client_selectionné'

        ###### plotting figure ######
        figure_boxplot = plt.figure(figsize=(4,2))
        ax = sns.histplot(x = 'Features', y = 'value', hue='data_origin', data=cdf, palette = 'tab10')
        #sns.stripplot(x = 'Features', y = 'value', data = df_loan, hue = 'data_origin', palette=['yellow'], s=8, linewidth=1.5, edgecolor='black')
        plt.xticks(fontsize=6, rotation=45)
        plt.yticks(fontsize=6)
        plt.ylabel('Valeur normalisée')
        leg = plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        fig = px.histogram(cdf, x = 'Features', y = 'value')
       
        st.plotly_chart(fig)
        #st.pyplot(figure_boxplot, clear_figure=True)



if features_importance:
    st.write('### Variables importantes dans la décision de prêt')
    st.write(f'Client: {selected_credit}')
    df_copie_no = df_copie.loc[:, ~df_copie.columns.str.match('TARGET')]
    bar_explainer = shap.Explainer(model, df_copie_no)
    bar_values = bar_explainer(df_copie_no, check_additivity=False)
    explainer = shap.TreeExplainer(model)
    #X_test_courant = df_copie_no.iloc[selected_credit]
    X_test_courant = df_copie_no[df_copie_no.index == int(selected_credit)]
    #X_test_courant = X_test_courant.drop('TARGET')
    X_test_courant_array = X_test_courant.values.reshape(1, -1)
    shap_values_courant = explainer.shap_values(X_test_courant_array)
    
    df_copie_noo = df_copie_no.reset_index()
    indice_cl = df_copie_noo.loc[df_copie_noo['index'] == selected_credit].index.item()
    plt.clf()
    # BarPlot du client courant
    shap.plots.bar(bar_values[indice_cl], max_display=40)
    fig_shap1 = plt.gcf()
    st.pyplot(fig_shap1)
    
    plt.clf()
    # Décision Plot
    shap.decision_plot(explainer.expected_value[1], shap_values_courant[1], X_test_courant)
    fig_shap2 = plt.gcf()
    # Plot the graph on the dashboard
    st.pyplot(fig_shap2)
    
    
    
    
    
    
if stats_generales:
	## Display input dataframe with multiselection of features for all the passenger list available (data are not standard scaled here!) ##
	st.write('### Informations générales:')
	st.write('Dimension des données: ' + str(df_test.shape[0]) + ' lignes et ' + str(df_test.shape[1]) + ' colonnes')
	selections = st.multiselect('Ajouter ou supprimer une caractéristique:', df_test.columns.tolist(),
 	df_test.columns.tolist()[0:10])
	st.dataframe(df_test.loc[:,selections].sort_index())
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
				values=df_test['NAME_EDUCATION_TYPE'].value_counts())])
	fig.update_traces(hoverinfo='label+percent', textinfo='label+percent', textfont_size=20,pull=[0, 0.05],
                  marker=dict(line=dict(color='#000000', width=1)))
	st.write("### Pourcentage par  type d\'education")
	st.plotly_chart(fig)



