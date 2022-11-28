import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt # matplotlib.use('agg') must be run before this sentence is executed
import seaborn as sns
import pickle
import time
import plotly.express as px
from zipfile import ZipFile
#from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
#from sklearn.neighbors import KNeighborsClassifier
import math
import re
import urllib
from urllib.request import urlopen
import json
import requests
import plotly.graph_objects as go
import shap

# -----------------------------------------------------------------------
def main():

    @st.cache()  # cache de la fonction pour exécution unique
    def load_data():
            #    PATH = '/Users/asus2/PROJET7/'

            # Données après préprocessing
        df = pd.read_csv('df_test_output.csv')

            # Données test avant features engineering
            # data_test = pd.read_csv(PATH + 'application_test.csv')
        #z = ZipFile("application_test.zip")
        #data_test = pd.read_csv(z.open('application_test.csv'), index_col='SK_ID_CURR', encoding='utf-8')
        data_test = pd.read_csv('application_test.csv')

            # Données train avant features engineering
        #z = ZipFile("application_test.zip")
        #data_train = pd.read_csv(z.open('application_train.csv'), index_col='SK_ID_CURR', encoding='utf-8')
        data_train = pd.read_csv('application_train.csv')

            # Description des features
        #z = ZipFile("HomeCredit_columns_description.zip")
        #description = pd.read_csv(z.open('HomeCredit_columns_description.csv'), usecols=['Row', 'Description'],\
         #                         index_col=0, encoding='unicode_escape')
        description = pd.read_csv('HomeCredit_columns_description.csv',usecols=['Row', 'Description'],\
                                  index_col=0, encoding='unicode_escape')

        return df, data_test, data_train, description

 # -----------------------------------------------------------------------#
   # @st.cache
    def load_model():
        '''loading the trained model'''
        pickle_in = open("RandomForestClassifier.pkl", 'rb')
        clf = pickle.load(pickle_in)
        return clf

#-----------------------------------------------------------#
   # @st.cache
    def get_client_info(data, id_client):
        client_info = data[data['SK_ID_CURR']==int(id_client)]
        return client_info
# -----------------------------------------------------------#

        #####------------------------------------------------------------------------------#####

            #----------Chargement des données--------------#

    df, data_test, data_train, description = load_data()

    features_dropped = ['Unnamed: 0', 'SK_ID_CURR', 'TARGET']
    relevant_features = [col for col in df if col not in features_dropped]

    print("df shape", df.shape)
    print("data_test shape", data_test.shape)
    print("data_train shape", data_train.shape)
    print("description shape", description.shape)
    print("df columns", df.columns)


        #-------- Chargement du modèle -------- #
    clf = load_model()
    print("model is:", clf)

#####---------------------------------------------------------------------------------#####
######------------------------------------ SIDEBAR -----------------------------------########


    LOGO_IMAGE = "logo.png"
    SHAP_GENERAL = "feature_importance_globale.png"

    with st.sidebar:
        st.header(" Prêt à dépenser")
        st.write("## ID Client")
        id_list = df["SK_ID_CURR"].tolist()
        id_client = st.selectbox("Veuillez entrer l'identifiant du client", id_list)
        st.write("## Actions à réaliser")
        credit_decision = st.checkbox("Afficher la décision de crédit")
        client_details = st.checkbox("Afficher les informations du client")
        #client_comparison = st.checkbox("Comparer aux autres clients")
        shap_general = st.checkbox("Afficher la feature importance globale")
        if (st.checkbox("Aide description des features")):
            list_features = description.index.to_list()
            list_features = list(dict.fromkeys(list_features))
            feature = st.selectbox('Sélectionner une variable', sorted(list_features))
            desc = description['Description'].loc[description.index == feature][:1]
            st.markdown('**{}**'.format(desc.iloc[0]))

#####------------------------------------------------------------#####
#####----------------------- HOME PAGE - MAIN CONTENT -----------#####

    # Display the title
    #st.title('Loan scoring dashboard')
    st.header("Moustapha ABDELLAHI - Data science P7")
# Titre principal

    html_temp = """
    <div style="background-color: tan; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Dashboard de scoring des clients </h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">
    Support d'aide à la décision  pour les gestionnaires de la relation client</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    with st.expander(" Quel est l'objectif de cette application ?"):
        st.write(
        "Dashboard interactif pour aider les gestionnaires de la relation client de l'entreprise **Prêt à dépenser** ")
        st.text('\n')
        st.write("**But**:  Etre transparent vis-à-vis des décisions d’octroi de crédit")
        st.image(LOGO_IMAGE)

    # Afficher l'ID Client sélectionné
    st.write("ID Client Sélectionné :", id_client)

    if (int(id_client) in id_list):
        client_info = get_client_info(data_test, id_client)

    # -------------------------------------------------------
    # Affichage de la décision de crédit
    # -------------------------------------------------------
    if (credit_decision):
        st.header('Scoring et décision du modèle')

    #------------------------------------ Appel de l'API--------------------------------- :
    #API_url = "http://127.0.0.1:5000/credit/" + str(id_client)
        with st.spinner('Chargement du score du client...'):
            ID=str(id_client)
            #response = json.loads(requests.get("http://127.0.0.1:5000/predict/{}".format(ID)).content)
            response = json.loads(requests.get("https://dashboard.heroku.com/apps/mabdellahi-api-p7/predict/{}".format(ID)).content)
            print(response)
            API_data=response
            #--------------------------------------

            classe_predite = API_data['prediction']
            if classe_predite == 1:
                decision = '❌ Crédit Refusé'
            else:
                decision = '✅ Crédit Accordé'

            proba = 1 - API_data['proba']

            client_score = round(proba * 100, 2)

            left_column, right_column = st.columns((1, 2))

            left_column.markdown('Risque de défaut: **{}%**'.format(str(client_score)))
            left_column.markdown('Seuil par défaut du modèle: **50%**')

            if classe_predite == 1:
                left_column.markdown(
                    'Decision: <span style="color:red">**{}**</span>'.format(decision), unsafe_allow_html=True)
            else:
                left_column.markdown('Decision: <span style="color:green">**{}**</span>'.format(decision), unsafe_allow_html=True)

            gauge = go.Figure(go.Indicator(
                mode="gauge+delta+number",
                title={'text': 'Pourcentage de risque de défaut'},
                value=client_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [None, 100]},
                       'steps': [
                           {'range': [0, 25], 'color': "lightgreen"},
                           {'range': [25, 50], 'color': "lightyellow"},
                           {'range': [50, 75], 'color': "orange"},
                           {'range': [75, 100], 'color': "red"},
                       ],
                       'threshold': {
                           'line': {'color': "black", 'width': 10},
                           'thickness': 0.8,
                           'value': client_score},

                       'bar': {'color': "black", 'thickness': 0.2},
                       },
                ))

            gauge.update_layout(width=450, height=250,
                                margin=dict(l=50, r=50, b=0, t=0, pad=4))

            right_column.plotly_chart(gauge)

    local_feature_importance = st.checkbox("Les variables ayant le plus contribué à la décision du modèle ?")
    if (local_feature_importance):
        shap.initjs()
        number = st.slider('Sélectionner le nombre de feautures à afficher ?',2, 20, 8)

        X = df[df['SK_ID_CURR'] == int(id_client)]
        X = X[relevant_features]

        fig, ax = plt.subplots(figsize=(15, 15))
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values[0], X, plot_type="bar",max_display=number, color_bar=False, plot_size=(8, 8))

        st.pyplot(fig)

    # -------------------------------------------------------
    # ----------------Affichage des informations du client ----------------###
    # -------------------------------------------------------

    personal_info_cols = {
        'CODE_GENDER': "GENRE",
        'DAYS_BIRTH': "AGE",
        'NAME_FAMILY_STATUS': "STATUT FAMILIAL",
        'CNT_CHILDREN': "NB ENFANTS",
        'FLAG_OWN_CAR': "POSSESSION VEHICULE",
        'FLAG_OWN_REALTY': "POSSESSION BIEN IMMOBILIER",
        'NAME_EDUCATION_TYPE': "NIVEAU EDUCATION",
        'OCCUPATION_TYPE': "EMPLOI",
        'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
        'AMT_INCOME_TOTAL': "REVENUS",
        'AMT_CREDIT': "MONTANT CREDIT",
        'NAME_CONTRACT_TYPE': "TYPE DE CONTRAT",
        'AMT_ANNUITY': "MONTANT ANNUITES",
        'NAME_INCOME_TYPE': "TYPE REVENUS",
        'EXT_SOURCE_1': "EXT_SOURCE_1",
        'EXT_SOURCE_2': "EXT_SOURCE_2",
        'EXT_SOURCE_3': "EXT_SOURCE_3",

    }

    default_list = \
        ["GENRE", "AGE", "STATUT FAMILIAL", "NB ENFANTS", "REVENUS", "MONTANT CREDIT"]
    numerical_features = ['DAYS_BIRTH', 'CNT_CHILDREN', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',\
                          'AMT_ANNUITY', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

    rotate_label = ["NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"]
    horizontal_layout = ["OCCUPATION_TYPE", "NAME_INCOME_TYPE"]

    if (client_details):
        st.header('Informations du client')

        with st.spinner('Chargement des informations du client...'):
             personal_info_df = client_info[list(personal_info_cols.keys())]
             personal_info_df['SK_ID_CURR'] = client_info['SK_ID_CURR']
             personal_info_df.rename(columns=personal_info_cols, inplace=True)
             personal_info_df["AGE"] = round(personal_info_df["AGE"] / 365 * (-1)).astype(int)
             personal_info_df["NB ANNEES EMPLOI"] = round(personal_info_df["NB ANNEES EMPLOI"] / 365 * (-1)).astype(int)
             filtered = st.multiselect("Choisir les informations à afficher", \
                                      options=list(personal_info_df.columns), \
                                      default=list(default_list))
             df_info = personal_info_df[filtered]
             df_info['SK_ID_CURR'] = client_info['SK_ID_CURR']
             df_info = df_info.set_index('SK_ID_CURR')

             st.table(df_info.astype(str).T)
             all_info = st.checkbox("Afficher toutes les informations (dataframe brute)")
             if (all_info):
                st.dataframe(client_info)

    # -------------------------------------------------------
    #### ---------------Comparaison du client choisi à d'autres clients--------------- #### -
    # -------------------------------------------------------

   # if (client_comparison):
   #     st.header(' Comparaison aux autres clients')
        # st.subheader("Comparaison avec l'ensemble des clients")
    #    with st.expander("Explication de la comparaison faite"):
    #        st.write(
    #            "Quand on sélectionne une variable, un graphique montrant la distribution de cette variable selon la classe (remboursé ou défaillant) est affiché avec une matérialisation du positionnement du client actuel.")

    #    with st.spinner('Chargement de la comparaison liée à la variable sélectionnée'):
    #        var = st.selectbox("Sélectionner une variable", list(personal_info_cols.values()))
    #        feature = list(personal_info_cols.keys()) \
    #            [list(personal_info_cols.values()).index(var)]


            #----------------------------------------------------------------------------
    #        if(st.checkbox("Afficher les clients similaires")):

                nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(df)
             #On récupère l'indice des plus proches voisins du client

                indices = nbrs.kneighbors(df[0:1])[1].flatten()
                st.dataframe(data_test.iloc[indices])


    # -------------------------------------------------------
    ######---------------- Afficher la feature importance globale ----------------###########
# -------------------------------------------------------
    if (shap_general):
        st.header('Feature importance globale')
        st.image('feature_importance_globale.png')
    #else:
       # st.markdown("**Identifiant non reconnu**")

if __name__ == '__main__':
    main()