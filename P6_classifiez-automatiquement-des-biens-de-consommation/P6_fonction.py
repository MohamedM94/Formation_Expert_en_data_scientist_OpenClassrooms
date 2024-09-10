import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import time

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords
import tensorflow_hub as hub

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE 
from sklearn import cluster, metrics

from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import time
from sklearn import cluster, metrics
from sklearn import manifold
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer

from matplotlib import offsetbox
from tensorflow.keras.utils import load_img, img_to_array
from matplotlib.image import imread
from skimage.transform import resize

import string
from wordcloud import WordCloud

# Nombre de données dupliquées 
def Nb_Data_Dupliq(data) :
    return print("Nombre de données dupliquées :",data.duplicated().sum())


def Graph_Pie_NaN(data):

    # Nombre de données non manquante
    Nb_data = data.count().sum()

    # Nombre de lignes
    Nb_lignes = data.shape[0] 
    print('Nombre de lignes :', Nb_lignes)
        
    # Nombre de colonnes
    Nb_colonnes = data.shape[1]
    print('Nombre de colonnes :',Nb_colonnes)
    
    #Nombre total de cellules manquantes
    print('Nombre total de cellules manquantes :', data.isna().sum().sum())

    # Nombre de données totale de ce jeu de données (colonnes*lignes)
    Nb_totale = (Nb_colonnes*Nb_lignes)

    #taux remplissage jeu de données
    rate_data = (Nb_data/Nb_totale)
    print("Le jeux de données est rempli à {:.2%}".format(rate_data))
    print("et il a {:.2%} de données manquantes".format(1-rate_data))
    print(" ")
    print(" ")

    # Pie Plot
    rates = [rate_data, 1 - rate_data]
    labels = ["Données", "NaN"]

    explode =(0,0.1) 
    colors = ['gold', 'pink']
    # Plot
    plt.figure(figsize=(12,15))
    plt.pie(rates, explode=explode, labels=labels, colors=colors, autopct='%.2f%%', shadow=True, textprops={'fontsize': 20})

    ttl=plt.title("Taux de remplissage du jeu de données", fontsize = 20)
    ttl.set_position([0.5, 0.85])

    plt.axis('equal')
    #ax.legend(labels, loc = "upper right", fontsize = 18)
    plt.tight_layout() 

    plt.show()
    
def missing_cells(df):
    '''Calcule le nombre de cellules manquantes sur le data set total.
    Keyword arguments:
    df -- le dataframe

    return : le nombre de cellules manquantes de df
    '''
    return df.isna().sum().sum()

def missing_cells_perc(df):
    '''Calcule le pourcentage de cellules manquantes sur le data set total.
    Keyword arguments:
    df -- le dataframe

    return : le pourcentage de cellules manquantes de df
    '''
    return df.isna().sum().sum()/(df.size)
    
def valeurs_manquantes(df):
    '''Prend un data frame en entrée et créer en sortie un dataframe contenant
    le nombre de valeurs manquantes et leur pourcentage pour chaque variables.
    Keyword arguments:
    df -- le dataframe

    return : dataframe contenant le nombre de valeurs manquantes et
    leur pourcentage pour chaque variable
    '''
    tab_missing = pd.DataFrame(columns=['Variable',
                                        'Missing values',
                                        'Missing (%)'])
    tab_missing['Variable'] = df.columns
    missing_val = list()
    missing_perc = list()

    for var in df.columns:
        nb_miss = missing_cells(df[var])
        missing_val.append(nb_miss)
        perc_miss = missing_cells_perc(df[var])
        missing_perc.append(perc_miss)

    tab_missing['Missing values'] = list(missing_val)
    tab_missing['Missing (%)'] = list(missing_perc)
    return tab_missing

def bar_missing(df):
    '''Affiche le barplot présentant le nombre de données présentes par variable.
    Keyword arguments:
    df -- le dataframe
    '''
    msno.bar(df,color="dodgerblue")
    plt.title('Nombre de données présentes par variable', size=15)
    plt.show()
    
def Liste_Colonne_Seuil_NaN(data,Seuil):  
    '''#Afficher les colonnes ayant plus d'un seuil 40 % valeur manquantes
    '''
    Taux_Data_Manquantes_Colonnes = (data.isnull().mean()*100).sort_values(ascending=False).reset_index()
    Taux_Data_Manquantes_Colonnes.columns = ['Variable','Taux_Data_NaN'] 
    Taux_NaN_Colonne_Seuil = Taux_Data_Manquantes_Colonnes[Taux_Data_Manquantes_Colonnes.Taux_Data_NaN >=Seuil] 
    print("Nombre de colonnes ayant plus de 30% valeur manquantes est :",len(Taux_NaN_Colonne_Seuil))
    print("La liste de colonnes ayant plus de 30% de valeurs manquantes est :",Taux_NaN_Colonne_Seuil["Variable"].tolist())
    return Taux_NaN_Colonne_Seuil


def categorical_distribution(feature_series, ordinal=False):
    """Function plotting the bar-plot and pie-plot (as subplots) for 
    a distribution of categorical features."""

    # importing libraries
    import matplotlib.pyplot as plt

    # filtering non-null data for the feature
    mask = feature_series.notnull()
    data_view = feature_series[mask]

    # Setting the data to plot
    x = data_view

    # Set frequencies and labels, sorting by index
    if ordinal == True:
        labels = list(x.value_counts().sort_index().index.astype(str))
        frequencies = x.value_counts().sort_index()

    elif ordinal == False:
        labels = list(x.value_counts().sort_values(
            ascending=False).index.astype(str))
        frequencies = x.value_counts().sort_values(ascending=False)

    # Graphical properties of the main figure
    fig = plt.figure(figsize=(14, 6))

    plt.suptitle("Distribution des catégories dans le dataset", size=25)

    # Main graphical properties of the first subplot (histogram)
    ax1 = plt.subplot(121)
    ax1.set_xlabel(" ", fontsize=24)
    ax1.set_ylabel("Fréquences", fontsize=24)
    ax1.set_xticklabels(labels, rotation='vertical',
                        horizontalalignment="right")

    # Main graphical properties of the second subplot (pieplot)
    ax2 = plt.subplot(122)
    ax2.set_xlabel("Fréquences relatives", fontsize=20)

    # plotting the plots
    ax1.bar(labels, frequencies, color=[
            'blue','red','green','chocolate', 'gold','aqua','tan'])
    ax2.pie(frequencies, colors=['blue','red','green','chocolate', 'gold','aqua','tan'],
            autopct='%1.2f%%',
            shadow=False,
            ) 
    
    ax2.legend(labels)
    plt.show()
    return fig

def lemma_fct(list_words) :
    '''Lemmatizer'''
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w

def transform_bow_lem_fct(text, stop_w) :
    '''Fonction de préparation du texte pour le bag of words avec lemmatization.'''
    word_tokens = tokenizer_fct(text)
    sw = stop_word_filter_fct(word_tokens, stop_w)
    lw = lower_start_fct(sw)
    lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lem_w)
    return transf_desc_text

def tokenizer_fct(sentence) :
    ''' Tokenisation du texte en argument.'''
    word_tokens = word_tokenize(sentence)
    return word_tokens

def stop_word_filter_fct(list_words, stop_w) :
    ''' Filtrage de la liste de mots avec les stopwords renseignés en argument.'''
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2

def stop_word_filter_fct(list_words, stop_w) :
    ''' Filtrage de la liste de mots avec les stopwords renseignés en argument.'''
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2

def lower_start_fct(list_words) :
    '''Transformation de la liste de mots en minuscule.'''
    lw = [w.lower() for w in list_words]
    return lw

def transform_bow_fct(text, stop_w) :
    '''Fonction de préparation du texte pour le bag of words (Countvectorizer et Tf_idf, Word2Vec)'''
    word_tokens = tokenizer_fct(text)
    sw = stop_word_filter_fct(word_tokens, stop_w)
    lw = lower_start_fct(sw)
    transf_desc_text = ' '.join(lw)
    return transf_desc_text
def stem_fct(list_words) :
    '''Lemmatizer'''
    stemmer = PorterStemmer()
    stem_w = [stemmer.stem(w) for w in list_words]
    return stem_w

def transform_bow_stem_fct(text, stop_w) :
    '''Fonction de préparation du texte pour le bag of words avec stemming.'''
    word_tokens = tokenizer_fct(text)
    sw = stop_word_filter_fct(word_tokens, stop_w)
    lw = lower_start_fct(sw)
    stem_w = stem_fct(lw)    
    transf_desc_text = ' '.join(stem_w)
    return transf_desc_text

def transform_dl_fct(text) :
    '''Fonction de préparation du texte pour le Deep learning (USE et BERT).'''
    word_tokens = tokenizer_fct(text)
    lw = lower_start_fct(word_tokens)
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

def ARI_fct_tsne(features, liste_cat, label_true) :
    '''Calcul Tsne, détermination des clusters et calcul ARI entre les vraies catégories et n° de clusters.'''
    time1 = time.time()
    nb_clusters = len(liste_cat)
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=2000, 
                init='random', learning_rate=200, random_state=0)
    X_tsne = tsne.fit_transform(features)
    
    # Détermination des clusters à partir des données après Tsne 
    cls = KMeans(n_clusters=nb_clusters, n_init=100, random_state=0)
    cls.fit(X_tsne)
    ARI = np.round(adjusted_rand_score(label_true, cls.labels_),4)
    time2 = np.round(time.time() - time1,0)
    print("ARI : ", ARI, "time : ", time2)
    
    return ARI, X_tsne, cls.labels_


def TSNE_visu_fct(X_tsne, liste_cat, label_true, labels, ARI) :
    '''Visualisation du Tsne selon les vraies catégories et selon les clusters.'''
    fig = plt.figure(figsize=(15,6))
    
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=label_true, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=liste_cat, loc="best", title="Categorie")
    plt.title('Représentation des produits par catégories réelles')
    
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(labels), loc="best", title="Clusters")
    plt.title('Représentation des produits par clusters')
    
    plt.show()
    print("ARI : ", ARI)
    

def plot_TSNE_images(X, df, path):
    '''Affiche le graphe TSNE en 2 dimensions avec les images à la place des points.'''
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(15, 15))
    ax = plt.subplot(111)

    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])
        for i in range(df.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 5e-4:
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            props={'boxstyle':'round', 'edgecolor':'white'}
            
            image = imread(path + df['image'][i])
            image = resize(image, (230, 230)) 

            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(image,zoom=0.1),
                                                X[i], bboxprops=props)
            ax.add_artist(imagebox)

def Eboulis(pca):
    '''Réalise un éboulis de valeurs propres'''
    scree = pca.explained_variance_ratio_*100
    scree_cum = scree.cumsum()
    
    plt.figure(figsize=(10,5))
    plt.bar(np.arange(len(scree)) + 1, scree)
    plt.plot(np.arange(len(scree)) + 1, scree_cum,c="red",marker='o')
    plt.xlabel("Rang de l'axe d'inertie")
    plt.ylabel("Pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)


def eboulis(pca,y,x):
    '''Réalise un éboulis de valeurs propres'''
    scree = pca.explained_variance_ratio_*100
    scree_cum = scree.cumsum()
    
    plt.figure(figsize=(10,5))
    plt.bar(np.arange(len(scree)) + 1, scree)
    plt.plot(np.arange(len(scree)) + 1, scree_cum,c="red",marker='o')
    plt.xlabel("Rang de l'axe d'inertie")
    plt.ylabel("Pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    
    plt.axhline(y, linestyle="--", 
            color="green",
            xmax=0.67,
            linewidth=1)
    plt.axvline(x, linestyle="--", 
            color="green",
            ymax=1.48,
            linewidth=1)
    
    
    
    plt.show(block=False)
    
    
def conf_mat_transform(y_true, y_pred, corresp):
    '''Créer une matrice de confusion en fonction de la correspondance donnée en paramètres.'''
    conf_mat = metrics.confusion_matrix(y_true,y_pred)
    
    if corresp == 'argmax':
        corresp = np.argmax(conf_mat, axis=0)
    
    print ("Correspondance des clusters : ", corresp)
    labels = pd.Series(y_true, name="y_true").to_frame()
    labels['y_pred'] = y_pred
    labels['y_pred_transform'] = labels['y_pred'].apply(lambda x : corresp[x]) 
    
    return labels['y_pred_transform']

def Fc_WordCloud(text) :
        
        # Create and generate a word cloud image:
        wordcloud = WordCloud().generate(text)

        # Display the generated image:
        fig = plt.figure(figsize=(10, 15))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        
        
def display_topics(model, feature_names, no_top_words):
    '''Affiche les topics trouvés.'''
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(" ".join([feature_names[i] 
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        
        
def display(imglist,size):
    cols = len(imglist)
    fig = plt.figure(figsize=(size*cols,size*cols))
    for i in range(0,cols):
        a = fig.add_subplot(1, cols, i+1)
        subfig = plt.imshow(imglist[i], cmap='gray')
        #plt.get_cmap('gray')
        subfig.axes.get_xaxis().set_visible(False)
        subfig.axes.get_yaxis().set_visible(False)
        
def ARI_fct(features, y_cat_num):
    """
    Calcul Tsne, détermination des clusters et
    calcul ARI entre vrais catégorie et n° de clusters
    """
    features_std = StandardScaler().fit_transform(features)
    print("Dimensions dataset avant réduction PCA : ", features_std.shape)
    pca = PCA(n_components=0.99)
    feat_pca= pca.fit_transform(features_std)
    print("Dimensions dataset après réduction PCA : ", feat_pca.shape)

    time1 = time.time()
    num_labels = 7  # On dispose de 7 catégories
    # Projection 2 dimensions
    tsne = manifold.TSNE(n_components=2,
                         perplexity=30,
                         n_iter=2000,
                         init='random',
                         learning_rate=200,
                         random_state=42)
    X_tsne = tsne.fit_transform(feat_pca)

    # Détermination des clusters à partir des données après Tsne
    cls = cluster.KMeans(n_clusters=num_labels,
                         n_init=100,
                         random_state=42)
    cls.fit(X_tsne)
    ARI = np.round(metrics.adjusted_rand_score(y_cat_num, cls.labels_), 4)
    time2 = np.round(time.time() - time1, 0)
    print("ARI : ", ARI, "time : ", time2)

    return ARI, X_tsne, cls.labels_



# Fonction de préparation des sentences
def bert_inp_fct(sentences, bert_tokenizer, max_length) :
    input_ids=[]
    token_type_ids = []
    attention_mask=[]
    bert_inp_tot = []

    for sent in sentences:
        bert_inp = bert_tokenizer.encode_plus(sent,
                                              add_special_tokens = True,
                                              max_length = max_length,
                                              padding='max_length',
                                              return_attention_mask = True, 
                                              return_token_type_ids=True,
                                              truncation=True,
                                              return_tensors="tf")
    
        input_ids.append(bert_inp['input_ids'][0])
        token_type_ids.append(bert_inp['token_type_ids'][0])
        attention_mask.append(bert_inp['attention_mask'][0])
        bert_inp_tot.append((bert_inp['input_ids'][0], 
                             bert_inp['token_type_ids'][0], 
                             bert_inp['attention_mask'][0]))

    input_ids = np.asarray(input_ids)
    token_type_ids = np.asarray(token_type_ids)
    attention_mask = np.array(attention_mask)
    
    return input_ids, token_type_ids, attention_mask, bert_inp_tot


# Fonction de création des features
def feature_BERT_fct(model, model_type, sentences, max_length, b_size, mode='HF') :
    batch_size = b_size
    batch_size_pred = b_size
    bert_tokenizer = AutoTokenizer.from_pretrained(model_type)
    time1 = time.time()

    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        input_ids, token_type_ids, attention_mask, bert_inp_tot = bert_inp_fct(sentences[idx:idx+batch_size], 
                                                                      bert_tokenizer, max_length)
        
        if mode=='HF' :    # Bert HuggingFace
            outputs = model.predict([input_ids, attention_mask, token_type_ids], batch_size=batch_size_pred)
            last_hidden_states = outputs.last_hidden_state

        if mode=='TFhub' : # Bert Tensorflow Hub
            text_preprocessed = {"input_word_ids" : input_ids, 
                                 "input_mask" : attention_mask, 
                                 "input_type_ids" : token_type_ids}
            outputs = model(text_preprocessed)
            last_hidden_states = outputs['sequence_output']
             
        if step ==0 :
            last_hidden_states_tot = last_hidden_states
            last_hidden_states_tot_0 = last_hidden_states
        else :
            last_hidden_states_tot = np.concatenate((last_hidden_states_tot,last_hidden_states))
    
    features_bert = np.array(last_hidden_states_tot).mean(axis=1)
    
    time2 = np.round(time.time() - time1,0)
    print("temps traitement : ", time2)
     
    return features_bert, last_hidden_states_tot

def feature_USE_fct(sentences, b_size) :
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    batch_size = b_size
    time1 = time.time()

    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        feat = embed(sentences[idx:idx+batch_size])

        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))

    time2 = np.round(time.time() - time1,0)
    return features

def preprocess_text_lem_stem(doc,
                    rejoin=True,
                    min_len_word=3,
                    force_is_alpha=True):
    """ preprocess_text est une fonction qui fait le prétraitement
    d'un document doc passer en paramètre. Elle retourne un text en
    minuscule sans les chiffres ni les stop-words avec lem et stem.
    """
    # lower
    doc = doc.lower().strip()

    # tokenize
    tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
    raw_tokens_list = tokenizer.tokenize(doc)

    # classics stopwords
    stop_words = set(stopwords.words('english'))
    cleaned_tokens_list = [w for w in raw_tokens_list if w not in stop_words]

    # no more len words
    more_than_N = [w for w in cleaned_tokens_list if len(w) >= min_len_word]

    # only alpha chars
    if force_is_alpha:
        alpha_tokens = [w for w in more_than_N if w.isalpha()]
    else:
        alpha_tokens = cleaned_tokens_list

    # lemm
    trans = WordNetLemmatizer()
    lemm_text = [trans.lemmatize(i) for i in alpha_tokens]
    # stem
    trans = PorterStemmer()
    stem_text = [trans.stem(i) for i in lemm_text]

    final = stem_text
    # manage return type
    if rejoin:
        return " ".join(final)

    return final
def describe_dataset(source_files):
    '''
        Outputs a presentation pandas dataframe for the dataset.

        Parameters
        ----------------
        sourceFiles     : dict with :
                            - keys : the names of the files
                            - values : a list containing two values :
                                - the dataframe for the data
                                - a brief description of the file

        Returns
        ---------------
        presentation_df : pandas dataframe :
                            - a column "Nom du fichier" : the name of the file
                            - a column "Nb de lignes"   : the number of rows per file
                            - a column "Nb de colonnes" : the number of columns per file
                            - a column "Description"    : a brief description of the file
    '''

    print("Les données se décomposent en {} fichier(s): \n".format(len(source_files)))

    filenames = []
    files_nb_lines = []
    files_nb_columns = []
    files_descriptions = []

    for filename, file_data in source_files.items():
        filenames.append(filename)
        files_nb_lines.append(len(file_data[0]))
        files_nb_columns.append(len(file_data[0].columns))
        files_descriptions.append(file_data[1])

    # Create a dataframe for presentation purposes
    presentation_df = pd.DataFrame({'Nom du fichier':filenames,
                                    'Nb de lignes':files_nb_lines,
                                    'Nb de colonnes':files_nb_columns,
                                    'Description': files_descriptions})

    presentation_df.index += 1

    return presentation_df

def afficher_image_histopixel(image, titre):
    '''
    Afficher côte à côte l'image et l'histogramme de répartiton des pixels.
    Parameters
    ----------
    image : image à afficher, obligatoire.
    Returns
    -------
    None.
    '''
    plt.figure(figsize=(40, 10))
    plt.subplot(131)
    plt.grid(False)
    plt.title(titre, fontsize=30)
    plt.imshow(image, cmap='gray')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.subplot(132)
    plt.title('Histogramme de répartition des pixels', fontsize=30)
    hist, bins = np.histogram(np.array(image).flatten(), bins=256)
    plt.bar(range(len(hist[0:255])), hist[0:255])
    plt.xlabel('Niveau de gris', fontsize=30)
    plt.ylabel('Nombre de pixels', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.subplot(133)
    plt.title('Histogramme cumulé des pixels', fontsize=30)
    plt.hist(np.array(image).flatten(), bins=range(256), cumulative=True, 
                           histtype='stepfilled')
    plt.xlabel('Niveau de gris', fontsize=24)
    plt.ylabel('Fréquence cumulée de pixels', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.show()
