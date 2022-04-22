

# In[4]:


#!/usr/bin/python

import _pickle as cPickle
import numpy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif



def preprocess(words_file = "C:\\Users\\Malek\\Downloads\\Email-Classification-master\\word_data.pkl", authors_file="C:\\Users\\Malek\\Downloads\\Email-Classification-master\\email_authors.pkl"):
    """ 
        cette fonction prend une liste prédéfinie de textes d'e-mails (par défaut word_data.pkl)
        
et les auteurs correspondants (par défaut email_authors.pkl) et effectuer un certain nombre d'étapes de prétraitement :
            -- se divise en ensembles de training/test (10 % de test)
            -- vectorise into tfidf matrix
            -- sélectionne/conserve les fonctionnalités les plus utiles

       après cela, les fonctionnalités et les Labels sont placées dans des tableaux numpy, qui fonctionnent bien avec les fonctions sklearn

4 objets sont retournés :
            -- training/testing features
            -- training/testing labels

    """

    ### les mots (caractéristiques) et les auteurs (étiquettes), déjà largement prétraités
    ### ce prétraitement sera répété dans le mini-projet d'apprentissage de texte
    authors_file_handler = open(authors_file, "rb")
    authors = cPickle.load(authors_file_handler)
    authors_file_handler.close()

    words_file_handler = open(words_file, "rb")
    word_data = cPickle.load(words_file_handler)
    words_file_handler.close()

    ### test_size est le pourcentage d'événements affectés à l'ensemble de test
    ### (le reste va en formation)
    features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)



    ### text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)



    ### sélection de fonctionnalités, car le texte est de très haute dimension et 
    ### can be really computationally chewy as a result
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()

    ### infos sur les données
    print("\nno. of Chris training emails:", sum(labels_train))
    print("\nno. of Sara training emails:", len(labels_train)-sum(labels_train))
    
    return features_train_transformed, features_test_transformed, labels_train, labels_test


# In[ ]:





# In[ ]:





# In[ ]:




