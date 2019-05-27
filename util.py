# Standard boiler plate imports
import pandas as pd
import numpy as np
import re
# import csv
# from timeit import default_timer as timer

# NLTK
# import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# from nltk.tokenize import TreebankWordTokenizer
from nltk import word_tokenize

# SKLEARN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# GRADIENT-BOOSTING
import lightgbm as lgb

# KERAS
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
# import keras
# from keras.utils import get_file, plot_model
# from keras import preprocessing
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Masking
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.optimizers import SGD, Adam
# from keras.utils import np_utils
# from keras import regularizers
# import keras.backend as K
# import keras_metrics

# HPO
# from hyperopt import hp
# from hyperopt.pyll.stochastic import sample
# from hyperopt import rand, tpe
# from hyperopt import Trials
# from hyperopt import fmin
# from hyperopt import STATUS_OK


"""
import os
import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

# from datetime import datetime, timedelta
import time
import random
import csv
import json
import re
from timeit import default_timer as timer

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from nltk import word_tokenize


# SKLEARN
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, PolynomialFeatures, QuantileTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# GRADIENT-BOOSTING
import lightgbm as lgb
import xgboost as xgb

# KERAS, NLP
import keras
from keras.utils import get_file, plot_model
from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Flatten, LSTM, Dense, Dropout, Masking, Activation, Input, Lambda
from keras.layers import Bidirectional, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
# from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import regularizers
import keras.backend as K
import keras_metrics
from keras.utils.np_utils import to_categorical
"""

stopwords = set(stopwords.words('english'))


def points_to_class(points):
    if points in range(0, 83):
        return 0
    elif points in range(83, 87):
        return 1
    elif points in range(87, 90):
        return 2
    elif points in range(90, 94):
        return 3
    else:
        return 4


def clean_description(text):
    text = word_tokenize(text.lower())
    text = ' '.join(token for token in text if token not in stopwords and token.isalpha())
    return text


# function for data preprocessing
def load_preprocess(labelenc=False):

    # Read in first data set
    data1 = pd.read_csv("./wine-reviews/winemag-data_first150k.csv", index_col=False)
    data1 = data1.drop(['Unnamed: 0'], axis=1)

    # Read in second data set
    data2 = pd.read_csv("./wine-reviews/winemag-data-130k-v2.csv", index_col=False)
    data2 = data2.drop(['Unnamed: 0', 'taster_name', 'taster_twitter_handle', 'title'], axis=1)

    # combine the two datasets and re-shuffle
    data = pd.concat([data1, data2])
    data = data.sample(frac=1, random_state=42)

    # remove duplicates: DROP duplicate values, keeping only one row
    data_no_dupl = data.drop_duplicates(subset="description", keep='first', inplace=False)

    # re-set the index to avoid ambiguities (same index from data set 1 and 2)
    data_no_dupl = data_no_dupl.reset_index().drop(['index'], axis=1)

    # target binning
    data_no_dupl["rating"] = data_no_dupl["points"].apply(points_to_class)
    data_no_dupl.drop(columns=["points"], inplace=True)

    # clean text description
    data_no_dupl["cleaned_text"] = data_no_dupl["description"].apply(clean_description)

    # drop high NaN features
    cols_drop = ['designation', 'region_1', 'region_2']
    data_no_dupl.drop(columns=cols_drop, inplace=True)

    # drop high-cardinality features
    cols_drop = ['province', 'variety', 'winery']
    data_no_dupl.drop(columns=cols_drop, inplace=True)

    # label encoding of categorical country feature with nan as sep category
    if labelenc:
        data_no_dupl['country'] = data_no_dupl['country'].fillna('NaN')
        encoder = LabelEncoder()
        data_no_dupl['country'] = encoder.fit_transform(data_no_dupl['country'])

    # re-arrange columns
    cols_ordered = ['rating', 'price', 'country', 'description', 'cleaned_text']
    data_no_dupl = data_no_dupl[cols_ordered]

    # one hot encoding of target variable
    Y = data_no_dupl['rating']
    y = to_categorical(Y)
    target = pd.DataFrame(data=y, index=data_no_dupl.index)
    data_no_dupl = target.join(data_no_dupl)
    data_no_dupl = data_no_dupl.rename(columns={0: "rating_0", 1: "rating_1",
                                                2: "rating_2", 3: "rating_3",
                                                4: "rating_4"})

    # split into train and dev set
    data_train, data_dev = train_test_split(data_no_dupl, test_size=0.2, random_state=0)

    # imputation for numerical price feature
    mean_price = data_train['price'].mean()
    data_train['price'].fillna(mean_price, inplace=True)
    data_dev['price'].fillna(mean_price, inplace=True)

    # get text only data set
    # cols_text = ['rating', 'description']
    # text_data = data_no_dupl[cols_text]
    # other_data = data_no_dupl.drop(columns=cols_text)

    # save datasets to disk
    data_train.to_csv('data_train_raw.csv', header=True)
    data_dev.to_csv('data_dev_raw.csv', header=True)

    return data_train, data_dev


def preprocess_rnn(data_train, data_dev, max_words=10000, maxlen=50):
    """
    preprocessing for RNN network using the Keras Tokenizer
    """

    # set number of words to consider as features
    max_words = max_words
    # Cut off the text after this max number of words (among the max_features most common words)
    maxlen = maxlen

    # features and target
    cols_target = ['rating_0', 'rating_1', 'rating_2', 'rating_3', 'rating_4']
    x_train = data_train['description']
    x_dev = data_dev['description']
    y_train = data_train[cols_target]
    y_dev = data_dev[cols_target]

    # define and fit Tokenizer object to TRAIN text
    tokenizer = Tokenizer(num_words=max_words, split=' ')
    tokenizer.fit_on_texts(x_train)

    # recover word index that was computed
    word_index = tokenizer.word_index
    idx_word = tokenizer.index_word

    # turn strings into lists of integer indices
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_dev_seq = tokenizer.texts_to_sequences(x_dev)

    # PADDING of sequences
    x_train_seq = pad_sequences(x_train_seq, maxlen=maxlen)
    x_dev_seq = pad_sequences(x_dev_seq, maxlen=maxlen)

    # create data frame after padding with target as first column
    # training set
    features_train = pd.DataFrame(data=x_train_seq, index=x_train.index)
    target_train = pd.DataFrame(data=y_train, index=x_train.index)
    model_data_train = target_train.join(features_train, lsuffix='_target')

    # dev set
    features_dev = pd.DataFrame(data=x_dev_seq, index=x_dev.index)
    target_dev = pd.DataFrame(data=y_dev, index=x_dev.index)
    model_data_dev = target_dev.join(features_dev, lsuffix='_target')

    # save datasets to disk
    model_data_train.to_csv('data_train_rnn.csv', header=True)
    model_data_dev.to_csv('data_dev_rnn.csv', header=True)

    return model_data_train, model_data_dev, word_index, idx_word, tokenizer, max_words


def preprocess_rnn2(x_train, x_dev, y_train, y_dev, max_words=10000, maxlen=50):
    """
    preprocessing for RNN network using the Keras Tokenizer
    """

    # set number of words to consider as features
    max_words = max_words
    # Cut off the text after this max number of words (among the max_features most common words)
    maxlen = maxlen

    # define and fit Tokenizer object to TRAIN text
    tokenizer = Tokenizer(num_words=max_words, split=' ')
    tokenizer.fit_on_texts(x_train)

    # recover word index that was computed
    word_index = tokenizer.word_index
    idx_word = tokenizer.index_word

    # turn strings into lists of integer indices
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_dev_seq = tokenizer.texts_to_sequences(x_dev)

    # PADDING of sequences
    x_train_seq = pad_sequences(x_train_seq, maxlen=maxlen)
    x_dev_seq = pad_sequences(x_dev_seq, maxlen=maxlen)

    # create data frame after padding with target as first column
    # training set
    features_train = pd.DataFrame(data=x_train_seq, index=x_train.index)
    target_train = pd.DataFrame(data=y_train, index=x_train.index)
    model_data_train = target_train.join(features_train, lsuffix='_target')

    # dev set
    features_dev = pd.DataFrame(data=x_dev_seq, index=x_dev.index)
    target_dev = pd.DataFrame(data=y_dev, index=x_dev.index)
    model_data_dev = target_dev.join(features_dev, lsuffix='_target')

    return model_data_train, model_data_dev, word_index, idx_word, tokenizer


# define cleaning function
def clean_reviews(data):
    """
    function to clean dataset
    """

    corpus = []
    for i in data.index:
        review = re.sub('[^a-zA-Z]', ' ', data['description'][i])
        # review = re.sub('[^a-zA-Z]', ' ', str(data['description'][i]))
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        # review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = [ps.stem(word) for word in review if not word in stopwords]
        review = ' '.join(review)
        corpus.append(review)

    corpus = pd.DataFrame(corpus, columns=['description'], index=data.index)
    result = corpus.join(data[['rating']])

    return result


def tfidfVec(data):
    """
    function to apply TFIDF
    """

    target = data['rating']

    tfidf = TfidfVectorizer()
    tfidf.fit(data['description'])
    features = tfidf.transform(data['description'])

    return features, target


def preprocess_cls(data_train, data_dev, min_df=0, max_df=1, max_features=10000, save_data=False):
    """
    function to preprocess for classical ML classifier (RF, XGB, ...)
    """

    # features and target
    cols = ['rating', 'description']
    text_target_train = data_train[cols]
    text_target_dev = data_dev[cols]

    # text cleaning
    text_target_train = clean_reviews(text_target_train)
    text_target_dev = clean_reviews(text_target_dev)

    # text vectorization with BOW
    tfidf = TfidfVectorizer(stop_words=stopwords,
                            min_df=min_df,
                            max_df=max_df,
                            max_features=max_features)
    tfidf.fit(text_target_train['description'])
    text_features_train = tfidf.transform(text_target_train['description'])
    text_features_dev = tfidf.transform(text_target_dev['description'])
    text_features_train = pd.DataFrame(text_features_train.toarray(),
                                       columns=tfidf.get_feature_names(),
                                       index=text_target_train.index)
    text_features_dev = pd.DataFrame(text_features_dev.toarray(),
                                     columns=tfidf.get_feature_names(),
                                     index=text_target_dev.index)

    text_target_train = pd.DataFrame(data=text_target_train['rating'],
                                     index=text_target_train.index).join(text_features_train)
    text_target_dev = pd.DataFrame(data=text_target_dev['rating'],
                                   index=text_target_dev.index).join(text_features_dev)

    # other numerical and categorical features
    feats_other = ['price', 'country']
    x_train_feat = data_train[feats_other]
    x_dev_feat = data_dev[feats_other]

    model_data_train = text_target_train.join(x_train_feat)
    model_data_dev = text_target_dev.join(x_dev_feat)

    # save data sets to disk
    if save_data:
        # model_data_train.to_csv('data_train_cls.csv', header=True)
        # model_data_dev.to_csv('data_dev_cls.csv', header=True)
        model_data_train.to_csv('data_train_cls.csv.gz', header=True, compression='gzip')
        model_data_train.to_csv('data_dev_cls.csv.gz', header=True, compression='gzip')

    return model_data_train, model_data_dev


def accuracy_summary(pipeline, X_train, X_test, y_train, y_test):
    """
    function to fit model, make predictions on dev set and print accuracy
    """
    # training
    sentiment_fit = pipeline.fit(X_train, y_train)
    # inference: predict
    y_pred = sentiment_fit.predict(X_test)
    # evaluate test set accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print accuracy
    print("Accuracy score on dev set: {0:.2f}%".format(accuracy * 100))

    return accuracy


def accuracy_summary2(pipeline, X_train, X_test, y_train, y_test):
    """
    function to fit model, make predictions on dev set and print accuracy
    return fitted model
    """
    # training
    sentiment_fit = pipeline.fit(X_train, y_train)
    # inference: predict
    y_pred = sentiment_fit.predict(X_test)
    # evaluate test set accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print accuracy
    print("Accuracy score on dev set: {0:.2f}%".format(accuracy * 100))

    return sentiment_fit, accuracy


def build_glove_embedding(word_index, max_words):
    """
    build GloVe embedding dictionary with embedding dimension=300
    """
    embedding_dim = 300

    # read-in embedding stored in txt file and store in dictionary "embeddings_index"
    file = open('glove.840B.300d.txt', encoding="utf8")

    embeddings_index = {}
    for line in file:
        values = line.split()
        # word = values[0]
        # coefs = np.asarray(values[1:], dtype='float32')
        word = ''.join(values[:-embedding_dim])
        coefs = np.asarray(values[-embedding_dim:], dtype='float32')
        embeddings_index[word] = coefs

    file.close()

    # construct embedding matrix
    not_found = 0

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            # words not found in the glove embedding index will be all zeros
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                not_found += 1

    print('There were %s words without pre-trained embeddings.' % not_found)

    # save embedding matrix
    filename = 'glove_embedding_matrix'
    joblib.dump(embedding_matrix, filename)

    return embedding_matrix


def find_closest(query, embedding_matrix, word_idx, idx_word, n=10):
    """
    Find closest words to a query word in embeddings
    """

    idx = word_idx.get(query, None)
    # Handle case where query is not in vocab
    if idx is None:
        print('%s not found in vocab.' % query)
        return
    else:
        vec = embedding_matrix[idx]
        # Handle case where word doesn't have an embedding
        if np.all(vec == 0):
            print('%s has no pre-trained embedding.' % query)
            return
        else:
            # Calculate distance between vector and all others
            dists = np.dot(embedding_matrix, vec)

            # Sort indexes in reverse order
            idxs = np.argsort(dists)[::-1][:n]
            sorted_dists = dists[idxs]
            closest = [idx_word[i] for i in idxs]

    print('Query: %s \n' % query)
    # max_len = max([len(i) for i in closest])
    # Print out the word and cosine distances
    for word, dist in zip(closest, sorted_dists):
        print('Word:', word, 'with cosine Similarity:', round(dist, 4))


# function to build and compile RNN model
def make_rnn_model(num_words,
                   embedding_matrix,
                   lstm_layers=1,
                   lstm_cells=100,
                   number_dense=50,
                   number_out=5,
                   trainable=False,
                   bi_direc=False,
                   dropout=0.3):
    """
    Make a word-level recurrent neural network with option for pretrained embeddings
    and varying numbers of LSTM cell layers.
    """

    # use Keras Sequential class
    model = Sequential()

    # Map words to an embedding
    if not trainable:
        model.add(Embedding(input_dim=num_words,
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False,
                            mask_zero=True))
        model.add(Masking(mask_value=0.0))
    else:
        model.add(Embedding(input_dim=num_words,
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=True))

    # Add multiple LSTM layers if desired (going DEEP)
    if lstm_layers > 1:
        for i in range(lstm_layers - 1):
            if bi_direc:
                model.add(Bidirectional(LSTM(lstm_cells, return_sequences=True,
                                             dropout=dropout, recurrent_dropout=dropout)))
            else:
                model.add(LSTM(lstm_cells, return_sequences=True,
                               dropout=dropout, recurrent_dropout=dropout))

    # Add final LSTM layer
    if bi_direc:
        model.add(Bidirectional(LSTM(lstm_cells, return_sequences=False,
                                     dropout=dropout, recurrent_dropout=dropout)))
    else:
        model.add(LSTM(lstm_cells, return_sequences=False,
                       dropout=dropout, recurrent_dropout=dropout))
    # Fully connected layer
    model.add(Dense(number_dense, activation='relu'))
    # Dropout for regularization
    model.add(Dropout(0.5))
    # Output layer
    model.add(Dense(number_out, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# function to define callbacks
def make_callbacks(model_name, save=True):
    """Make list of callbacks for training"""

    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

    if save:
        callbacks.append(ModelCheckpoint(model_name + '.h5',
                                         save_best_only=True,
                                         save_weights_only=False))
    return callbacks


def build_and_fit_lgb(X_train, X_dev, Y_train, Y_dev,
                      n_estimators=100, feature_fraction=1.0, max_depth=-1):
    """
    function to build and train gradient-boost model
    """

    # MODEL
    # define the model with hyperparameters
    model = lgb.LGBMClassifier(n_estimators=n_estimators,
                               feature_fraction=feature_fraction,
                               max_depth=max_depth,
                               seed=0)

    # train the model
    model.fit(X_train, Y_train)

    # predictions training-set
    Y_train_pred = model.predict(X_train)
    # Y_train_pred_proba = model.predict_proba(X_train)[:, 1]

    # predictions dev-set
    Y_dev_pred = model.predict(X_dev)
    # Y_dev_pred_proba = model.predict_proba(X_dev)[:, 1]

    # METRICS
    # print accuracies TRAINING set
    print('Predictions TRAINING SET:')
    accuracy = accuracy_score(Y_train, Y_train_pred)
    print("Accuracy score on train set: {0:.2f}%".format(accuracy * 100))
    # print('')

    # print accuracies DEV set
    print('Predictions DEV SET:')
    accuracy = accuracy_score(Y_dev, Y_dev_pred)
    print("Accuracy score on dev set: {0:.2f}%".format(accuracy * 100))
    # print('')

    # FEATURE IMPORTANCE
    feature_imp = pd.DataFrame({'feature name': X_train.columns,
                                'feature importance': model.feature_importances_}).sort_values('feature importance',
                                                                                               ascending=False)
    return model, feature_imp