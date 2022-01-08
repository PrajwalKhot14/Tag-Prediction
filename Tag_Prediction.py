import numpy as np
import pandas as pd
import string
import nltk
nltk.download('wordnet')
from sklearn.metrics import precision_score

from nltk.stem import WordNetLemmatizer
import nltk
import re
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

print("Loading dataset")
question= pd.read_csv('Dataset/Questions.csv', encoding='latin')
answer= pd.read_csv('Dataset/Answers.csv', encoding='latin')
tags= pd.read_csv('Dataset/Tags.csv', encoding='latin')

print("Dataset loaded")
#Merging answers dataframe wrt questoins ID
answer.drop(columns=['Id','OwnerUserId', 'CreationDate'],inplace=True)
answer.columns=['Id', 'A_Score', 'A_Body']
grouped_answer = answer.groupby("Id")['A_Body'].apply(lambda answer: ' '.join(answer))
grouped_answer= grouped_answer.to_frame()
grouped_answer= grouped_answer.sort_values(by='Id')

#Performing the same operation of tags dataframe
tags['Tag']= tags['Tag'].astype(str)
grouped_tags = tags.groupby("Id")['Tag'].apply(lambda tags: ' '.join(tags))

grouped_tags= grouped_tags.to_frame()
grouped_tags= grouped_tags.sort_values(by='Id')

#Combining questions, grouped_answer and grouped_tags dataframe
grouped_answer['Ids']= grouped_answer.index
grouped_tags['Ids']= grouped_tags.index
question.columns= ['Ids', 'OwnerUserId', 'CreationDate', 'ClosedDate', 'Score', 'Title','Body']
question= question.sort_values(by='Ids')
df= pd.merge(question,grouped_answer,how='left')
df1= pd.merge(df,grouped_tags,how='left',on='Ids')


#Removing unnecessary columns and removing duplicates
df1.drop(columns=['Ids', 'OwnerUserId', 'CreationDate', 'ClosedDate'],inplace=True)
df1=df1.drop_duplicates()


df2= df1.groupby(by='Tag')['Tag'].count().sort_values(ascending=False).to_frame()
df2.columns= ['Tag_count']
df2['Tags']=df2.index

df1.columns= ['Score', 'Title', 'Body', 'A_Body', 'Tags']
df1= pd.merge(df1,df2,how='left',on='Tags')

#Selecting only those tags which have been repeated for atleast 1000 times and the score is more than 3 for better prediction
df1= df1[df1['Tag_count']>=1000]
df1= df1[df1['Score']>3]
df1.drop(columns=['A_Body'],inplace=True)

#Removing punctuation
def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

df1['Title']= df1['Title'].astype(str)
df1['Title']= df1['Title'].apply(remove_punctuation)
df1['Title']=df1['Title'].str.lower()
df1['Title']= df1['Title'].str.split()

df1['Body']= df1['Body'].astype(str)
df1['Body']= df1['Body'].apply(lambda x: re.sub('<[^<]+?>','',x))
df1['Body']= df1['Body'].apply(remove_punctuation)
df1['Body']=df1['Body'].str.lower()
df1['Body']= df1['Body'].str.split()

lematizer= WordNetLemmatizer()

def word_lemmatizer(text):
    lem_text=[lematizer.lemmatize(i) for i in text]
    return lem_text

df1['Title']= df1['Title'].apply(lambda x: word_lemmatizer(x))
df1['Body']= df1['Body'].apply(lambda x: word_lemmatizer(x))


sp= spacy.load('en_core_web_sm')
all_stopwords= sp.Defaults.stop_words
df1['Title']= df1['Title'].apply(lambda x:[word for word in x if not word in all_stopwords])
df1['Body']= df1['Body'].apply(lambda x:[word for word in x if not word in all_stopwords])

#Final dataset
df1.drop(columns=['Tag_count','Score'], inplace=True)

#TF-IDF
vectorizer = TfidfVectorizer()
df1['Title']= df1['Title'].astype(str)
X1 = vectorizer.fit_transform(df1['Title'].str.lower())

df1['Body']= df1['Body'].astype(str)
X2 = vectorizer.fit_transform(df1['Body'].str.lower())

#Converting categorical values to numerical values/ Encoding
le= LabelEncoder()
df1['Tags']= le.fit_transform(df1['Tags'])

y = df1['Tags'].values

#Splitting the data for training and testing
x_train, x_test, y_train, y_test = train_test_split(X2, y, test_size=0.30, random_state=42)

#Modelling
#1. Logistic Regression
lr = LogisticRegression(C=10)

lr=lr.fit(x_train,y_train)
prediction=lr.predict(x_test)
print("Logistic Regression\n")
print(metrics.confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))
print("-"*85)

#2. XGB Classifier
xgb=XGBClassifier(max_depth=2, learning_rate=0.2, n_estimators=400, objective='binary:logistic', booster='gbtree')
xgb=xgb.fit(x_train,y_train)
pred=xgb.predict(x_test)

print("XGB Classifier\n")
print(metrics.confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print("-"*85)

#3. Multinomial NB
mnb = MultinomialNB().fit(x_train,y_train)
pred = mnb.predict(x_test)

print("Multinomial NB\n")
print(metrics.confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print("-"*85)

#4. KNN
knn = KNeighborsClassifier(n_neighbors=4)
knn = knn.fit(x_train,y_train)
pred = knn.predict(x_test)

print("KNN\n")
print(metrics.confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print("-"*85)

#5. Random Forest Classifier
rfc = RandomForestClassifier(max_depth=4, n_estimators=600,criterion='entropy')
rfc = rfc.fit(x_train,y_train)
pred = rfc.predict(x_test)

print("Random Forest Classifier\n")
print(metrics.confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print("-"*85)