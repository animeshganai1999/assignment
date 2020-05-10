import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')
x_train = df1[['country','review_title','province','points','price']]
x_train = x_train.fillna(method = 'ffill')
y_train = df1['variety']

x_test = df2[['country','review_title','province','points','price']]
x_test = x_test.fillna(method = 'ffill')

X = [x_train,x_test]
X = pd.concat(X)
X = X.reset_index()
del X['index']

#cleaning the review_title part
corpus = []
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
for i in range(len(X)):
    review = re.sub('[^a-zA-Z]',' ',X['review_title'][i])
    
    review = review.lower()
    #to remove all preposition , articals and stemming
    review = review.split() 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
 

X = X.iloc[:,:].values
y_train = y_train.iloc[:].values

X = np.delete(X,[1],axis = 1)

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
X[:,0] = labelencoder_x.fit_transform(X[:,0])
labelencoder_x = LabelEncoder()
X[:,1] = labelencoder_x.fit_transform(X[:,1])

#creating bag of word model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
x1 = cv.fit_transform(corpus).toarray()


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder',OneHotEncoder(sparse=True),[0,1])],remainder = 'passthrough')
X = ct.fit_transform(X).toarray()
X = np.delete(X,[0,100],axis = 1)

#feature scalling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
X_data = np.hstack((x1,X))

X_train = X_data[:len(x_train)]
X_test = X_data[len(x_train):]

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200,criterion = 'entropy',n_jobs = -1)
classifier.fit(X_train,y_train)


#predicting result
y_pred = classifier.predict(X_test)

#score on training set
print(classifier.score(X_train,y_train))
