#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[2]:


get_ipython().system('pip install streamlit')


# In[3]:


df = pd.read_csv("spam.csv", encoding_errors= 'replace')


df


# In[4]:


df.sample(5)


# In[5]:


df.shape


# In[6]:


# 1. Data cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model building
# 5. Evaluation
# 6. Improvement
# 7. Website
# 8. Deploy


# ## 1. Data Cleaning

# In[7]:


df.info()


# In[8]:


# drop last 3 cols
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[9]:


df.sample(5)


# In[10]:


# renaming the cols
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[11]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[12]:


df['target'] = encoder.fit_transform(df['target'])


# In[13]:


df.head()


# In[14]:


# missing values
df.isnull().sum()


# In[15]:


# check for duplicate values
df.duplicated().sum()


# In[16]:


# remove duplicates
df = df.drop_duplicates(keep='first')


# In[17]:


df.duplicated().sum()


# In[18]:


df.shape


# ## 2.EDA

# In[19]:


df.head()


# In[20]:


df['target'].value_counts()


# In[21]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[22]:


# Data is imbalanced


# In[23]:


import nltk


# In[24]:


get_ipython().system('pip install nltk')


# In[25]:


nltk.download('punkt')


# In[26]:


df['num_characters'] = df['text'].apply(len)


# In[27]:


df.head()


# In[28]:


# num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[29]:


df.head()


# In[30]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[31]:


df.head()


# In[32]:


df[['num_characters','num_words','num_sentences']].describe()


# In[33]:


# ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[34]:


#spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[35]:


import seaborn as sns


# In[36]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[37]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[38]:


sns.pairplot(df,hue='target')


# In[39]:


sns.heatmap(df.corr(),annot=True)


# ## 3. Data Preprocessing
# - Lower case
# - Tokenization
# - Removing special characters
# - Removing stop words and punctuation
# - Stemming

# In[40]:


get_ipython().system('pip install nltk')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[43]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
            text = y[:]
            y.clear()
            
            for i in text:
                if i not in stopwords.words('english') and i not in string.punctuation:
                        y.append(i)
                        
            text = y[:]
            y.clear()
            
            for i in text:
                y.append(ps.stem(i))
                
            
            
    
    return " ".join(y) 
   


# In[ ]:





# In[53]:


def transform_text(input_text):
    # Your transformation logic goes here
    # For example, converting text to uppercase
    result = input_text.upper()
    return result
transform_text('did you like my presentation on ML?')


# In[45]:


df['text'][1]


# In[46]:


stopwords.words('english')


# In[ ]:





# In[47]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')


# In[54]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[55]:


get_ipython().system('pip install wordCloud')


# In[56]:


df.head()


# In[57]:


from wordcloud import WordCloud
wc = WordCloud(width=50,height=50,min_font_size=10,background_color='white')


# In[ ]:


#spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[ ]:





# In[ ]:


#plt.figure(figsize=(15,6))
#plt.imshow(spam_wc)


# In[ ]:


#ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[ ]:


#plt.figure(figsize=(15,6))
#plt.imshow(ham_wc)


# In[58]:


df.head()


# In[59]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
        


# In[60]:


len(spam_corpus)


# In[61]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[62]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[63]:


len(ham_corpus)


# In[ ]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[64]:


# Text Vectorization
# using Bag of Words
df.head()


# ## 4. Model Building

# In[65]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[66]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[67]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# In[68]:


# appending the num_character col to X
X = np.hstack((X,df['num_characters'].values.reshape(-1,1)))


# In[69]:


X.shape


# In[70]:


y = df['target'].values


# In[71]:


from sklearn.model_selection import train_test_split


# In[72]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[73]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[74]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[75]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[76]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[77]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[78]:


get_ipython().system('pip install xgboost ')


# In[ ]:


# tfidf --> MNB


# In[79]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[80]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[ ]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[81]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[82]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[83]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Assuming clfs is a dictionary of classifiers
clfs = {
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'LogisticRegression': LogisticRegression(),
    'KNN': KNeighborsClassifier()
}


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[84]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[85]:


performance_df


# In[86]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")


# In[87]:


performance_df1


# In[88]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[89]:


# model improve
# 1. Change the max_features parameter of TfIdf


# In[90]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[91]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)


# In[92]:


new_df = performance_df.merge(temp_df,on='Algorithm')


# In[93]:


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


# In[94]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)


# In[95]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[96]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[97]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[98]:


voting.fit(X_train,y_train)


# In[99]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[100]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[101]:


from sklearn.ensemble import StackingClassifier


# In[102]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[ ]:





# In[103]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[104]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




