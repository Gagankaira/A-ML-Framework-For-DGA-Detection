pip install PublicSuffixList
#import libraries
import numpy as np
import pandas as pd
import re
from publicsuffixlist import PublicSuffixList
import gc
import math
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Conv1D, MaxPooling1D, Input, Flatten
from keras import regularizers
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#for svm
from sklearn import svm,datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier


RANDOM_SEED = 1

# Load Bengin Domains data from .csv file and set the labels
benign_domain = pd.read_csv('drive/My Drive/data/top-1m-domain.csv', header=None, names=['Domain'])
benign_domain.head()
benign_domain['DGA_Family'] = 'none'
benign_domain['Type'] = 'Normal'
benign_domain =benign_domain[['DGA_Family','Domain','Type']]
print("---------------------------------------------------------------------------------------------------")
print(benign_domain.describe())


# Load DGA Domains data from .txt file and set the labels
dga_domain = pd.read_table('drive/My Drive/data/360_dga.txt',names=['DGA_Family','Domain','Start_time','End_time'])
dga_domain = dga_domain.iloc[:, 0:2]
dga_domain['Type']='DGA'
dga_domain.to_csv('drive/My Drive/data/360_dga_domain.csv', index = False)
print("----------------------------------------------------------------------------------------------------")
print(dga_domain.describe())

# Load DGA Domains from .txt file and set labels
bambenek_dga_domain = pd.read_csv('drive/My Drive/data/dga-feed.txt',header=None,sep=',',
                                  names=['Domain','DGA_Family','time','description_url'])
bambenek_dga_domain = bambenek_dga_domain[['DGA_Family','Domain']]
bambenek_dga_domain['Type']='DGA'
dga_familiy_list = list()
bambenek_dga_domain['DGA_Family']=bambenek_dga_domain['DGA_Family'].apply(lambda x: x.split(' ')[3])
bambenek_dga_domain.to_csv('drive/My Drive/data/bambenek_dga_domain.csv', index = False)
print("---------------------------------------------------------------------------------------------------")
print(bambenek_dga_domain.describe())

# Combine the Benign Domains and DGA Domains dataset
dga_domain = pd.concat([dga_domain,bambenek_dga_domain],axis=0)

dga_domain = dga_domain.drop_duplicates()
df_domain = pd.concat([dga_domain,benign_domain])
print("---------------------------------------------------------------------------------------------------")
print(df_domain.describe())

# Shuffle the dataset
df_domain_shuffle = df_domain.sample(frac = 1, random_state=RANDOM_SEED)
df_domain_shuffle.to_csv('drive/My Drive/data/mixed_domain.csv', index = False)
print("---------------------------------------------------------------------------------------------------")
print(df_domain_shuffle.head())

# Generate a copy of original dataset
domain_withFeatures = df_domain_shuffle.copy()
print("---------------------------------------------------------------------------------------------------")
print(domain_withFeatures.head())

# Load Valid Top Level Domains data
import sys

topLevelDomain = []
with open('drive/My Drive/data/tlds-alpha-by-domain.txt', 'r') as content:
    for line in content:
        topLevelDomain.append((line.strip('\n')))
print("---------------------------------------------------------------------------------------------------")        
print(topLevelDomain)

psl = PublicSuffixList()

def ignoreVPS(domain):
    # Return the rest of domain after ignoring the Valid Public Suffixes:
    validPublicSuffix = '.' + psl.publicsuffix(domain)
    if len(validPublicSuffix) < len(domain):
         # If it has VPS
        subString = domain[0: domain.index(validPublicSuffix)]  
    elif len(validPublicSuffix) == len(domain):
        return 0
    else:
        # If not
        subString = domain
    
    return subString

def typeTo_Binary(type):
  # Convert Type to Binary variable DGA = 1, Normal = 0
  if type == 'DGA':
    return 1
  else:
    return 0

def domain_length(domain):
  # Generate Domain Name Length (DNL)
  return len(domain)

def subdomains_number(domain):
  # Generate Number of Subdomains (NoS)
    subdomain = ignoreVPS(domain)
    return (subdomain.count('.') + 1)

def subdomain_length_mean(domain):
  # enerate Subdomain Length Mean (SLM) 
    subdomain = ignoreVPS(domain)
    result = (len(subdomain) - subdomain.count('.')) / (subdomain.count('.') + 1)
    return result

def has_www_prefix(domain):
  # Generate Has www Prefix (HwP)
  if domain.split('.')[0] == 'www':
    return 1
  else:
    return 0
  
def has_hvltd(domain):
  # Generate Has a Valid Top Level Domain (HVTLD)
  if domain.split('.')[len(domain.split('.')) - 1].upper() in topLevelDomain:
    return 1
  else:
    return 0
  
def contains_single_character_subdomain(domain):
  # Generate Contains Single-Character Subdomain (CSCS) 
    domain = ignoreVPS(domain)
    str_split = domain.split('.')
    minLength = len(str_split[0])
    for i in range(0, len(str_split) - 1):
        minLength = len(str_split[i]) if len(str_split[i]) < minLength else minLength
    if minLength == 1:
        return 1
    else:
        return 0

def contains_TLD_subdomain(domain):
  # Generate Contains TLD as Subdomain (CTS)
    subdomain = ignoreVPS(domain)
    str_split = subdomain.split('.')
    for i in range(0, len(str_split) - 1):
        if str_split[i].upper() in topLevelDomain:
            return 1
    return 0

def underscore_ratio(domain):
  # Generate Underscore Ratio (UR) on dataset
    subString = ignoreVPS(domain)
    result = subString.count('_') / (len(subString) - subString.count('.'))
    return result

def contains_IP_address(domain):
  # Generate Contains IP Address (CIPA) on datasetx
    splitSet = domain.split('.')
    for element in splitSet:
        if(re.match("\d+", element)) == None:
            return 0
    return 1  

def contains_digit(domain):
    """
    Contains Digits 
    """
    subdomain = ignoreVPS(domain)
    for item in subdomain:
        if item.isdigit():
            return 1
    return 0

def vowel_ratio(domain):
    """
    calculate Vowel Ratio 
    """
    VOWELS = set('aeiou')
    v_counter = 0
    a_counter = 0
    ratio = 0
    subdomain = ignoreVPS(domain)
    for item in subdomain:
        if item.isalpha():
            a_counter+=1
            if item in VOWELS:
                v_counter+=1
    if a_counter>1:
        ratio = v_counter/a_counter
    return ratio

def digit_ratio(domain):
    """
    calculate digit ratio

---


    """
    d_counter = 0
    counter = 0
    ratio = 0
    subdomain = ignoreVPS(domain)
    for item in subdomain:
        if item.isalpha() or item.isdigit():
            counter+=1
            if item.isdigit():
                d_counter+=1
    if counter>1:
        ratio = d_counter/counter
    return ratio
  
def prc_rrc(domain):
    """
    calculate the Ratio of Repeated Characters in a subdomain
    """
    subdomain = ignoreVPS(domain)
    subdomain = re.sub("[.]", "", subdomain)
    char_num=0
    repeated_char_num=0
    d = collections.defaultdict(int)
    for c in list(subdomain):
        d[c] += 1
    for item in d:
        char_num +=1
        if d[item]>1:
            repeated_char_num +=1
    ratio = repeated_char_num/char_num
    return ratio

def prc_rcc(domain):
    """
    calculate the Ratio of Consecutive Consonants
    """
    VOWELS = set('aeiou')
    counter = 0
    cons_counter=0
    subdomain = ignoreVPS(domain)
    for item in subdomain:
        i = 0
        if item.isalpha() and item not in VOWELS:
            counter+=1
        else:
            if counter>1:
                cons_counter+=counter
            counter=0
        i+=1
    if i==len(subdomain) and counter>1:
        cons_counter+=counter
    ratio = cons_counter/len(subdomain)
    return ratio

def prc_rcd(domain):
    """
    calculate the ratio of consecutive digits
    """
    counter = 0
    digit_counter=0
    subdomain = ignoreVPS(domain)
    for item in subdomain:
        i = 0
        if item.isdigit():
            counter+=1
        else:
            if counter>1:
                digit_counter+=counter
            counter=0
        i+=1
    if i==len(subdomain) and counter>1:
        digit_counter+=counter
    ratio = digit_counter/len(subdomain)
    return ratio

def prc_entropy(domain):
    """
    calculate the entropy of subdomain
    :param domain_str: subdomain
    :return: the value of entropy
    """
    subdomain = ignoreVPS(domain)
    # get probability of chars in string
    prob = [float(subdomain.count(c)) / len(subdomain) for c in dict.fromkeys(list(subdomain))]

    # calculate the entropy
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    return entropy

def extract_features():
    domain_withFeatures['DNL'] = domain_withFeatures['Domain'].apply(lambda x: domain_length(x))
    domain_withFeatures['NoS'] = domain_withFeatures['Domain'].apply(lambda x: subdomains_number(x))
    domain_withFeatures['SLM'] = domain_withFeatures['Domain'].apply(lambda x: subdomain_length_mean(x))
    domain_withFeatures['HwP'] = domain_withFeatures['Domain'].apply(lambda x: has_www_prefix(x))
    domain_withFeatures['HVTLD'] = domain_withFeatures['Domain'].apply(lambda x: has_hvltd(x))
    domain_withFeatures['CSCS'] = domain_withFeatures['Domain'].apply(lambda x: contains_single_character_subdomain(x))
    domain_withFeatures['CTS'] = domain_withFeatures['Domain'].apply(lambda x: contains_TLD_subdomain(x))
    domain_withFeatures['UR'] = domain_withFeatures['Domain'].apply(lambda x: underscore_ratio(x))
    domain_withFeatures['CIPA'] = domain_withFeatures['Domain'].apply(lambda x: contains_IP_address(x))
    domain_withFeatures['contains_digit']= domain_withFeatures['Domain'].apply(lambda x:contains_digit(x))
    domain_withFeatures['vowel_ratio']= domain_withFeatures['Domain'].apply(lambda x:vowel_ratio(x))
    domain_withFeatures['digit_ratio']= domain_withFeatures['Domain'].apply(lambda x:digit_ratio(x))
    domain_withFeatures['RRC']= domain_withFeatures['Domain'].apply(lambda x:prc_rrc(x))
    domain_withFeatures['RCC']= domain_withFeatures['Domain'].apply(lambda x:prc_rcc(x))
    domain_withFeatures['RCD']= domain_withFeatures['Domain'].apply(lambda x:prc_rcd(x))
    domain_withFeatures['Entropy']= domain_withFeatures['Domain'].apply(lambda x:prc_entropy(x))
extract_features()

domain_withFeatures['Type'] = domain_withFeatures['Type'].apply(lambda x: typeTo_Binary(x))

print("-----------------------------------------------------------------------------------------")                                                                
print(domain_withFeatures.head())
print("---------------------------------------------------------------------------------------------------")
print(domain_withFeatures.dtypes)

#print(domain_withFeatures.head())
print("---------------------------------------------------------------------------------------------------")
print(domain_withFeatures.describe())
print("---------------------------------------------------------------------------------------------------")
#print(domain_withFeatures.describe())

# Save the data to the disk
domain_withFeatures.to_csv('drive/My Drive/data/domain_withFeatures.csv', index=False)
 # Load the data so can save time when rerunning this notbook
domain_withFeatures = pd.read_csv('drive/My Drive/data/domain_withFeatures.csv')
print("---------------------------------------------------------------------------------------------------")
print(domain_withFeatures.head())


drop_column = {'DGA_Family', 'Domain', 'HwP', 'HVTLD', 'UR', 'CSCS', 'CTS'}

# Drop the unnecessary columns
domain_withFeatures_fixed = domain_withFeatures.drop(drop_column, axis = 1)
print("---------------------------------------------------------------------------------------------------")
print(domain_withFeatures_fixed.head())
print("---------------------------------------------------------------------------------------------------")
print(domain_withFeatures_fixed.describe())

# Checking whether there is null value
print("---------------------------------------------------------------------------------------------------")
print(domain_withFeatures_fixed.isnull().sum())
# Get independent variables and dependent variables
attributes = domain_withFeatures_fixed.drop('Type', axis=1)
observed = domain_withFeatures_fixed['Type']
print("---------------------------------------------------------------------------------------------------")
print(attributes.shape, observed.shape)

train_X, test_X, train_y, test_y = train_test_split(attributes, observed, test_size = 0.20, random_state = RANDOM_SEED)
train_X.shape, test_X.shape, train_y.shape, test_y.shape

##Random forest
rf = RandomForestClassifier(random_state= RANDOM_SEED)
rf.fit(train_X, train_y)
test_rf_pred = rf.predict(test_X)
print("---------------------------------------------------------------------------------------------------")
print(test_rf_pred)
print("---------------------------------------------------------------------------------------------------")
print(confusion_matrix(test_y,test_rf_pred))
score_rf_test = round(accuracy_score(test_y, test_rf_pred) * 100, 2)
print("Accuracy of Random Forest Model: ", score_rf_test)
precision = precision_score(test_y,test_rf_pred, average='binary')
print('Precision of Random Forest: %.3f' % precision)
recall = recall_score(test_y,test_rf_pred, average='binary')
print('Recall: %.3f' % recall)
score = f1_score(test_y,test_rf_pred, average='binary')
print('F-Measure: %.3f' % score)
print("---------------------------------------------------------------------------------------------------")




'''#The kernel function is what is applied on each data instance to map the original
# non-linear observations into a higher-dimensional space in which they become separable. Using the dog breed prediction example again, kernels offer a better alternative.



##lstm

#https://datascienceplus.com/long-short-term-memory-lstm-and-how-to-implement-lstm-using-python/

print('LSTM model')
model = Sequential()
#Dense layer performs the operation on the input layers and returns the output and every neuron at 
#the previous layer is connected to the neurons in the next layer hence it is called fully connected Dense layer
model.add(Dense(11,activation='relu',input_dim=11))
#The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero. ... 
#The rectified linear activation function overcomes the vanishing gradient problem, allowing models to learn faster and perform better
model.add(Dense(1,activation='sigmoid'))
#sigmoid activation function to make 0 or 1 predictions for the two classes (good and bad) in the problem.
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#Finally, because this is a classification problem we use a Dense output layer with a single neuron .
#error is calculated by loss function ‘mean squared error’ ( as it is a regression problem so we use mean squared error loss function).


#Because it is a binary classification problem, log loss is used as the loss function (binary_crossentropy in Keras). The efficient ADAM optimization algorithm is used. 
#The model is fit for only 2 epochs because it quickly overfits the problem. A large batch size of 64 reviews is used to space out weight updates.
#Computes the cross-entropy loss between true labels and predicted labels.
#Use this cross-entropy loss when there are only two label classes (assumed to be 0 and 1). For each example, there should be a single floating-point value per prediction.
model.fit(train_X, train_y,epochs=2)
Y_pre = model.predict(test_X)


'''
'''Adaptive Movement Estimation algorithm, or Adam for short, is an extension to the gradient descent optimization algorithm.
The algorithm was described in the 2014 paper by Diederik Kingma and Jimmy Lei Ba titled “Adam: A Method for Stochastic Optimization.”
Adam is designed to accelerate the optimization process, e.g. decrease the number of function evaluations required to reach the optima,
 or to improve the capability of the optimization algorithm, e.g. result in a better final result.
This is achieved by calculating a step size for each input parameter that is being optimized. Importantly,
 each step size is automatically adapted throughput the search process based on the gradients (partial derivatives) encountered for each variable.
 Compile the model using ‘adam optimizer’ (It is a learning rate optimization algorithm used while training of DNN models)
  and error is calculated by loss function ‘mean squared error’ ( as it is a regression problem so we use mean squared error loss function).
Then fit the model on 30 epoch(epochs are the number of times we pass the data into the neural network) 
and a batch size of 50(we pass the data in batches, segmenting the data into smaller parts so as for network to process the data in parts).'''
'''
'''

"""svm
print('svm accuracy')

#Import svm model
from sklearn import svm
from sklearn import metrics

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(train_X, train_y)

#Predict the response for test dataset
y_pred = clf.predict(test_X)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_y, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(test_y, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(test_y, y_pred))
"""
''''''

# Use Logisitic Regression to build the model
print('LOGISTIC REGRESSION')
from sklearn import metrics
lg = LogisticRegression(random_state=42)
lg.fit(train_X, train_y)
train_lg_pred = lg.predict(train_X)
y_pred = lg.predict(test_X)
# Calculate the accuracy
score_lg_train = round(accuracy_score(train_y, train_lg_pred) * 100, 5)
score_lg_test = round(accuracy_score(test_y, y_pred) * 100, 5)
print('---------------------------------------------------------------------------------------------------------------------------------')
print('LOGISTIC REGRESSION')
print("Accuracy of Logistic Regression on training dataset: ", score_lg_train)
print("Accuracy of Logistic Regression on test dataset: ", score_lg_test)
print(metrics.classification_report(test_y, y_pred))

#import libraries
import numpy as np
import pandas as pd
import re
from publicsuffixlist import PublicSuffixList
import gc
import math
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Conv1D, MaxPooling1D, Input, Flatten
from keras import regularizers
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#for svm
from sklearn import svm,datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier


RANDOM_SEED = 1

# Load Bengin Domains data from .csv file and set the labels
benign_domain = pd.read_csv('drive/My Drive/data/top-1m-domain.csv', header=None, names=['Domain'])
benign_domain.head()
benign_domain['DGA_Family'] = 'none'
benign_domain['Type'] = 'Normal'
benign_domain =benign_domain[['DGA_Family','Domain','Type']]
print("---------------------------------------------------------------------------------------------------")
print(benign_domain.describe())


# Load DGA Domains data from .txt file and set the labels
dga_domain = pd.read_table('drive/My Drive/data/360_dga.txt',names=['DGA_Family','Domain','Start_time','End_time'])
dga_domain = dga_domain.iloc[:, 0:2]
dga_domain['Type']='DGA'
dga_domain.to_csv('drive/My Drive/data/360_dga_domain.csv', index = False)
print("----------------------------------------------------------------------------------------------------")
print(dga_domain.describe())

# Load DGA Domains from .txt file and set labels
bambenek_dga_domain = pd.read_csv('drive/My Drive/data/dga-feed.txt',header=None,sep=',',names=['Domain','DGA_Family','time','description_url'])
bambenek_dga_domain = bambenek_dga_domain[['DGA_Family','Domain']]
bambenek_dga_domain['Type']='DGA'
dga_familiy_list = list()
bambenek_dga_domain['DGA_Family']=bambenek_dga_domain['DGA_Family'].apply(lambda x: x.split(' ')[3])
bambenek_dga_domain.to_csv('drive/My Drive/data/bambenek_dga_domain.csv', index = False)
print("---------------------------------------------------------------------------------------------------")
print(bambenek_dga_domain.describe())

benign_domains = benign_domain['Domain'].tolist()
dga_domains = dga_domain['Domain'].tolist() + bambenek_dga_domain['Domain'].tolist()
len(benign_domains)
len(dga_domains)
X = benign_domains + dga_domains
unique_chars = enumerate(set(''.join(X))) 
chars_dict = dict()
for i, x in unique_chars: #index of enum starts with 0
    #print('i: ' + str(i) + '  x: ' + x)
    chars_dict[x] = i + 1 #leave 0 for padding
#index 0 is also going to be a feature(padding/unknown).
max_features_num = len(chars_dict) + 1
print(max_features_num)

# Convert characters to int
X_in_int = []
for domain in X:
    domain_in_int = []
    for c in domain:
        domain_in_int.append(chars_dict[c])
    X_in_int.append(domain_in_int)

print(X_in_int[1])

print(X[1])

#update X
X = X_in_int
'''
mu, sigma = 3,1
# mean and standard deviation
s = np.random.lognormal(mu, sigma, 1000)
count, bins, ignored = plt.hist(s, 100, normed=True, align='mid')
x = np.linspace(min(bins), max(bins), 10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
      / (x * sigma * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')
plt.show()
'''
#max length will be the max length of domain in our dataset
maxlen = np.max([len(x) for x in X])

print(maxlen)

#pad to max length
X = sequence.pad_sequences(X, maxlen=maxlen)

print(X.shape)

#Generate corresponding Y, 0 for 'benign'; 1 for 'dga'
Y = np.hstack([np.zeros(len(benign_domains)),np.ones(len(dga_domains))])
#LSTM

def build_model(max_features_num, maxlen):
    """Build LSTM model"""
    model = Sequential()
    model.add(Embedding(max_features_num, 64, input_length=maxlen))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['binary_crossentropy','acc'])

    return model

pos_neg_cutpoint = len(benign_domains)
print("The cut point will be "+ str(pos_neg_cutpoint))

#set new sampling szie as 300K
sampling_size = 300000
import random
pos_indices = random.sample(range(pos_neg_cutpoint),sampling_size)
neg_indices = random.sample(range(pos_neg_cutpoint, len(X)),sampling_size)

print(len(pos_indices))


print(pos_indices[:10])

print(len(neg_indices))
print(neg_indices[:10])

new_X = X[pos_indices + neg_indices]
new_Y = Y[pos_indices + neg_indices]

print(new_X.shape)

#training parameters

max_epoch=2
nfolds=10
batch_size=128
#call backs
from keras.callbacks import EarlyStopping
cb = []

cb.append(EarlyStopping(monitor='val_loss', 
                        min_delta=0, #an absolute change of less than min_delta, will count as no improvement
                        patience=5, #number of epochs with no improvement after which training will be stopped
                        verbose=0, 
                        mode='auto', 
                        baseline=None, 
                        restore_best_weights=False))

model = build_model(max_features_num, maxlen)
train_X, test_X, train_y, test_y = train_test_split(X,Y ,test_size = 0.20, random_state = RANDOM_SEED)
print(model.summary())
history = model.fit(x=new_X, y=new_Y,epochs=max_epoch)
#history = model.fit(train_X,train_y,epochs=max_epoch)
print(history)

model.save('LSTM_on_300K')

# Plot training & validation accuracy values 
plt.plot(history.history['acc']) 
plt.plot(history.history['acc']) 
plt.title('Model accuracy') 
plt.ylabel('Accuracy') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

# Plot training & validation loss values 
plt.plot(history.history['loss']) 
plt.plot(history.history['loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()


score_lstm_test = 98.66

# All model accuracy data (2014)
model_Score = {
    
    
    'Random Forest':score_rf_test,
    'Logistic regression':score_lg_test,
    'LSTM NN': score_lstm_test
}

print(model_Score)


# Plot each model score
def showScore(model_score_dict, title):
    score = pd.Series(model_score_dict)
    score = score.sort_values(ascending=False)
    plt.figure(figsize=(12,8))
    #Colors
    ax = score.plot(kind='bar') 
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.ylim([60.0, 100.0])
    plt.xlabel('Model')
    plt.ylabel('Percentage')
    plt.title(title)
    plt.show()

print(showScore(model_Score, 'The score of model for DGA Detection'))

corrmat = domain_withFeatures.corr()

plt.figure(figsize=(15,15))
sns.heatmap(corrmat, annot=True, cmap= "RdBu_r")

k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Type')['Type'].index
cm = np.corrcoef(domain_withFeatures[cols].values.T)
f, ax = plt.subplots(figsize=(16, 16))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cmap = "RdBu_r", cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

corrmat = domain_withFeatures.corr()

plt.figure(figsize=(15,15))
sns.heatmap(corrmat, annot=True, cmap= "RdBu_r")

k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Type')['Type'].index
cm = np.corrcoef(domain_withFeatures[cols].values.T)
f, ax = plt.subplots(figsize=(16, 16))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cmap = "RdBu_r", cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
