
# Naive Bayes


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

tweet_df = pd.read_csv('data/harvey_tweet_df.csv')
```


```python
# Choose features:
xVar = tweet_df[['data_source', 'has_coords', 'is_reply', 'is_quoting', 'lang_en', \
                 'source_other', 'source_Instagram', 'source_TwitterforiPhone', \
                 'source_TwitterWebClient', 'source_TwitterforAndroid', 'source_Paper.li', \
                 'source_Hootsuite', 'source_TweetMyJOBS', 'source_IFTTT', 'source_Facebook', \
                 'source_TweetDeck', 'source_TwitterforiPad', 'source_BubbleLife', \
                 'source_TwitterLite', 'hashtag_count', 'url_count', 'mention_count',
                 
                 'betweenness_centrality', 'closeness_centrality', 
                 'user_data_source', 'default_profile', 'default_profile_image', 
                 'degree_centrality', 'eigenvector_centrality', 'favourites_count', 
                 'followers_count', 'friends_count', 'geo_enabled', 'has_extended_profile', 
                 'listed_count', 'load_centrality', 
                 'ratio_detected', 'ratio_original', 
                 'statuses_count', 'tweets_per_hour', 'undirected_eigenvector_centrality', 
                 'verified', 'user_lang_en', 'profile_has_url', 
                 'profile_has_local_location'
                ]]

# 'in_degree','out_degree' are exlcuded as they contain negative values, may be error in data collection method.
# 'katz_centrality' is excluded as it is not calculated for this dataset.
# 'user_class' is 2 for all values
# 'ratio_media', was not captured for this dataset

yVar = tweet_df['data_code_id']

# Partition data sets:
X_train, X_test, y_train, y_test = train_test_split(xVar, yVar, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
```

    (1744, 45) (1744,)
    (436, 45) (436,)



```python
# Checking dataframe for NaN/negative values
pd.set_option('display.max_columns', None)

# Filling centrality measures from non-principal component
xVar = xVar.fillna(0)
#xVar.columns[xVar.isna().any()].tolist()
X_train, X_test, y_train, y_test = train_test_split(xVar, yVar, test_size=0.2)


# Check min/max values for columns (negatives are incompatible)
xVar.max().to_frame().T

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data_source</th>
      <th>has_coords</th>
      <th>is_reply</th>
      <th>is_quoting</th>
      <th>lang_en</th>
      <th>source_other</th>
      <th>source_Instagram</th>
      <th>source_TwitterforiPhone</th>
      <th>source_TwitterWebClient</th>
      <th>source_TwitterforAndroid</th>
      <th>source_Paper.li</th>
      <th>source_Hootsuite</th>
      <th>source_TweetMyJOBS</th>
      <th>source_IFTTT</th>
      <th>source_Facebook</th>
      <th>source_TweetDeck</th>
      <th>source_TwitterforiPad</th>
      <th>source_BubbleLife</th>
      <th>source_TwitterLite</th>
      <th>hashtag_count</th>
      <th>url_count</th>
      <th>mention_count</th>
      <th>betweenness_centrality</th>
      <th>closeness_centrality</th>
      <th>user_data_source</th>
      <th>default_profile</th>
      <th>default_profile_image</th>
      <th>degree_centrality</th>
      <th>eigenvector_centrality</th>
      <th>favourites_count</th>
      <th>followers_count</th>
      <th>friends_count</th>
      <th>geo_enabled</th>
      <th>has_extended_profile</th>
      <th>listed_count</th>
      <th>load_centrality</th>
      <th>ratio_detected</th>
      <th>ratio_original</th>
      <th>statuses_count</th>
      <th>tweets_per_hour</th>
      <th>undirected_eigenvector_centrality</th>
      <th>verified</th>
      <th>user_lang_en</th>
      <th>profile_has_url</th>
      <th>profile_has_local_location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>2.0</td>
      <td>11.0</td>
      <td>0.007306</td>
      <td>0.223369</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.009476</td>
      <td>0.128534</td>
      <td>317855.0</td>
      <td>4989.0</td>
      <td>5000.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1123.0</td>
      <td>0.007015</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>9999.0</td>
      <td>16.623686</td>
      <td>0.099818</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.metrics import classification_report, confusion_matrix

def benchmark_clf(clf):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    preds = clf.predict(X_test)
    
    print('=' * 80)
    print(str(clf).split('(')[0]) 
    print("%d mislabeled points out of a total %d" % 
          ((1-score)*y_test.shape[0], y_test.shape[0]))
    print("Accuracy: %.2f%%" % (score * 100))
    print(classification_report(y_test, preds))
    
    # Precision and Recall for chosen classes:
    calc_prec_recall([1,2], y_test, preds)
    #Confusion Matrix:
    print(pd.crosstab(y_test, preds, rownames=['Actual Result'], 
                      colnames=['Predicted Result'], dropna=False))
    #print(confusion_matrix(y_test, preds))

    
# Calculate precision and recall for the chosen categories:
def calc_prec_recall(code_group, y_test, preds):
    correct = 0
    total_true = 0
    for true, pred in list(zip(y_test, preds)):
        if true in code_group:
            total_true += 1
            if true == pred:
                correct += 1
    print('Recall for selected classes:', round(correct/total_true, 2) * 100, '%')
    correct = 0
    total_pred = 0
    for true, pred in list(zip(y_test, preds)):
        if pred in code_group:
            total_pred += 1
            if true == pred:
                correct += 1
    print('Precision for selected classes:', round(correct/total_pred, 2) * 100, '%')
    print('Naive Precision:', round(total_true/y_test.shape[0], 2) * 100, '%\n')
    return
```

## Naive Bayes

Naive Bayes makes the assumption that all features are independent (hence naive) and that all features are equally important in prediction. Neither are accurate for this application, however they represent a useful baseline classifier against which to measure.

NB don't require training (just calculation of probablities) so are useful for highly-dimensional data.

NB uses Maximum A Posteriori (MAP) decision rule, which incorporates a prior distribution. Therefore it is a regularisation of a Maximum Likelihood (ML) decision rule. Regularisation is useful when classes are not evenly distrubuted (consider breast cancer case).


```python
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB

benchmark_clf(MultinomialNB(alpha=.01))
benchmark_clf(ComplementNB(alpha=.01))
benchmark_clf(BernoulliNB(alpha=.01))
```

    ================================================================================
    MultinomialNB
    332 mislabeled points out of a total 436
    Accuracy: 23.85%
                  precision    recall  f1-score   support
    
               1       0.00      0.00      0.00         7
               2       0.11      0.06      0.08        64
               3       0.06      0.04      0.05        28
               4       0.38      0.23      0.29        99
               5       0.10      0.06      0.07        34
               6       0.39      0.10      0.15       114
               7       0.29      0.70      0.41        90
    
        accuracy                           0.24       436
       macro avg       0.19      0.17      0.15       436
    weighted avg       0.28      0.24      0.21       436
    
    Recall for selected classes: 6.0 %
    Precision for selected classes: 4.0 %
    Naive Precision: 16.0 %
    
    Predicted Result   1   2  3   4  5   6   7
    Actual Result                             
    1                  0   1  0   1  2   2   1
    2                  9   4  2   9  2   2  36
    3                  4   2  1   8  0   2  11
    4                  7   8  2  23  3   8  48
    5                  8   4  0   3  2   2  15
    6                 19  12  6  10  9  11  47
    7                  8   5  5   6  2   1  63
    ================================================================================
    ComplementNB
    299 mislabeled points out of a total 436
    Accuracy: 31.42%
                  precision    recall  f1-score   support
    
               1       0.00      0.00      0.00         7
               2       0.00      0.00      0.00        64
               3       0.00      0.00      0.00        28
               4       0.31      0.33      0.32        99
               5       0.00      0.00      0.00        34
               6       0.34      0.31      0.32       114
               7       0.31      0.77      0.44        90
    
        accuracy                           0.31       436
       macro avg       0.14      0.20      0.16       436
    weighted avg       0.22      0.31      0.25       436
    
    Recall for selected classes: 0.0 %
    Precision for selected classes: 0.0 %
    Naive Precision: 16.0 %
    
    Predicted Result  2   4   6   7
    Actual Result                  
    1                 0   1   4   2
    2                 0  12  15  37
    3                 0  14   7   7
    4                 2  33  20  44
    5                 1   7  12  14
    6                 0  29  35  50
    7                 1  11   9  69
    ================================================================================
    BernoulliNB
    231 mislabeled points out of a total 436
    Accuracy: 47.02%
                  precision    recall  f1-score   support
    
               1       0.00      0.00      0.00         7
               2       0.30      0.33      0.31        64
               3       0.00      0.00      0.00        28
               4       0.50      0.53      0.51        99
               5       0.25      0.03      0.05        34
               6       0.52      0.70      0.60       114
               7       0.53      0.57      0.55        90
    
        accuracy                           0.47       436
       macro avg       0.30      0.31      0.29       436
    weighted avg       0.42      0.47      0.44       436
    
    Recall for selected classes: 30.0 %
    Precision for selected classes: 30.0 %
    Naive Precision: 16.0 %
    
    Predicted Result   2  3   4  5   6   7
    Actual Result                         
    1                  1  0   2  0   1   3
    2                 21  1  11  3  16  12
    3                  3  0  18  0   4   3
    4                  7  3  52  0  30   7
    5                  6  2   4  1  11  10
    6                  9  0  15  0  80  10
    7                 23  3   2  0  11  51


    /home/rosles/projects/crisis-data/venv/lib/python3.6/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /home/rosles/projects/crisis-data/venv/lib/python3.6/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)



```python
from sklearn.feature_selection import SelectKBest, chi2

ch2 = SelectKBest(chi2, k=10)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)

ch2.scores_
#X_train[0]
#X_train.shape
```




    {'k': 10,
     'score_func': <function sklearn.feature_selection.univariate_selection.chi2(X, y)>}




```python

```


```python

```


```python

```
