
# General data questions and exploration



```python
##TODO:
# Try ML with and without network metrics
# Test at different time periods
# Test other event datasets
# Check steps from book - change to best practice sklearn


# What correlates with witness label
# GPS accounts tend to be spam/businesses?
# Compare GPS from stream / profile / hand coding
# Explore age of account
# Is detecting co-occuring tags viable?
# What kind of data/user is likely to be deleted?
# Is user name change / user deletion/protection a useful predictor
# Compare Change in network, whether it's useful to collect.
# Check gps count, location in profile
# Check timezone distribution
# 'Ordinary person' vs bot/celeb/business/news -- using source field, tweet rate, timezone

# prop of gps -- users and tweets. Automated, instagram sourced?
# prop of sources
# prop of media/urls
# users with location on profile? Some set 'in solidarity'?
# Cycadian posting rythym - can identify real people vs bots?
# location via friend network?
# language

# \item Tweets which were automatically generated from Instagram posts were much more likely to include GPS coordinates, and as media, more likely to represent a ground truth. Therefore this content may be worth focusing on.
# \item Aid requests were very rare. Those that were identified were often reposts rather than originals, and are often referring to the same original message which begins to trend.
# \item Info for affected class should differentiate between immediate and non-immediate content. E.g. a call to mobilise a clean-up or rescue crew vs. a link to an insurance claim form.
# \item For `unrelated' messages, those which matched the keyword stream were highly represented by automated messages coming from a particular set of sources which presumably uses trending tags to gain exposure. This is easy to pre-filter.
# \item Geographically-tagged Tweets are predominantly either: Instagram cross-posts, or automatically generated job listings from a small set of sources (and therefore easy to pre-filter).
        

# Sum of network edge reciprocity
# k-cohesiveness -- Structural cohesion
```


```python
### Initialisation ###
import os
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = [6, 4]

EVENT_NAME = Event.objects.all()[0].name.replace(' ', '')
DIR = './data/harvey_user_location/'
DF_FILENAME = 'df_users.csv'

# Confirm correct database is set in Django settings.py
if 'Harvey' not in EVENT_NAME:
    raise Exception('Event name mismatch -- check database set in Django')
```


```python
# Open original Dataframe
users_df = pd.read_csv(DIR + DF_FILENAME, index_col=0)
users_df.shape
```




    (1500, 46)



## Geographic Metadata and Manual Coding
Manual coding of users targetted the perceived locality of the user to the event. We can compare the geographic metadata provided by Twitter to these codes to determine their usefulness as a predictor for this value.


```python
# has_tweet_from_locality
# is_local_profile_location

# is_local_timezone
# is_lang_en

# is_coded_as_witness
# is_coded_as_non_witness
```


```python
# users_df.loc[users_df["is_local_profile_location"] == 1]["is_coded_as_witness"]

vals = users_df.loc[users_df["is_local_profile_location"] == 1]["is_coded_as_witness"].value_counts()
vals2 = users_df.loc[users_df["is_coded_as_witness"] == 1]["is_local_profile_location"].value_counts()


print('{:.4}% of {} users with local profile locations were coded as witness'.format(vals[1]/sum(vals)*100, sum(vals)))
print('{:.4}% of {} users were classified as having a local profile'.format(sum(vals)/len(users_df)*100, len(users_df)))
print('{:.4}% of {} witness codes had a local profile'.format(vals[1]/sum(vals)*100, sum(vals2)))
#vals2
```

    64.99% of 397 users with local profile locations were coded as witness
    26.47% of 1500 users were classified as having a local profile
    64.99% of 386 witness codes had a local profile



```python
import pandas as pd

def confusion_matrix(df: pd.DataFrame, col1: str, col2: str):
    """
    Given a dataframe with at least
    two categorical columns, create a 
    confusion matrix of the count of the columns
    cross-counts
    """
    return (
            df
            .groupby([col1, col2])
            .size()
            .unstack(fill_value=0)
            )


def calc_agreement_coefs(df: pd.DataFrame):
    """
    Calculates Cohen's Kappa and
    Krippendorff's Alpha for a
    given confusion matrix.
    """
    arr = df.to_numpy()
    n = arr.sum()
    p_o = 0
    for i in range(len(arr)):
        p_o += arr[i][i]/n
    p_e = 0
    for i in range(len(arr)):
        p_e += (arr.sum(axis=1)[i] *
                arr.sum(axis=0)[i]) / (n*n)
    kappa = (p_o-p_e)/(1-p_e)
    
    coin_arr = np.transpose(arr) + arr
    exp_distribution = [sum(x) for x in coin_arr]
    p_e_krippendorf = sum([a * (a-1) for a in exp_distribution])/(2*n*((2*n)-1))
    alpha = (p_o - p_e_krippendorf) / (1-p_e_krippendorf)
    
    return p_o, kappa, alpha


def calc_prec_recall(df: pd.DataFrame):
    """
    Calculates precision, recall and
    f-score for a given confusion matrix.
    
    Assumes true condition as ROW heading and
    ascending integer labels.
    """
    arr = df.to_numpy()
    if len(arr) != 2:
        return null
    results = {}
    results['Prevalence'] = arr.sum(axis=0)[1]/arr.sum()
    results['Accuracy'] = (arr[0][0] + arr[1][1])/arr.sum()
    results['Prec'] = arr[1][1]/arr.sum(axis=1)[1]
    results['Recall'] = arr[1][1]/arr.sum(axis=0)[1]
    results['f1Score'] = (2 * results['Prec'] * results['Recall'])/(results['Prec']+results['Recall'])
    results['Specificity'] = arr[0][0]/arr.sum(axis=0)[0]
    results['FalseNegRate'] = arr[0][1]/arr.sum(axis=0)[1]
    return results
    
```


```python
conf = confusion_matrix(users_df, 'is_local_profile_location', 'is_coded_as_witness')
conf
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>is_coded_as_witness</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>is_local_profile_location</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>975</td>
      <td>128</td>
    </tr>
    <tr>
      <th>1</th>
      <td>139</td>
      <td>258</td>
    </tr>
  </tbody>
</table>
</div>




```python
results = calc_prec_recall(conf)

p_o, kappa, alpha = calc_agreement_coefs(conf)
results['Cohen\'s Kappa'] = kappa
results['Krippendorff\'s Alpha'] = alpha

pd.DataFrame.from_dict(results, orient='index')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Prevalence</th>
      <td>0.257333</td>
    </tr>
    <tr>
      <th>Accuracy</th>
      <td>0.822000</td>
    </tr>
    <tr>
      <th>Prec</th>
      <td>0.649874</td>
    </tr>
    <tr>
      <th>Recall</th>
      <td>0.668394</td>
    </tr>
    <tr>
      <th>f1Score</th>
      <td>0.659004</td>
    </tr>
    <tr>
      <th>Specificity</th>
      <td>0.875224</td>
    </tr>
    <tr>
      <th>FalseNegRate</th>
      <td>0.331606</td>
    </tr>
    <tr>
      <th>Cohen's Kappa</th>
      <td>0.538603</td>
    </tr>
    <tr>
      <th>Krippendorff's Alpha</th>
      <td>0.538725</td>
    </tr>
  </tbody>
</table>
</div>




```python
conf = confusion_matrix(users_df, 'is_non_local_profile_location', 'is_coded_as_non_witness')
results = calc_prec_recall(conf)
p_o, kappa, alpha = calc_agreement_coefs(conf)
results['Cohen\'s Kappa'] = kappa
results['Krippendorff\'s Alpha'] = alpha
pd.DataFrame.from_dict(results, orient='index')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Prevalence</th>
      <td>0.722000</td>
    </tr>
    <tr>
      <th>Accuracy</th>
      <td>0.694000</td>
    </tr>
    <tr>
      <th>Prec</th>
      <td>0.892947</td>
    </tr>
    <tr>
      <th>Recall</th>
      <td>0.654663</td>
    </tr>
    <tr>
      <th>f1Score</th>
      <td>0.755461</td>
    </tr>
    <tr>
      <th>Specificity</th>
      <td>0.796163</td>
    </tr>
    <tr>
      <th>FalseNegRate</th>
      <td>0.345337</td>
    </tr>
    <tr>
      <th>Cohen's Kappa</th>
      <td>0.371632</td>
    </tr>
    <tr>
      <th>Krippendorff's Alpha</th>
      <td>0.346952</td>
    </tr>
  </tbody>
</table>
</div>




```python
# users_df.loc[users_df["is_local_profile_location"] == 1]["is_coded_as_witness"]

vals = users_df.loc[users_df["is_local_profile_location"] == 1]["is_coded_as_witness"].value_counts()
vals2 = users_df.loc[users_df["is_coded_as_witness"] == 1]["is_local_profile_location"].value_counts()


print('{:.3}% of {} users with local profile locations were coded as witness'.format(vals[1]/sum(vals)*100, sum(vals)))
print('{:.3}% of {} users were classified as having a local profile'.format(sum(vals)/len(users_df)*100, len(users_df)))
print('{:.3}% of {} witness codes had a local profile'.format(vals[1]/sum(vals)*100, sum(vals2)))
#vals2
```

    65.0% of 397 users with local profile locations were coded as witness
    26.5% of 1500 users were classified as having a local profile
    65.0% of 386 witness codes had a local profile



```python
users_df.loc[users_df["is_coded_as_witness"] == 1]
#users_df.loc[users_df["is_local_profile_location"] == 1]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>added_at</th>
      <th>centrality_betweenness</th>
      <th>centrality_closeness</th>
      <th>centrality_degree</th>
      <th>centrality_eigenvector</th>
      <th>centrality_load</th>
      <th>centrality_undirected_eigenvector</th>
      <th>created_at</th>
      <th>default_profile</th>
      <th>default_profile_image</th>
      <th>...</th>
      <th>account_age</th>
      <th>day_of_detection</th>
      <th>description_length</th>
      <th>is_lang_en</th>
      <th>has_translator_type</th>
      <th>has_url</th>
      <th>has_changed_screen_name</th>
      <th>is_data_source_3</th>
      <th>is_coded_as_witness</th>
      <th>is_coded_as_non_witness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>2017-08-26 02:13:06.809104+00:00</td>
      <td>5.731425e-07</td>
      <td>0.141014</td>
      <td>0.000304</td>
      <td>8.062092e-07</td>
      <td>4.541555e-07</td>
      <td>2.957769e-04</td>
      <td>2012-08-23 18:34:19+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1835</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017-08-27 12:23:23.280713+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2010-08-25 17:57:36+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2564</td>
      <td>2</td>
      <td>130.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2017-08-26 18:51:13.614464+00:00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000061</td>
      <td>1.764451e-52</td>
      <td>0.000000e+00</td>
      <td>6.314252e-04</td>
      <td>2009-04-24 04:21:28+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3053</td>
      <td>1</td>
      <td>154.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2017-08-29 22:42:56.193578+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2014-03-08 21:17:11+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1273</td>
      <td>4</td>
      <td>125.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2017-08-26 12:26:48.742614+00:00</td>
      <td>2.589952e-03</td>
      <td>0.197807</td>
      <td>0.005528</td>
      <td>3.330289e-03</td>
      <td>2.527955e-03</td>
      <td>9.876560e-03</td>
      <td>2013-08-29 21:44:12+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1464</td>
      <td>1</td>
      <td>125.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2017-08-26 15:03:37.124131+00:00</td>
      <td>1.021308e-04</td>
      <td>0.176893</td>
      <td>0.000243</td>
      <td>1.058691e-04</td>
      <td>1.066500e-04</td>
      <td>5.427109e-04</td>
      <td>2015-02-18 23:21:10+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>926</td>
      <td>1</td>
      <td>34.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2017-08-27 04:47:31.659585+00:00</td>
      <td>6.128523e-05</td>
      <td>0.142966</td>
      <td>0.001033</td>
      <td>1.492774e-06</td>
      <td>6.602367e-05</td>
      <td>3.808750e-03</td>
      <td>2010-12-21 05:00:23+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>2447</td>
      <td>2</td>
      <td>118.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2017-08-29 00:35:24.310316+00:00</td>
      <td>6.792954e-05</td>
      <td>0.179065</td>
      <td>0.001033</td>
      <td>2.100564e-04</td>
      <td>7.441400e-05</td>
      <td>1.077327e-03</td>
      <td>2009-05-26 22:02:47+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3020</td>
      <td>4</td>
      <td>157.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>2017-08-26 14:59:23.625120+00:00</td>
      <td>2.462279e-04</td>
      <td>0.145853</td>
      <td>0.000972</td>
      <td>3.079477e-06</td>
      <td>2.612296e-04</td>
      <td>1.813336e-03</td>
      <td>2013-08-22 14:25:25+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1471</td>
      <td>1</td>
      <td>120.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2017-08-26 12:54:18.582976+00:00</td>
      <td>1.862631e-04</td>
      <td>0.158961</td>
      <td>0.001033</td>
      <td>1.465418e-05</td>
      <td>2.042437e-04</td>
      <td>3.921818e-04</td>
      <td>2014-09-22 03:30:33+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1076</td>
      <td>1</td>
      <td>129.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>2017-08-30 13:54:40.717687+00:00</td>
      <td>1.501624e-05</td>
      <td>0.168973</td>
      <td>0.000790</td>
      <td>6.106422e-05</td>
      <td>1.857329e-05</td>
      <td>1.737427e-03</td>
      <td>2012-11-17 20:29:00+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1749</td>
      <td>5</td>
      <td>160.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>68</th>
      <td>2017-08-27 16:24:49.975234+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2011-08-17 17:12:51+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2207</td>
      <td>2</td>
      <td>48.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>69</th>
      <td>2017-08-26 20:35:07.360652+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-07-20 15:27:58+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>43</td>
      <td>1</td>
      <td>94.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>76</th>
      <td>2017-08-26 17:59:00.239952+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2013-09-21 16:23:14+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1441</td>
      <td>1</td>
      <td>91.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>2017-08-27 17:00:42.224858+00:00</td>
      <td>7.360066e-04</td>
      <td>0.157571</td>
      <td>0.000911</td>
      <td>2.240523e-05</td>
      <td>7.627344e-04</td>
      <td>4.387886e-04</td>
      <td>2009-04-22 02:32:14+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3055</td>
      <td>2</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2017-08-27 21:39:10.291725+00:00</td>
      <td>7.895957e-05</td>
      <td>0.169586</td>
      <td>0.001336</td>
      <td>4.698408e-05</td>
      <td>1.020997e-04</td>
      <td>1.335720e-03</td>
      <td>2014-02-03 23:12:22+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1306</td>
      <td>2</td>
      <td>142.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2017-08-30 01:33:46.154708+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-04-12 18:54:25+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>507</td>
      <td>5</td>
      <td>126.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2017-08-27 02:32:43.292861+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009-08-25 18:45:42+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>2929</td>
      <td>2</td>
      <td>28.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2017-08-28 20:19:05.191716+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2014-03-07 22:41:46+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1274</td>
      <td>3</td>
      <td>95.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2017-08-27 01:42:10.880829+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2015-09-02 03:47:22+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>731</td>
      <td>2</td>
      <td>13.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2017-08-27 01:12:50.304603+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-07-10 02:45:45+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>54</td>
      <td>2</td>
      <td>158.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2017-08-30 04:04:46.350055+00:00</td>
      <td>0.000000e+00</td>
      <td>0.122000</td>
      <td>0.000061</td>
      <td>5.360746e-08</td>
      <td>0.000000e+00</td>
      <td>2.894322e-06</td>
      <td>2013-09-18 00:06:18+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1445</td>
      <td>5</td>
      <td>116.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2017-08-31 20:55:01.803093+00:00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000061</td>
      <td>1.764451e-52</td>
      <td>0.000000e+00</td>
      <td>6.938452e-04</td>
      <td>2009-06-08 13:56:36+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>3007</td>
      <td>6</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2017-08-27 02:41:51.619053+00:00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000061</td>
      <td>1.764451e-52</td>
      <td>0.000000e+00</td>
      <td>2.234946e-08</td>
      <td>2014-06-01 14:56:01+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1188</td>
      <td>2</td>
      <td>56.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2017-08-26 21:03:49.946308+00:00</td>
      <td>1.131406e-04</td>
      <td>0.181681</td>
      <td>0.001276</td>
      <td>4.821716e-04</td>
      <td>1.163165e-04</td>
      <td>1.399197e-03</td>
      <td>2009-05-06 14:45:59+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3040</td>
      <td>1</td>
      <td>37.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2017-08-27 12:35:13.107419+00:00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000121</td>
      <td>1.764451e-52</td>
      <td>0.000000e+00</td>
      <td>2.558735e-04</td>
      <td>2009-03-27 15:04:14+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>3080</td>
      <td>2</td>
      <td>159.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2017-08-27 19:23:36.665836+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009-03-20 06:12:46+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>3088</td>
      <td>2</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>108</th>
      <td>2017-08-27 12:36:33.606872+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2011-03-07 16:16:22+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2370</td>
      <td>2</td>
      <td>136.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>115</th>
      <td>2017-08-26 01:17:13.852736+00:00</td>
      <td>2.040063e-04</td>
      <td>0.147549</td>
      <td>0.000243</td>
      <td>3.291531e-06</td>
      <td>2.072710e-04</td>
      <td>4.546241e-05</td>
      <td>2014-10-28 17:25:59+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1039</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>116</th>
      <td>2017-08-26 01:35:10.917436+00:00</td>
      <td>3.197095e-04</td>
      <td>0.170157</td>
      <td>0.001276</td>
      <td>6.562688e-05</td>
      <td>3.339476e-04</td>
      <td>5.621615e-04</td>
      <td>2010-02-18 21:06:24+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2752</td>
      <td>1</td>
      <td>160.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1353</th>
      <td>2017-08-29 04:09:49.219537+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2013-11-30 02:57:04+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1372</td>
      <td>4</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1364</th>
      <td>2017-08-31 20:08:59.293730+00:00</td>
      <td>0.000000e+00</td>
      <td>0.136390</td>
      <td>0.000121</td>
      <td>8.283912e-07</td>
      <td>0.000000e+00</td>
      <td>1.236722e-05</td>
      <td>2009-03-20 06:48:30+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3088</td>
      <td>6</td>
      <td>98.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1373</th>
      <td>2017-08-31 21:12:33.120591+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2015-05-03 20:32:45+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>852</td>
      <td>6</td>
      <td>131.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1381</th>
      <td>2017-08-27 20:34:48.431485+00:00</td>
      <td>4.535512e-05</td>
      <td>0.106133</td>
      <td>0.000121</td>
      <td>1.181880e-09</td>
      <td>4.547175e-05</td>
      <td>1.712531e-05</td>
      <td>2010-08-20 16:33:34+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>2569</td>
      <td>2</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1388</th>
      <td>2017-08-26 14:53:01.794582+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2011-12-05 23:03:14+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2097</td>
      <td>1</td>
      <td>87.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1390</th>
      <td>2017-08-27 01:19:19.470741+00:00</td>
      <td>6.202051e-04</td>
      <td>0.168726</td>
      <td>0.002430</td>
      <td>3.153153e-05</td>
      <td>6.814444e-04</td>
      <td>1.144427e-03</td>
      <td>2015-07-26 15:52:28+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>768</td>
      <td>2</td>
      <td>98.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1394</th>
      <td>2017-08-26 17:17:48.602346+00:00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000182</td>
      <td>1.764451e-52</td>
      <td>0.000000e+00</td>
      <td>6.581222e-04</td>
      <td>2013-09-02 03:54:11+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1461</td>
      <td>1</td>
      <td>69.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1401</th>
      <td>2017-08-26 19:29:13.411402+00:00</td>
      <td>0.000000e+00</td>
      <td>0.128148</td>
      <td>0.000121</td>
      <td>6.172333e-07</td>
      <td>0.000000e+00</td>
      <td>2.706614e-05</td>
      <td>2009-03-13 22:19:48+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3094</td>
      <td>1</td>
      <td>25.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1409</th>
      <td>2017-08-30 03:10:41.592148+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-04-27 02:47:18+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>128</td>
      <td>5</td>
      <td>56.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1418</th>
      <td>2017-08-26 05:49:38.000956+00:00</td>
      <td>3.449831e-04</td>
      <td>0.199186</td>
      <td>0.002248</td>
      <td>1.111846e-02</td>
      <td>3.281119e-04</td>
      <td>1.274399e-02</td>
      <td>2010-03-01 15:12:58+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2741</td>
      <td>1</td>
      <td>53.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1420</th>
      <td>2017-08-27 01:00:03.553351+00:00</td>
      <td>0.000000e+00</td>
      <td>0.158240</td>
      <td>0.000061</td>
      <td>1.457839e-05</td>
      <td>0.000000e+00</td>
      <td>2.258724e-04</td>
      <td>2011-05-03 15:09:39+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>2313</td>
      <td>2</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1422</th>
      <td>2017-08-27 15:01:42.896862+00:00</td>
      <td>1.347667e-04</td>
      <td>0.173490</td>
      <td>0.000607</td>
      <td>1.642878e-03</td>
      <td>1.488943e-04</td>
      <td>5.432849e-03</td>
      <td>2012-01-09 05:27:17+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>2063</td>
      <td>2</td>
      <td>160.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1426</th>
      <td>2017-08-30 16:20:56.361614+00:00</td>
      <td>0.000000e+00</td>
      <td>0.136123</td>
      <td>0.000061</td>
      <td>4.259804e-07</td>
      <td>0.000000e+00</td>
      <td>1.684583e-05</td>
      <td>2013-09-18 21:34:56+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1444</td>
      <td>5</td>
      <td>136.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1431</th>
      <td>2017-08-26 18:06:57.586864+00:00</td>
      <td>0.000000e+00</td>
      <td>0.126605</td>
      <td>0.000121</td>
      <td>1.165049e-07</td>
      <td>0.000000e+00</td>
      <td>1.820440e-06</td>
      <td>2013-03-03 02:17:47+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1644</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1435</th>
      <td>2017-08-27 23:21:15.890220+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009-04-27 09:36:06+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3050</td>
      <td>2</td>
      <td>145.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1445</th>
      <td>2017-08-27 21:20:29.888414+00:00</td>
      <td>0.000000e+00</td>
      <td>0.145347</td>
      <td>0.000121</td>
      <td>1.821303e-06</td>
      <td>0.000000e+00</td>
      <td>1.721403e-05</td>
      <td>2011-09-03 02:25:30+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2191</td>
      <td>2</td>
      <td>56.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1446</th>
      <td>2017-08-27 05:29:56.615508+00:00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000061</td>
      <td>1.764451e-52</td>
      <td>0.000000e+00</td>
      <td>2.526239e-05</td>
      <td>2013-06-02 19:19:49+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1552</td>
      <td>2</td>
      <td>36.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1449</th>
      <td>2017-08-26 11:10:45.231620+00:00</td>
      <td>5.009433e-03</td>
      <td>0.201764</td>
      <td>0.004374</td>
      <td>2.873966e-02</td>
      <td>4.888084e-03</td>
      <td>2.877367e-02</td>
      <td>2007-08-01 20:12:17+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3684</td>
      <td>1</td>
      <td>147.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>2017-08-28 21:18:54.715028+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-02-19 19:37:47+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>194</td>
      <td>3</td>
      <td>94.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1467</th>
      <td>2017-08-26 20:10:19.075625+00:00</td>
      <td>1.806673e-05</td>
      <td>0.120692</td>
      <td>0.000182</td>
      <td>3.469903e-08</td>
      <td>2.370332e-05</td>
      <td>6.818985e-06</td>
      <td>2011-10-24 23:48:59+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>2139</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1470</th>
      <td>2017-08-31 01:24:02.432749+00:00</td>
      <td>1.345656e-04</td>
      <td>0.121710</td>
      <td>0.000243</td>
      <td>6.810232e-08</td>
      <td>1.354154e-04</td>
      <td>8.903805e-06</td>
      <td>2013-05-22 16:10:06+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1563</td>
      <td>6</td>
      <td>146.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1477</th>
      <td>2017-08-26 11:29:22.918598+00:00</td>
      <td>1.416872e-04</td>
      <td>0.161885</td>
      <td>0.000607</td>
      <td>2.035617e-05</td>
      <td>1.528807e-04</td>
      <td>4.802351e-04</td>
      <td>2009-03-31 21:20:48+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3076</td>
      <td>1</td>
      <td>39.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1479</th>
      <td>2017-08-27 13:10:30.982501+00:00</td>
      <td>1.242511e-03</td>
      <td>0.157140</td>
      <td>0.002673</td>
      <td>3.242543e-05</td>
      <td>1.299964e-03</td>
      <td>2.914323e-03</td>
      <td>2009-03-14 00:36:50+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3094</td>
      <td>2</td>
      <td>120.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1484</th>
      <td>2017-09-01 04:54:34.598433+00:00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000061</td>
      <td>1.764451e-52</td>
      <td>0.000000e+00</td>
      <td>2.601490e-04</td>
      <td>2010-07-30 02:24:09+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2591</td>
      <td>7</td>
      <td>24.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1486</th>
      <td>2017-08-26 17:36:11.884877+00:00</td>
      <td>1.121404e-04</td>
      <td>0.165625</td>
      <td>0.000304</td>
      <td>4.475406e-05</td>
      <td>1.141641e-04</td>
      <td>2.071474e-04</td>
      <td>2012-04-12 16:56:14+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1968</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1488</th>
      <td>2017-08-26 15:45:15.874913+00:00</td>
      <td>1.639120e-03</td>
      <td>0.202446</td>
      <td>0.004313</td>
      <td>1.794975e-03</td>
      <td>1.699172e-03</td>
      <td>4.783937e-03</td>
      <td>2010-09-17 13:47:24+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2541</td>
      <td>1</td>
      <td>102.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1490</th>
      <td>2017-08-28 21:39:06.530713+00:00</td>
      <td>6.472373e-04</td>
      <td>0.166932</td>
      <td>0.001458</td>
      <td>2.282133e-05</td>
      <td>6.913189e-04</td>
      <td>1.295462e-03</td>
      <td>2009-11-25 16:23:32+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2837</td>
      <td>3</td>
      <td>128.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1492</th>
      <td>2017-08-28 22:41:15.042343+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-03-20 20:48:10+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>530</td>
      <td>3</td>
      <td>16.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1496</th>
      <td>2017-08-30 06:47:16.777195+00:00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000121</td>
      <td>1.764451e-52</td>
      <td>0.000000e+00</td>
      <td>9.548810e-06</td>
      <td>2012-12-06 04:18:02+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1731</td>
      <td>5</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1499</th>
      <td>2017-08-26 21:12:55.859089+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2013-05-01 15:03:20+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1584</td>
      <td>1</td>
      <td>153.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>386 rows × 46 columns</p>
</div>




```python

```

    1    258
    0    139
    Name: is_coded_as_witness, dtype: int64
    1    258
    0    128
    Name: is_local_profile_location, dtype: int64



```python

```
