
# User Location Classification in Hurricane Harvey
The goal of this analysis is to evaluate methods by which users Tweeting about Hurricane Harvey may be classified as in the area or otherwise.

Data was collected ......

## Data Cleaning & Enrichment


```python
import pandas as pd

# Get account-based codings:
account_codings = Coding.objects.filter(coding_id=1).filter(user__isnull=False).filter(data_code__data_code_id__gt=0)
account_codings_secondary = Coding.objects.filter(coding_id=2).filter(user__isnull=False)

# Check available coding schema:
dimensions = DataCodeDimension.objects.all()[1:]
for d in dimensions:
    print('Coding Dimension: \'{}\'\nSubject: {}\nClasses: {}\n'.format(d.name, d.coding_subject, list(d.datacode.values_list('name', flat=True))))
# Note these totals combine all user or Tweet codes, so can be misleading if more than one dimension is used.
print("{} Accounts coded by primary coder, {} by secondary coder.".format(account_codings.count(), account_codings_secondary.count()))
```


```python
# Get all Users coded by primary coder:
# (exclude data_code_id=0 as this is the temporary 'to be coded' class)
users = User.objects.filter(coding_for_user__coding_id=1, 
                            coding_for_user__data_code__data_code_id__gt=0)
users_df = pd.DataFrame(list(users.values()))
#users_df.count()
```


```python
# Get all authors from coded Tweets:
#coded_tweets = Tweet.objects.filter(coding_for_tweet__coding_id=1, 
#                            coding_for_tweet__data_code__data_code_id__gt=0)
#authors = User.objects.filter(tweet__in=coded_tweets).distinct()
#author_df = pd.DataFrame(list(authors.values()))
```

### Adding Location Data
There are a number of options which can represent the ground truth location of the user.
* Location listed on a user's profile
* User Timezone (deprecated)
* Manual Coding
* Location data derived from Tweet stream
    * GPS tagged Tweets
    * Mention of location in Tweet body

#### Parsing user-set location in profile field
The location the user sets as a string is evaluated and a locality decision made. In this instance, a location is considered 'local' if its coordinates (supplied by the Google geolocation API or parsed directly from the location string) fall within the bounding box used for geographic Twitter data collection, or if it contains the string 'houston' or 'christi' (representing the town Corpus Christi). Both of these locations fall within the bounding box, and are used here as a time-saving operation.

Note that as this field can be set manually, it is unverifiable and therefore not a perfect representation of location, even where it exists. Users may neglect to update their location after moving, and some observations were made of users setting their location to that of a disaster event as a 'show of solidarity'.


```python
## This block supports manual coding of locations as local or non-local.
## It has been superceded by the next block which uses the Googlemaps API

#location_list = users_df.location.unique()
#with open('data/harvey_user_location/all_profile_locations.txt', 'w') as f:
#    for item in location_list:
#        f.write("%s\n" % item)

################
## This list then manually sorted and non-local locations removed.
## List of local locations then re-imported.
## Note this list excludes any locations containing 'Christi' or 'Houston'
## Note: if more users are coded, this list needs to be re-examined. Raise alert:
#if users_df.shape[0] != 931:
#    print('ALERT: New codings detected. Consider updating manual locality selection')
################

#with open('data/harvey_user_location/local_profile_locations_manual_check.txt', 'r') as f:
#    local_locations_list = f.read().splitlines()
    
## Create column for users with local location listed in profile
#users_df['local_profile_location_manual'] = \
#    (users_df.location.str.contains('houston|christi', case=False, regex=True) |
#    users_df.location.isin(local_locations_list))
```


```python
import re

def parse_coordinates(string):
    '''Parse a string for coordinates'''
    reg = '[nsewNSEW]?\s?-?\d+[\.°]\s?\d+°?\s?[nsewNSEW]?'
    result = re.findall(reg, string)
    if len(result) == 2: # Coordinates detected
        for i in range(len(result)):
            # Replace middle degree symbol with decimal:
            reg_middle_degree = '(\d+)°\s?(\d+)'
            result[i] = re.sub(reg_middle_degree, r'\1.\2', result[i])
            # Remove trailing degree symbol, N and E marks:
            reg_strip = '[°neNE\s]'
            result[i] = re.sub(reg_strip, '', result[i])
            # Replace south/west with negative sign:
            reg_replace_sw = '[swSW](\d+\.\d+)|(\d+\.\d+)[swSW]'
            result[i] = re.sub(reg_replace_sw, r'-\1\2', result[i])
            # Remove double negative (where string contained eg. '-99.10w')
            result[i] = re.sub('--', '-', result[i])
        return (float(result[0]), float(result[1]))
    else:
        return False
```


```python
import yaml
import googlemaps


def is_in_bounding_box(coords, boxes):
    '''
    Check whether coordinates fall within defined bounding box:
    Boxes are defined as their NW and SE points.
    '''
    for box in boxes:
        if coords[0] < box[0][0] and coords[0] > box[1][0]:
            if coords[1] > box[0][1] and coords[1] < box[1][1]:
                return True
    return False


def is_local(location, boxes, known_localities=[]):
    '''
    Check whether a location string falls within a set of 
    bounding boxes using Googlemaps API.
    '''
    # Check known localities first to save on API requests:
    for x in known_localities:
        if x in location:
            return True
    # Try and parse coordinates from string rather than API query:
    coords = parse_coordinates(location)
    # Get coords from API:
    if not coords:
        # Get API key from file:
        with open("auth.yml", 'r') as ymlfile:
            auth = yaml.load(ymlfile, Loader=yaml.BaseLoader)
        gmaps = googlemaps.Client(key=auth['apikeys']['googlemaps'])
        #########################################################
        ####### OVERRIDE API OBJECT TO PREVENT API CALLS: #######
        #geocode_result = gmaps.geocode(location)
        geocode_result = False
        print('WARNING -- API DISABLED')
        #########################################################
        #########################################################
        if geocode_result:
            lat = geocode_result[0]['geometry']['location']['lat']
            lon = geocode_result[0]['geometry']['location']['lng']
            coords = (lat, lon)
    if coords:
        return(is_in_bounding_box(coords, boxes))
    return False
```


```python
# Bounding boxes used for Hurricane Harvey dataset:
boxes = [[(29.1197,-99.9590682),(26.5486063,-97.5021)],
        [(30.3893434,-97.5021),(26.5486063,-93.9790001)]]
# Don't need to look these up (save on API requests)
known_localities = ['houston', 'christi']

# Get list of locations in profiles:
location_list = users_df.location.dropna().str.lower().unique()

# Create sublist of local locations
local_location_list = [loc for loc in location_list if is_local(loc, boxes, known_localities)]
# Create sublist of non-local locations (for manual verification)
non_local_location_list = [loc for loc in location_list if loc not in local_location_list]

# Create column for users with local location listed in profile
users_df['local_profile_location'] = users_df.location.str.lower().isin(local_location_list)

# Write lists to file to save calling API on kernel restart:
with open('data/harvey_user_location/local_locations_list_from_api.txt', 'w') as f:
    for item in local_location_list:
        f.write("%s\n" % item)
with open('data/harvey_user_location/non_local_locations_list_from_api.txt', 'w') as f:
    for item in non_local_location_list:
        f.write("%s\n" % item)
```

#### Timezone Field
Timezone data provided by Twitter when capturing the user objects is less specific than other methods, but may be useful as a supplementary source.
As this data field has been deprecated by Twitter, it will not be available in new data sets.


```python
# Create column for profiles in relevant time zone:
timezone = 'Central Time (US & Canada)'
users_df['local_timezone'] = users_df.time_zone == timezone
users_df = users_df.drop(['time_zone', 'utc_offset'], axis=1)
```

#### Manual Coding
Accounts were manually coded as 'local' or 'non-local'.

Coders were shown the user account details as well as the Twitter stream of the user. The coders were instructed to determine whether the user account was in an area affected by the hurricane at any point during the data collection period. Therefore, the term 'local' may be misleading to the reader, as the definition given to the coders will include anyone visiting the area as, for example, a responder or aid worker. This larger set of 'on the ground' users is a more useful target for classification.


```python
# Create column to represent manual coding:
users_df['coded_as'] = \
    users_df['screen_name'].apply(lambda x: User.objects.get(screen_name=x).coding_for_user.get(coding_id=1).data_code.name)
users_df['coded_as_witness'] = users_df['coded_as'] == 'Witness'
# Remove original column:
users_df = users_df.drop(['coded_as'], axis=1)
```

#### GPS from Tweet stream


```python
# Check whether any of a user's Tweets fall within the bounding box and update column:
users_df['tweet_from_locality'] = False
users = users_df.screen_name.tolist()
for u in users:
    try:
        geo_tweets = User.objects.get(screen_name=u).tweet.filter(coordinates_lat__isnull=False)
    except:
        print('Error with user: ', u)
        continue
    for tweet in geo_tweets:
        coords = (tweet.coordinates_lat, tweet.coordinates_lon)
        if is_in_bounding_box(coords, boxes):
            users_df.loc[users_df['screen_name'] == u, 'tweet_from_locality'] = True
            break

```

#### Combination Columns
Combining data from columns may improve accuracy (at the cost of recall)


```python
users_df['three_local_metrics'] = users_df[['tweet_from_locality', 
                    'local_timezone', 'local_profile_location']].all(axis='columns')
users_df['local_tw_and_local_profile'] = users_df[['tweet_from_locality', 
                    'local_profile_location']].all(axis='columns')
users_df['local_tw_and_local_tz'] = users_df[['tweet_from_locality', 
                    'local_timezone']].all(axis='columns')
users_df['local_tz_and_local_profile'] = users_df[['local_profile_location', 
                    'local_timezone']].all(axis='columns')
```


```python
########################################
########################################
## Write/read dataframe to temp file ###
########################################
########################################

import pandas as pd
path = 'data/harvey_user_location/temp_users_df.csv'

#users_df.to_csv(path)

users_df = pd.read_csv(path, index_col=0) 
#users_df = users_df.drop(['Unnamed: 0.1'], axis=1)

# Reading currently splits a row, delete rogue row:
if users_df.shape[0] == 932:
    print('Dropping row 222, to verify')
    users_df = users_df.drop(users_df.index[222])
print(users_df.shape)

########################################
########################################
########################################
```

    (931, 58)


#### Compare Stand-In Metrics
Now we can compare the metrics against the hand-coded classifications to decide whether they are suitable as stand-in values. Note that while the combination columns do increase precision (as expected), they drastically impact recall. 

This suggests that the columns have little correlation.

local_tw_and_local_profile has the lowest drop in pos_cases (i.e. the highest correlation) but we do not see an increase in precision, and do suffer a drop in recall. Therefore these combination columns are not considered useful.


```python
from sklearn.metrics import classification_report

# Columns to compare:
columns = ['tweet_from_locality', 'local_timezone', 'local_profile_location', 
            'local_profile_location_manual', 
           'three_local_metrics', 'local_tw_and_local_profile', 'local_tw_and_local_tz', 
           'local_tz_and_local_profile']

# Create reporting dataframe:
results = pd.DataFrame(columns=['column_name', 'pos_count', 'pos_precision', 'pos_recall', 
                                'accuracy', 'weighted_precision', 'weighted_recall'])

# Fill NA values
users_df.coded_as_witness.fillna(False, inplace=True)

for col in columns:
    users_df[col].fillna(False, inplace=True)
    report = classification_report(users_df['coded_as_witness'], users_df[col], 
                                    output_dict = True)
    row = pd.Series({'column_name': col, 'pos_count': users_df[col].sum(), 
                    'pos_precision': round(report['True']['precision'], 2),
                    'pos_recall': round(report['True']['recall'], 2),
                    'accuracy': round(report['accuracy'], 2),
                    'weighted_precision': round(report['weighted avg']['precision'], 2),
                    'weighted_recall': round(report['weighted avg']['recall'], 2)})
    results = results.append(row, ignore_index=True)


print('Total positive cases: ', users_df['coded_as_witness'].sum())
print('Total negative cases: ', users_df['coded_as_witness'].count() - users_df['coded_as_witness'].sum())
# Print table:
results
```

    Total positive cases:  248
    Total negative cases:  683





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column_name</th>
      <th>pos_count</th>
      <th>pos_precision</th>
      <th>pos_recall</th>
      <th>accuracy</th>
      <th>weighted_precision</th>
      <th>weighted_recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tweet_from_locality</td>
      <td>295</td>
      <td>0.53</td>
      <td>0.62</td>
      <td>0.75</td>
      <td>0.77</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>local_timezone</td>
      <td>222</td>
      <td>0.38</td>
      <td>0.34</td>
      <td>0.68</td>
      <td>0.67</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>2</th>
      <td>local_profile_location</td>
      <td>246</td>
      <td>0.63</td>
      <td>0.62</td>
      <td>0.80</td>
      <td>0.80</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>local_profile_location_manual</td>
      <td>242</td>
      <td>0.64</td>
      <td>0.62</td>
      <td>0.81</td>
      <td>0.81</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>4</th>
      <td>three_local_metrics</td>
      <td>67</td>
      <td>0.69</td>
      <td>0.19</td>
      <td>0.76</td>
      <td>0.74</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>5</th>
      <td>local_tw_and_local_profile</td>
      <td>175</td>
      <td>0.62</td>
      <td>0.44</td>
      <td>0.78</td>
      <td>0.76</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>6</th>
      <td>local_tw_and_local_tz</td>
      <td>97</td>
      <td>0.61</td>
      <td>0.24</td>
      <td>0.76</td>
      <td>0.73</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>7</th>
      <td>local_tz_and_local_profile</td>
      <td>94</td>
      <td>0.67</td>
      <td>0.25</td>
      <td>0.77</td>
      <td>0.75</td>
      <td>0.77</td>
    </tr>
  </tbody>
</table>
</div>



### Misc. Data Enrichment


```python
users_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>added_at</th>
      <th>betweenness_centrality</th>
      <th>closeness_centrality</th>
      <th>created_at</th>
      <th>data_source</th>
      <th>default_profile</th>
      <th>default_profile_image</th>
      <th>degree_centrality</th>
      <th>description</th>
      <th>eigenvector_centrality</th>
      <th>...</th>
      <th>local_profile_location</th>
      <th>local_timezone</th>
      <th>coded_as</th>
      <th>coded_as_witness</th>
      <th>tweet_from_locality</th>
      <th>three_local_metrics</th>
      <th>local_tw_and_local_profile</th>
      <th>local_tw_and_local_tz</th>
      <th>local_tz_and_local_profile</th>
      <th>local_profile_location_manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-08-28 20:42:59.273657+00:00</td>
      <td>0.000043</td>
      <td>0.135798</td>
      <td>2013-03-01 19:23:11+00:00</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>0.000304</td>
      <td>If You Want To Live A Happy Life ❇ change your...</td>
      <td>3.905631e-07</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>Non-Witness</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-08-30 13:58:20.296918+00:00</td>
      <td>0.000015</td>
      <td>0.122066</td>
      <td>2014-01-20 00:34:57+00:00</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>0.000243</td>
      <td>Employee Giving PM @Microsoft.A daydreamer w/ ...</td>
      <td>1.785776e-07</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>Non-Witness</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-08-26 19:51:45.107222+00:00</td>
      <td>0.000000</td>
      <td>0.077120</td>
      <td>2012-07-24 13:47:47+00:00</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>0.000061</td>
      <td>Making an impact isn’t something reserved for ...</td>
      <td>8.518251e-14</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>Non-Witness</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-08-26 11:13:05.769123+00:00</td>
      <td>0.000383</td>
      <td>0.167070</td>
      <td>2010-12-16 17:30:04+00:00</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>0.000668</td>
      <td>Eyeing global entropy through a timeline windo...</td>
      <td>4.315565e-05</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>Non-Witness</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-08-26 14:19:23.604361+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009-04-24 12:08:14+00:00</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>Producer. Show Control Designer. Project Coord...</td>
      <td>NaN</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>Non-Witness</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
# Create column to represent length of profile description:
users_df['description_length'] = users_df.description.str.len()
#users_df = users_df.drop(['description_length'], axis=1)

# Profile language is English:
users_df['lang_is_en'] = users_df['lang'] == 'en'
users_df = users_df.drop(['lang'], axis=1)

# translator_type exists:
users_df['has_translator_type'] = users_df['translator_type'] != 'none'
users_df = users_df.drop(['translator_type'], axis=1)

# Url in profile:
users_df['has_url'] = users_df['url'].notnull()

# User has changed screen_name during collection period:
users_df['changed_screen_name'] = users_df['old_screen_name'].notnull()
```


    ----------------------------------------------------------

    KeyError                 Traceback (most recent call last)

    <ipython-input-219-1694d4515c95> in <module>
          9 # translator_type exists:
         10 users_df['has_translator_type'] = users_df['translator_type'] != 'none'
    ---> 11 users_df = users_df.drop(['lang'], axis=1)
         12 
         13 # Url in profile:


    ~/projects/crisis-data/venv/lib/python3.6/site-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
       3938                                            index=index, columns=columns,
       3939                                            level=level, inplace=inplace,
    -> 3940                                            errors=errors)
       3941 
       3942     @rewrite_axis_style_signature('mapper', [('copy', True),


    ~/projects/crisis-data/venv/lib/python3.6/site-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
       3778         for axis, labels in axes.items():
       3779             if labels is not None:
    -> 3780                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
       3781 
       3782         if inplace:


    ~/projects/crisis-data/venv/lib/python3.6/site-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors)
       3810                 new_axis = axis.drop(labels, level=level, errors=errors)
       3811             else:
    -> 3812                 new_axis = axis.drop(labels, errors=errors)
       3813             result = self.reindex(**{axis_name: new_axis})
       3814 


    ~/projects/crisis-data/venv/lib/python3.6/site-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
       4963             if errors != 'ignore':
       4964                 raise KeyError(
    -> 4965                     '{} not found in axis'.format(labels[mask]))
       4966             indexer = indexer[~mask]
       4967         return self.delete(indexer)


    KeyError: "['lang'] not found in axis"



```python
# Drop columns with only one unique value:
for col in users_df.columns:
    if len(users_df[col].value_counts()) <= 1:
        print('Dropping columns: ', col)
        users_df = users_df.drop([col], axis=1)
        
# TODO: consider dropping where value_counts() == 1, or is the alternative NaN value useful?:
    # protected and ratio_media

```

    Dropping columns:  is_deleted
    Dropping columns:  is_deleted_observed
    Dropping columns:  katz_centrality
    Dropping columns:  needs_phone_verification
    Dropping columns:  protected
    Dropping columns:  ratio_media
    Dropping columns:  suspended
    Dropping columns:  user_followers
    Dropping columns:  user_followers_update
    Dropping columns:  user_following
    Dropping columns:  user_following_update
    Dropping columns:  user_network_update_observed_at



```python
# Create columns to represent age of account at time of detection, and how soon
# after the beginning of the event that the account was first detected.

from datetime import datetime

# Calculate whole days between two dates:
def get_age_in_days(date_str, anchor_date):
    if date_str[-3:-2] == ":":
        date_str = date_str[:-3] + date_str[-2:]
    try:
        datetime_object = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S%z')
    except:
        datetime_object = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f%z')
    return abs((anchor_date - datetime_object).days)

    
# Get dates of event:
e = Event.objects.all()[0]
end = max(e.time_end, e.kw_stream_end, e.gps_stream_end)
start = min(e.time_start, e.kw_stream_start, e.gps_stream_start)


# Create column for age of account at end of data collection period:
users_df['account_age'] = users_df['created_at'].apply(get_age_in_days, args=(end,))

# Create column for how early from beginning of event account was first detected:
users_df['day_of_detection'] = users_df['added_at'].apply(get_age_in_days, args=(start,))
```


```python
##########################################################################################
########## Temporary handling of row with nan for ########################################
########## geo_enabled, has_extended_profile, is_translation_enabled, verified: ##########
##########################################################################################
users_df = users_df.drop(users_df.index[221])
```


    ----------------------------------------------------------

    KeyError                 Traceback (most recent call last)

    ~/projects/crisis-data/venv/lib/python3.6/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2656             try:
    -> 2657                 return self._engine.get_loc(key)
       2658             except KeyError:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 221

    
    During handling of the above exception, another exception occurred:


    KeyError                 Traceback (most recent call last)

    <ipython-input-864-6d0dcdaaf6e8> in <module>
          3 ########## geo_enabled, has_extended_profile, is_translation_enabled, verified: ##########
          4 ##########################################################################################
    ----> 5 users_df[221]
          6 #users_df = users_df.drop(users_df.index[221])


    ~/projects/crisis-data/venv/lib/python3.6/site-packages/pandas/core/frame.py in __getitem__(self, key)
       2925             if self.columns.nlevels > 1:
       2926                 return self._getitem_multilevel(key)
    -> 2927             indexer = self.columns.get_loc(key)
       2928             if is_integer(indexer):
       2929                 indexer = [indexer]


    ~/projects/crisis-data/venv/lib/python3.6/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2657                 return self._engine.get_loc(key)
       2658             except KeyError:
    -> 2659                 return self._engine.get_loc(self._maybe_cast_indexer(key))
       2660         indexer = self.get_indexer([key], method=method, tolerance=tolerance)
       2661         if indexer.ndim > 1 or indexer.size > 1:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 221



```python
# Check columns for categorical candidates:
for col in users_df.columns:
    if len(users_df[col].value_counts()) <= 20:
        if len(users_df[col].unique()) == 2 and 0 in users_df[col].unique() and 1 in users_df[col].unique():
            continue # Already encoded as True/False
        print(col)
        print(users_df[col].unique(),'\n')
```

    data_source
    [1 3 0] 
    
    user_class
    [2. 1.] 
    
    coded_as
    ['Non-Witness' 'Witness' 'Unsure'] 
    
    day_of_detection
    [3 5 1 2 6 4 7 8] 
    



```python
# Encoding categorical columns as one-hot

# Only two user_class categories, so binarise:
users_df['is_user_class_2'] = users_df['user_class'] == 1.
users_df = users_df.drop(['user_class'], axis=1)

# Encode data_source categories (data_source=2 excluded as not extant in this dataset)
# data_source=0 is not encoded as we only need n-1 columns to represent n categories.
users_df['is_data_source_1'] = users_df['data_source'] == 1.
#users_df['is_data_source_2'] = users_df['data_source'] == 2.
users_df['is_data_source_3'] = users_df['data_source'] == 3.
users_df = users_df.drop(['data_source'], axis=1)
```


```python
# Convert True/False columns to 0/1
for col in users_df.columns:
    if len(users_df[col].value_counts()) == 2:
        if True in users_df[col].values and False in users_df[col].values:
            print(col)
            print(users_df[col].unique())
            users_df[col] = users_df[col].astype(int)
```

    default_profile
    [0 1]
    default_profile_image
    [0 1]
    geo_enabled
    [1 0]
    has_extended_profile
    [1 0]
    is_translation_enabled
    [0 1]
    verified
    [0 1]
    local_profile_location
    [0 1]
    local_timezone
    [1 0]
    coded_as_witness
    [0 1]
    tweet_from_locality
    [0 1]
    three_local_metrics
    [0 1]
    local_tw_and_local_profile
    [0 1]
    local_tw_and_local_tz
    [0 1]
    local_tz_and_local_profile
    [0 1]
    local_profile_location_manual
    [0 1]
    lang_is_en
    [1 0]
    has_translator_type
    [1 0]
    is_user_class_2
    [0 1]
    is_data_source_1
    [1 0]
    is_data_source_3
    [0 1]
    has_url
    [0 1]
    changed_screen_name
    [0 1]



```python
# Fix negative values in in_degree and out_degree: an error from data collection:
users_df.loc[users_df['in_degree'] < 0, 'in_degree'] = 0
users_df.loc[users_df['out_degree'] < 0, 'out_degree'] = 0
```

## Classification

The data is now ready for classification. Random forests will be used initially and will suit the relatively small dataset. These will be compared to XGBoost models.


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Display all columns of DF in cell:
pd.set_option('display.max_columns', None)

# Check available columns
users_df.columns
```




    Index(['added_at', 'betweenness_centrality', 'closeness_centrality',
           'created_at', 'default_profile', 'default_profile_image',
           'degree_centrality', 'description', 'eigenvector_centrality',
           'favourites_count', 'followers_count', 'friends_count', 'geo_enabled',
           'has_extended_profile', 'id', 'in_degree', 'is_translation_enabled',
           'listed_count', 'load_centrality', 'location', 'name',
           'old_screen_name', 'out_degree', 'ratio_detected', 'ratio_original',
           'screen_name', 'statuses_count', 'tweets_per_hour',
           'undirected_eigenvector_centrality', 'url', 'user_id', 'verified',
           'local_profile_location', 'local_timezone', 'coded_as',
           'coded_as_witness', 'tweet_from_locality', 'three_local_metrics',
           'local_tw_and_local_profile', 'local_tw_and_local_tz',
           'local_tz_and_local_profile', 'local_profile_location_manual',
           'description_length', 'lang_is_en', 'has_translator_type',
           'account_age', 'day_of_detection', 'is_user_class_2',
           'is_data_source_1', 'is_data_source_3', 'has_url',
           'changed_screen_name'],
          dtype='object')



Columns which are not relevant to the classification problem are dropped, and the test and train partitions are created. 

The column 'local_profile_location_manual' is removed as 'local_profile_location' is preferred for scalability.

Any remaining NAN values are also filled with zeroes. 

The dataframe is also checked for any columns that were not cleaned appropriately (for example, still True/False rather than 1/0)


```python
# Excluded features:
dropped_cols = ['added_at', 'description', 'created_at', 'id',  'location', 'name', 'url', 
                'screen_name', 'user_id', 'coded_as', 'coded_as_witness', 'old_screen_name',
                'local_profile_location_manual']

xVar = users_df.drop(dropped_cols, axis=1)

yVar = users_df['coded_as_witness']

# Partition data sets:
xVar = xVar.fillna(0)
X_train, X_test, y_train, y_test = train_test_split(xVar, yVar, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

xVar.head()
```

    (743, 39) (743,)
    (186, 39) (186,)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>betweenness_centrality</th>
      <th>closeness_centrality</th>
      <th>default_profile</th>
      <th>default_profile_image</th>
      <th>degree_centrality</th>
      <th>eigenvector_centrality</th>
      <th>favourites_count</th>
      <th>followers_count</th>
      <th>friends_count</th>
      <th>geo_enabled</th>
      <th>has_extended_profile</th>
      <th>in_degree</th>
      <th>is_translation_enabled</th>
      <th>listed_count</th>
      <th>load_centrality</th>
      <th>out_degree</th>
      <th>ratio_detected</th>
      <th>ratio_original</th>
      <th>statuses_count</th>
      <th>tweets_per_hour</th>
      <th>undirected_eigenvector_centrality</th>
      <th>verified</th>
      <th>local_profile_location</th>
      <th>local_timezone</th>
      <th>tweet_from_locality</th>
      <th>three_local_metrics</th>
      <th>local_tw_and_local_profile</th>
      <th>local_tw_and_local_tz</th>
      <th>local_tz_and_local_profile</th>
      <th>description_length</th>
      <th>lang_is_en</th>
      <th>has_translator_type</th>
      <th>account_age</th>
      <th>day_of_detection</th>
      <th>is_user_class_2</th>
      <th>is_data_source_1</th>
      <th>is_data_source_3</th>
      <th>has_url</th>
      <th>changed_screen_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000043</td>
      <td>0.135798</td>
      <td>0</td>
      <td>0</td>
      <td>0.000304</td>
      <td>3.905631e-07</td>
      <td>2030.0</td>
      <td>519.0</td>
      <td>1859.0</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>36.0</td>
      <td>0.000048</td>
      <td>3.0</td>
      <td>0.006803</td>
      <td>0.476190</td>
      <td>6042.0</td>
      <td>0.830619</td>
      <td>5.377061e-05</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>96.0</td>
      <td>1</td>
      <td>1</td>
      <td>1645</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000015</td>
      <td>0.122066</td>
      <td>1</td>
      <td>0</td>
      <td>0.000243</td>
      <td>1.785776e-07</td>
      <td>1015.0</td>
      <td>446.0</td>
      <td>661.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>6.0</td>
      <td>0.000019</td>
      <td>0.0</td>
      <td>0.200000</td>
      <td>0.400000</td>
      <td>531.0</td>
      <td>0.028252</td>
      <td>2.210768e-06</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>124.0</td>
      <td>1</td>
      <td>0</td>
      <td>1321</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.077120</td>
      <td>0</td>
      <td>0</td>
      <td>0.000061</td>
      <td>8.518251e-14</td>
      <td>12.0</td>
      <td>277.0</td>
      <td>48.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>492.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.851852</td>
      <td>1.000000</td>
      <td>774.0</td>
      <td>0.152563</td>
      <td>7.589479e-11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>134.0</td>
      <td>1</td>
      <td>0</td>
      <td>1865</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000383</td>
      <td>0.167070</td>
      <td>0</td>
      <td>0</td>
      <td>0.000668</td>
      <td>4.315565e-05</td>
      <td>347.0</td>
      <td>608.0</td>
      <td>496.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>138.0</td>
      <td>0.000388</td>
      <td>0.0</td>
      <td>0.033898</td>
      <td>0.220339</td>
      <td>6383.0</td>
      <td>0.333378</td>
      <td>3.327919e-04</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>128.0</td>
      <td>1</td>
      <td>0</td>
      <td>2451</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>25.0</td>
      <td>321.0</td>
      <td>1687.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>15.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.027855</td>
      <td>0.006685</td>
      <td>8360.0</td>
      <td>10.142596</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>136.0</td>
      <td>1</td>
      <td>0</td>
      <td>3052</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



A baseline RandomForest classifier is created for comparison to parameter-tuned models.


```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=100)
clf.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=-1, oob_score=False, random_state=0, verbose=0,
                           warm_start=False)




```python
from sklearn.metrics import precision_score, recall_score

def get_results(clf):
    # Evaluate Training Data
    pred_train = clf.predict(X_train)
    errors_train = sum(abs(pred_train - y_train))
    train_acc = round((1 - errors_train/len(pred_train)) * 100, 2)
    print('Accuracy on Training Data:', train_acc, '%.')
    # Evaluate Test Data
    pred = clf.predict(X_test)
    errors = sum(abs(pred - y_test))
    test_acc = round((1 - errors/len(pred)) * 100, 2)
    print('Accuracy on Test Data:', test_acc, '%.')
    # Evaluate AUC Score
    auc_score = round(roc_auc_score(y_test, pred), 2)
    print('AUC score: ', auc_score)
    # Confusion Matrix:
    print(pd.crosstab(y_test, pred, rownames=['Actual Result'], colnames=['Predicted Result']))
    # Evaluate Precision & Recall:
    prec = round(precision_score(y_test, pred), 2)
    recall = round(recall_score(y_test, pred), 2)
    return(train_acc, test_acc, auc_score, prec, recall)
```


```python
from sklearn.metrics import roc_auc_score

results_df = pd.DataFrame(columns=['model', 'train_acc', 'test_acc', 'auc', 'precision', 'recall'])
results_df.loc[len(results_df)] = ['rfc_baseline'] + list(get_results(clf))

results_df
```

    Accuracy on Training Data: 100.0 %.
    Accuracy on Test Data: 80.65 %.
    AUC score:  0.73
    Predicted Result    0   1
    Actual Result            
    0                 121  15
    1                  21  29





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>train_acc</th>
      <th>test_acc</th>
      <th>auc</th>
      <th>precision</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>baseline_rfc</td>
      <td>100.0</td>
      <td>80.65</td>
      <td>0.73</td>
      <td>0.66</td>
      <td>0.58</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_jobs': [-1],
    'class_weight': ['balanced'],
    'bootstrap': [True],
#    'max_depth': [2, 10, 20, 50, 100],
    'max_depth': [10, 50, 100],
#    'max_features': [0.1, 0.2, 0.3, 0.4, 'sqrt', 'log2'],
    'max_features': [0.4, 'sqrt', 'log2'],
#    'min_samples_leaf': [1, 2, 3],
#    'min_samples_split': [2, 3, 5],
#    'n_estimators': [10, 15, 20, 40, 60, 100, 200, 300, 400, 500, 600, 700, 800],
    'n_estimators': [300, 400, 600],
    'criterion': ['gini']
}

rfc = RandomForestClassifier() 

clf = GridSearchCV(rfc, param_grid, scoring='roc_auc')
clf.fit(X_train, y_train)
best_clf_rf = clf.best_estimator_
best_params_rf = clf.best_params_

print(best_clf_rf)

results_df.loc[len(results_df)] = ['rfc_gs'] + list(get_results(best_clf_rf))
results_df
```

    RandomForestClassifier(bootstrap=True, class_weight='balanced',
                           criterion='gini', max_depth=10, max_features='log2',
                           max_leaf_nodes=None, min_impurity_decrease=0.0,
                           min_impurity_split=None, min_samples_leaf=1,
                           min_samples_split=2, min_weight_fraction_leaf=0.0,
                           n_estimators=300, n_jobs=-1, oob_score=False,
                           random_state=None, verbose=0, warm_start=False)
    Accuracy on Training Data: 98.92 %.
    Accuracy on Test Data: 77.96 %.
    AUC score:  0.69
    Predicted Result    0   1
    Actual Result            
    0                 120  16
    1                  25  25





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>train_acc</th>
      <th>test_acc</th>
      <th>auc</th>
      <th>precision</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>baseline_rfc</td>
      <td>100.00</td>
      <td>80.65</td>
      <td>0.73</td>
      <td>0.66</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gs_rfc</td>
      <td>98.92</td>
      <td>77.96</td>
      <td>0.69</td>
      <td>0.61</td>
      <td>0.50</td>
    </tr>
  </tbody>
</table>
</div>



### Feature Pruning
We can check the importance of each feature as per the model and use these values to inform the choice of features in future iterations.


```python
feat_importance = sorted(list(zip(X_train, best_clf_rf.feature_importances_.round(decimals=2))), key=lambda x: x[1], reverse=True)
feat_importance
```




    [('local_profile_location', 0.1),
     ('favourites_count', 0.05),
     ('statuses_count', 0.05),
     ('undirected_eigenvector_centrality', 0.05),
     ('tweet_from_locality', 0.05),
     ('description_length', 0.05),
     ('account_age', 0.05),
     ('followers_count', 0.04),
     ('friends_count', 0.04),
     ('listed_count', 0.04),
     ('ratio_detected', 0.04),
     ('ratio_original', 0.04),
     ('tweets_per_hour', 0.04),
     ('day_of_detection', 0.04),
     ('is_data_source_3', 0.04),
     ('closeness_centrality', 0.03),
     ('degree_centrality', 0.03),
     ('local_tw_and_local_profile', 0.03),
     ('is_data_source_1', 0.03),
     ('betweenness_centrality', 0.02),
     ('eigenvector_centrality', 0.02),
     ('load_centrality', 0.02),
     ('out_degree', 0.02),
     ('local_tz_and_local_profile', 0.02),
     ('default_profile', 0.01),
     ('geo_enabled', 0.01),
     ('has_extended_profile', 0.01),
     ('in_degree', 0.01),
     ('local_timezone', 0.01),
     ('three_local_metrics', 0.01),
     ('local_tw_and_local_tz', 0.01),
     ('has_url', 0.01),
     ('default_profile_image', 0.0),
     ('is_translation_enabled', 0.0),
     ('verified', 0.0),
     ('lang_is_en', 0.0),
     ('has_translator_type', 0.0),
     ('is_user_class_2', 0.0),
     ('changed_screen_name', 0.0)]




```python
# Remove features with lowest importance:
features_to_remove = [x[0] for x in feat_importance if x[1] < 0.01]
# We can also remove features which are duplicated:
duplicated_features = ['local_tz_and_local_profile']
features_to_remove += duplicated_features
```


```python
# Temporary storage
X_train_unpruned = X_train
X_test_unpruned = X_test

# Prune
X_train = X_train.drop(features_to_remove, axis=1)
X_test = X_test.drop(features_to_remove, axis=1)

# Use params from previous GridSearch (may not be optimal given pruned set)
clf = RandomForestClassifier().set_params(**best_params_rf)
clf.fit(X_train, y_train)
results_df.loc[len(results_df)] = ['rfc_pruned_0.01'] + list(get_results(clf))


############ Second round of pruning ############

# Remove features with lowest importance:
features_to_remove = [x[0] for x in feat_importance if x[1] < 0.02 and x[1] >= 0.01 and x[0] not in duplicated_features]
X_train = X_train.drop(features_to_remove, axis=1)
X_test = X_test.drop(features_to_remove, axis=1)

# Use params from previous GridSearch (may not be optimal given pruned set)
clf = RandomForestClassifier().set_params(**best_params_rf)
clf.fit(X_train, y_train)
results_df.loc[len(results_df)] = ['rfc_pruned_0.02'] + list(get_results(clf))


# Unprune datasets
X_train = X_train_unpruned
X_test = X_test_unpruned 

results_df
```

    Accuracy on Training Data: 99.33 %.
    Accuracy on Test Data: 80.11 %.
    AUC score:  0.71
    Predicted Result    0   1
    Actual Result            
    0                 123  13
    1                  24  26
    Accuracy on Training Data: 99.6 %.
    Accuracy on Test Data: 80.65 %.
    AUC score:  0.73
    Predicted Result    0   1
    Actual Result            
    0                 122  14
    1                  22  28





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>train_acc</th>
      <th>test_acc</th>
      <th>auc</th>
      <th>precision</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>baseline_rfc</td>
      <td>100.00</td>
      <td>80.65</td>
      <td>0.73</td>
      <td>0.66</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gs_rfc</td>
      <td>98.92</td>
      <td>77.96</td>
      <td>0.69</td>
      <td>0.61</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pruned_0.01_rfc</td>
      <td>99.33</td>
      <td>80.11</td>
      <td>0.71</td>
      <td>0.67</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pruned_0.02_rfc</td>
      <td>99.60</td>
      <td>80.65</td>
      <td>0.73</td>
      <td>0.67</td>
      <td>0.56</td>
    </tr>
  </tbody>
</table>
</div>



### XGBoost


```python
from xgboost import XGBClassifier

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Baseline model
clf = XGBClassifier(max_depth=6, eval_metric='auc')
clf.fit(X_train, y_train)
results_df.loc[len(results_df)] = ['xgb_baseline'] + list(get_results(clf))
```

    Accuracy on Training Data: 100.0 %.
    Accuracy on Test Data: 81.18 %.
    AUC score:  0.74
    Predicted Result    0   1
    Actual Result            
    0                 121  15
    1                  20  30



```python
# Set weights for unbalanced classes as their proportion:
total = sum(y_train.value_counts())
y_weights = y_train.apply(lambda x: (total-y_train.value_counts()[x]) /
                          y_train.value_counts()[x])
clf = XGBClassifier(max_depth=6, eval_metric='auc')
clf.fit(X_train, y_train, sample_weight=y_weights)
results_df.loc[len(results_df)] = ['xgb_weighted'] + list(get_results(clf))

results_df
```

    Accuracy on Training Data: 98.52 %.
    Accuracy on Test Data: 81.18 %.
    AUC score:  0.8
    Predicted Result    0   1
    Actual Result            
    0                 113  23
    1                  12  38





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>train_acc</th>
      <th>test_acc</th>
      <th>auc</th>
      <th>precision</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>baseline_rfc</td>
      <td>100.00</td>
      <td>80.65</td>
      <td>0.73</td>
      <td>0.66</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gs_rfc</td>
      <td>98.92</td>
      <td>77.96</td>
      <td>0.69</td>
      <td>0.61</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pruned_0.01_rfc</td>
      <td>99.33</td>
      <td>80.11</td>
      <td>0.71</td>
      <td>0.67</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pruned_0.02_rfc</td>
      <td>99.60</td>
      <td>80.65</td>
      <td>0.73</td>
      <td>0.67</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>baseline_xgb</td>
      <td>100.00</td>
      <td>81.18</td>
      <td>0.74</td>
      <td>0.67</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>5</th>
      <td>weighted_xgb</td>
      <td>98.52</td>
      <td>81.18</td>
      <td>0.80</td>
      <td>0.62</td>
      <td>0.76</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import GridSearchCV

# negative_instances / positive_instances (for scale_pos_weight) = 
neg_proportion = (len(y_train)-sum(y_train)) / sum(y_train)

param_grid = {
    'max_depth': [6, 7, 8, 10],
#    'max_depth': [6, 10, 30],
    'eval_metric' : ['auc'],
#    'gamma' : [0, 0.5, 1, 2, 5],
    'gamma' : [1],
#    'learning_rate' : [0.05, 0.1, 0.3, 0.5],
    'learning_rate' : [0.05, 0.1],
    'max_delta_step' : [0, 1],
#    'n_estimators' : [5, 10, 50, 100, 200, 300],
    'n_estimators' : [200, 300],
    'n_jobs' : [-1],
    'scale_pos_weight' : [1],
    'min_child_weight' : [0, 1, 5]
}

xgb = XGBClassifier()

clf = GridSearchCV(xgb, param_grid, scoring='roc_auc')
clf.fit(X_train, y_train, sample_weight=y_weights)
best_clf_xgb = clf.best_estimator_
best_params_xgb = clf.best_params_

results_df.loc[len(results_df)] = ['xgb_weighted_gs'] + list(get_results(best_clf_xgb))
print(best_clf_xgb)

results_df
```

    Accuracy on Training Data: 91.39 %.
    Accuracy on Test Data: 79.57 %.
    AUC score:  0.8
    Predicted Result    0   1
    Actual Result            
    0                 108  28
    1                  10  40
    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, eval_metric='auc',
                  gamma=1, learning_rate=0.1, max_delta_step=0, max_depth=8,
                  min_child_weight=5, missing=None, n_estimators=200, n_jobs=-1,
                  nthread=None, objective='binary:logistic', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=None, subsample=1, verbosity=1)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>train_acc</th>
      <th>test_acc</th>
      <th>auc</th>
      <th>precision</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>baseline_rfc</td>
      <td>100.00</td>
      <td>80.65</td>
      <td>0.73</td>
      <td>0.66</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gs_rfc</td>
      <td>98.92</td>
      <td>77.96</td>
      <td>0.69</td>
      <td>0.61</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pruned_0.01_rfc</td>
      <td>99.33</td>
      <td>80.11</td>
      <td>0.71</td>
      <td>0.67</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pruned_0.02_rfc</td>
      <td>99.60</td>
      <td>80.65</td>
      <td>0.73</td>
      <td>0.67</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>baseline_xgb</td>
      <td>100.00</td>
      <td>81.18</td>
      <td>0.74</td>
      <td>0.67</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>5</th>
      <td>weighted_xgb</td>
      <td>98.52</td>
      <td>81.18</td>
      <td>0.80</td>
      <td>0.62</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>6</th>
      <td>gs_weighted_xgb</td>
      <td>91.39</td>
      <td>79.57</td>
      <td>0.80</td>
      <td>0.59</td>
      <td>0.80</td>
    </tr>
  </tbody>
</table>
</div>




```python
feat_importance = sorted(list(zip(X_train, best_clf_xgb.feature_importances_.round(decimals=2))), key=lambda x: x[1], reverse=True)
feat_importance
```




    [('local_profile_location', 0.2),
     ('tweet_from_locality', 0.13),
     ('betweenness_centrality', 0.04),
     ('listed_count', 0.04),
     ('ratio_original', 0.04),
     ('undirected_eigenvector_centrality', 0.04),
     ('day_of_detection', 0.04),
     ('has_url', 0.04),
     ('default_profile', 0.03),
     ('degree_centrality', 0.03),
     ('favourites_count', 0.03),
     ('followers_count', 0.03),
     ('has_extended_profile', 0.03),
     ('ratio_detected', 0.03),
     ('statuses_count', 0.03),
     ('tweets_per_hour', 0.03),
     ('description_length', 0.03),
     ('account_age', 0.03),
     ('closeness_centrality', 0.02),
     ('eigenvector_centrality', 0.02),
     ('friends_count', 0.02),
     ('geo_enabled', 0.02),
     ('in_degree', 0.02),
     ('out_degree', 0.02),
     ('local_timezone', 0.02),
     ('default_profile_image', 0.0),
     ('is_translation_enabled', 0.0),
     ('load_centrality', 0.0),
     ('verified', 0.0),
     ('three_local_metrics', 0.0),
     ('local_tw_and_local_profile', 0.0),
     ('local_tw_and_local_tz', 0.0),
     ('local_tz_and_local_profile', 0.0),
     ('lang_is_en', 0.0),
     ('has_translator_type', 0.0),
     ('is_user_class_2', 0.0),
     ('is_data_source_1', 0.0),
     ('is_data_source_3', 0.0),
     ('changed_screen_name', 0.0)]



Notes from feature importance:
* Geographic features are expectedly the top two features
* The four local combination metrics are not important (three_local_metrics, local_tw_and_local_profile, local_tw_and_local_tz, local_tz_and_local_profile)
* data_source and user_class are unimportant
* betweeness_centrality and undirected_eigenvector_centrality are the most important network metrics
* day_of_detection is more important than expected

### Compare to Model Without Network Metrics
Network metrics are expensive to obtain from the Twitter API, yet they appear to be important to the above models.

We can test equivalent models excluding these features to measure their true value.


```python
# Temporary storage of unpruned dataset
X_train_unpruned = X_train
X_test_unpruned = X_test


features_to_remove = ['undirected_eigenvector_centrality', 'degree_centrality', 'closeness_centrality', 
                      'eigenvector_centrality', 'load_centrality', 'betweenness_centrality']
X_train = X_train.drop(features_to_remove, axis=1)
X_test = X_test.drop(features_to_remove, axis=1)


# Use params from previous GridSearch (may not be optimal given pruned set)
clf = RandomForestClassifier().set_params(**best_params_rf)
clf.fit(X_train, y_train)
results_df.loc[len(results_df)] = ['rfc_no_network'] + list(get_results(clf))

xgb = XGBClassifier().set_params(**best_params_xgb)
xgb.fit(X_train, y_train)
results_df.loc[len(results_df)] = ['xgb_no_network'] + list(get_results(xgb))


# Unprune datasets
X_train = X_train_unpruned
X_test = X_test_unpruned 

results_df
```

    Accuracy on Training Data: 99.6 %.
    Accuracy on Test Data: 79.03 %.
    AUC score:  0.72
    Predicted Result    0   1
    Actual Result            
    0                 119  17
    1                  22  28
    Accuracy on Training Data: 98.52 %.
    Accuracy on Test Data: 79.57 %.
    AUC score:  0.73
    Predicted Result    0   1
    Actual Result            
    0                 118  18
    1                  20  30





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>train_acc</th>
      <th>test_acc</th>
      <th>auc</th>
      <th>precision</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>baseline_rfc</td>
      <td>100.00</td>
      <td>80.65</td>
      <td>0.73</td>
      <td>0.66</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gs_rfc</td>
      <td>98.92</td>
      <td>77.96</td>
      <td>0.69</td>
      <td>0.61</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pruned_0.01_rfc</td>
      <td>99.33</td>
      <td>80.11</td>
      <td>0.71</td>
      <td>0.67</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pruned_0.02_rfc</td>
      <td>99.60</td>
      <td>80.65</td>
      <td>0.73</td>
      <td>0.67</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>baseline_xgb</td>
      <td>100.00</td>
      <td>81.18</td>
      <td>0.74</td>
      <td>0.67</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>5</th>
      <td>weighted_xgb</td>
      <td>98.52</td>
      <td>81.18</td>
      <td>0.80</td>
      <td>0.62</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>6</th>
      <td>gs_weighted_xgb</td>
      <td>91.39</td>
      <td>79.57</td>
      <td>0.80</td>
      <td>0.59</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>7</th>
      <td>no_network_rfc</td>
      <td>99.60</td>
      <td>79.03</td>
      <td>0.72</td>
      <td>0.62</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>8</th>
      <td>no_network_xgb</td>
      <td>98.52</td>
      <td>79.57</td>
      <td>0.73</td>
      <td>0.62</td>
      <td>0.60</td>
    </tr>
  </tbody>
</table>
</div>



As shown above, removing network features from the XGB model causes a significant drop in AUC, precision and recall. The biggest change is in recall, and as this is the most important feature in this application, we must consider collecting these.


```python
xVar.columns
```




    Index(['betweenness_centrality', 'closeness_centrality', 'default_profile',
           'default_profile_image', 'degree_centrality', 'eigenvector_centrality',
           'favourites_count', 'followers_count', 'friends_count', 'geo_enabled',
           'has_extended_profile', 'in_degree', 'is_translation_enabled',
           'listed_count', 'load_centrality', 'out_degree', 'ratio_detected',
           'ratio_original', 'statuses_count', 'tweets_per_hour',
           'undirected_eigenvector_centrality', 'verified',
           'local_profile_location', 'local_timezone', 'tweet_from_locality',
           'three_local_metrics', 'local_tw_and_local_profile',
           'local_tw_and_local_tz', 'local_tz_and_local_profile',
           'description_length', 'lang_is_en', 'has_translator_type',
           'account_age', 'day_of_detection', 'is_user_class_2',
           'is_data_source_1', 'is_data_source_3', 'has_url',
           'changed_screen_name'],
          dtype='object')




```python
# Which features are not available at the time of Tw detection or update throughout event?
```
