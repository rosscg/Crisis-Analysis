
# User Location Classification in Hurricane Harvey
The goal of this analysis is to evaluate methods by which users Tweeting about Hurricane Harvey may be classified as in the area or otherwise.

Data was collected with custom software which observed several Twitter streams and enhanced this information by querying the Twitter REST APIs for the network data (friends and followers) of each author. Stream volume which exceeded the capacity of the REST requests was discarded. 
* The keyword stream monitored the terms: [#harvey, #harveystorm, #hurricaneharvey, #corpuschristi]
* The GPS stream used the bounding box: [-99.9590682, 26.5486063, -93.9790001, 30.3893434]
* The collection period ran from 2017-08-26 01:32:18 until 2017-09-02 10:30:52 
* 55,605 Tweets by 33,585 unique authors were recorded

Data was coded using an interface built into the collection software by a primary coder. A secondary coder coded a sub-set of coded users for validation of the coding schema. User instances were coded by whether they 'appeared to be in the affected area'.


## Data Cleaning & Enrichment
First we get all the coding instances made by the primary and secondary coders, and check the total codings of each class. There may be multiple coding dimensions (sets of coding schema), in which case the code requires adjustment to constrain to one.


```python
import pandas as pd

# Get coding instances of user objects:
account_codings = (Coding.objects
                    .filter(coding_id=1)
                    .filter(user__isnull=False)
                    .filter(data_code__data_code_id__gt=0)
                  )
account_codings_secondary = (Coding.objects
                                 .filter(coding_id=2)
                                 .filter(user__isnull=False)
                            )

# Check available coding schema:
dimensions = DataCodeDimension.objects.all()[1:]
for d in dimensions:
    print('Coding Dimension: ', d.name)
    print('Subject: ', d.coding_subject)
    print('Class Totals (primary / secondary): ')
    for code in d.datacode.all():
        print("\t {}: \t{} \t/ {}"
                .format(code.name, 
                    account_codings.filter(data_code__id=code.id).count(), 
                    account_codings_secondary.filter(data_code__id=code.id).count())
             )
print("{} Accounts coded by primary coder, {} by secondary coder.".format(account_codings.count(), account_codings_secondary.count()))
if len(dimensions) > 1: 
    print('\tNote: Totals represent sum of all codes from {} dimensions.'.format(len(dimensions)))
    print('WARNING: Code in cells below assume one dimension -- adjust to constrain.')
```

    Coding Dimension:  Local
    Subject:  user
    Class Totals (primary / secondary): 
    	 Unsure: 	31 	/ 5
    	 Non-Witness: 	1070 	/ 94
    	 Witness: 	378 	/ 52
    1479 Accounts coded by primary coder, 151 by secondary coder.


We then create a dataframe of all users which have been coded by the primary coder to create the initial dataset. The subjects of the secondary coder are a subset of this set by design.


```python
# Get all Users coded by primary coder:
# (exclude data_code_id=0 as this is the temporary 'to be coded' class)
users = User.objects.filter(coding_for_user__coding_id=1, 
                            coding_for_user__data_code__data_code_id__gt=0)
users_df = pd.DataFrame(list(users.values()))

# Check for missing values by column:
users_df.count()[users_df.count() != account_codings.count()].sort_values(ascending=False)
```




    utc_offset                           906
    time_zone                            906
    undirected_eigenvector_centrality    878
    closeness_centrality                 878
    degree_centrality                    878
    eigenvector_centrality               878
    load_centrality                      878
    betweenness_centrality               878
    url                                  799
    old_screen_name                       95
    suspended                              0
    user_network_update_observed_at        0
    needs_phone_verification               0
    user_followers                         0
    katz_centrality                        0
    is_deleted_observed                    0
    is_deleted                             0
    user_followers_update                  0
    user_following                         0
    user_following_update                  0
    dtype: int64



* The centrality measures have a common value. As these values are only calculated for the largest connected component of the graph, this consistency makes sense.
* The fields time_zone, utc_offset, old_screen_name, and url are nullable.
* Twitter has deprecated the field 'needs_phone_verification', so no values were returned.
* Various 0 value fields were not implemented at the time of collection but have been added to the database. These can be safely dropped.


```python
# Dropping empty columns
dropped_cols = users_df.columns[users_df.isnull().all()]
users_df.drop(dropped_cols, axis=1, inplace=True)
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
The location the user sets in their profile as a string is evaluated and a locality decision made. In this instance, a location is considered 'local' if its coordinates (supplied by the Google geolocation API or parsed directly from the location string) fall within the bounding box used for geographic Twitter data collection, or if it contains the string 'houston' or 'christi' (representing the town Corpus Christi). Both of these locations fall within the bounding box, and are used here as a time-saving operation.

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
    if not location:
        return False
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
While the Tweets detected by the system may not contain GPS data, the author may have made other GPS-enabled Tweets during the collection period from which we can infer location. We createa column representing whether the user made any geolocated Tweets within the bounding box during the collection period.


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

#users_df = pd.read_csv(path, index_col=0) 
#users_df = users_df.drop(['Unnamed: 0.1'], axis=1)

# Reading currently splits a row, delete rogue row:
#if users_df.shape[0] == 1500:
#    print('Dropping row 222, to verify')
#    users_df = users_df.drop(users_df.index[222])
#print(users_df.shape)

########################################
########################################
########################################
```


```python
users_df.shape
```




    (1478, 56)



#### Compare Stand-In Metrics
Now we can compare the metrics against the hand-coded classifications to decide whether they are suitable as stand-in values. Note that while the combination columns do increase precision (as expected), they drastically impact recall. 

This suggests that the columns have little correlation.

local_tw_and_local_profile has the lowest drop in pos_cases (i.e. the highest correlation) but we do not see an increase in precision, and do suffer a drop in recall. Therefore these combination columns are not considered useful.


```python
from sklearn.metrics import classification_report

# Columns to compare:
columns = ['tweet_from_locality', 'local_timezone', 'local_profile_location', 
            #'local_profile_location_manual', 
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

    Total positive cases:  377
    Total negative cases:  1101





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
      <td>462</td>
      <td>0.53</td>
      <td>0.65</td>
      <td>0.77</td>
      <td>0.78</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>1</th>
      <td>local_timezone</td>
      <td>335</td>
      <td>0.41</td>
      <td>0.37</td>
      <td>0.71</td>
      <td>0.69</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>2</th>
      <td>local_profile_location</td>
      <td>393</td>
      <td>0.64</td>
      <td>0.67</td>
      <td>0.82</td>
      <td>0.82</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>3</th>
      <td>three_local_metrics</td>
      <td>105</td>
      <td>0.69</td>
      <td>0.19</td>
      <td>0.77</td>
      <td>0.75</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>local_tw_and_local_profile</td>
      <td>277</td>
      <td>0.64</td>
      <td>0.47</td>
      <td>0.80</td>
      <td>0.79</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>5</th>
      <td>local_tw_and_local_tz</td>
      <td>150</td>
      <td>0.63</td>
      <td>0.25</td>
      <td>0.77</td>
      <td>0.75</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>6</th>
      <td>local_tz_and_local_profile</td>
      <td>153</td>
      <td>0.66</td>
      <td>0.27</td>
      <td>0.78</td>
      <td>0.76</td>
      <td>0.78</td>
    </tr>
  </tbody>
</table>
</div>



### Misc. Data Enrichment


```python
users_df.columns
```




    Index(['added_at', 'betweenness_centrality', 'closeness_centrality',
           'created_at', 'data_source', 'default_profile', 'default_profile_image',
           'degree_centrality', 'description', 'eigenvector_centrality',
           'favourites_count', 'followers_count', 'friends_count', 'geo_enabled',
           'has_extended_profile', 'id', 'in_degree', 'is_deleted',
           'is_deleted_observed', 'is_translation_enabled', 'katz_centrality',
           'lang', 'listed_count', 'load_centrality', 'location', 'name',
           'needs_phone_verification', 'old_screen_name', 'out_degree',
           'protected', 'ratio_detected', 'ratio_media', 'ratio_original',
           'screen_name', 'statuses_count', 'suspended', 'translator_type',
           'tweets_per_hour', 'undirected_eigenvector_centrality', 'url',
           'user_class', 'user_followers', 'user_followers_update',
           'user_following', 'user_following_update', 'user_id',
           'user_network_update_observed_at', 'verified', 'local_profile_location',
           'local_timezone', 'coded_as_witness', 'tweet_from_locality',
           'three_local_metrics', 'local_tw_and_local_profile',
           'local_tw_and_local_tz', 'local_tz_and_local_profile'],
          dtype='object')




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
    date_str = str(date_str) # In case date_str is already a datetime obj
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
#users_df = users_df.drop(users_df.index[221])
```


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
    [2 1] 
    
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
    [False  True]
    default_profile_image
    [False  True]
    geo_enabled
    [ True False]
    has_extended_profile
    [ True False]
    is_translation_enabled
    [False  True]
    verified
    [False  True]
    local_profile_location
    [False  True]
    local_timezone
    [ True False]
    coded_as_witness
    [False  True]
    tweet_from_locality
    [False  True]
    three_local_metrics
    [False  True]
    local_tw_and_local_profile
    [False  True]
    local_tw_and_local_tz
    [False  True]
    local_tz_and_local_profile
    [False  True]
    lang_is_en
    [ True False]
    has_translator_type
    [ True False]
    has_url
    [False  True]
    changed_screen_name
    [False  True]
    is_user_class_2
    [False  True]
    is_data_source_1
    [ True False]
    is_data_source_3
    [False  True]



```python
# Fix negative values in in_degree and out_degree: an error from data collection:
users_df.loc[users_df['in_degree'] < 0, 'in_degree'] = 0
users_df.loc[users_df['out_degree'] < 0, 'out_degree'] = 0
```


```python
########################################
########################################
## Write/read dataframe to temp file ###
########################################
########################################

import pandas as pd
path = 'data/harvey_user_location/cleaned_users_df.csv'

users_df.to_csv(path)

#users_df = pd.read_csv(path, index_col=0) 
#users_df = users_df.drop(['Unnamed: 0.1'], axis=1)

print(users_df.shape)

########################################
########################################
########################################
```

    (1499, 50)

