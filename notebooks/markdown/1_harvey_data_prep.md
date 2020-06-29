
# User Location Classification in Hurricane Harvey
This is the first notebook in a series which are written primarily as a research logbook for the author. They are therefore not to be considered complete and do not represent the final analysis. For this -- see the final published papers and thesis, or contact the author directly.

The goal of this analysis is to evaluate methods by which users Tweeting about Hurricane Harvey may be classified as in the area or otherwise.

Data was collected with custom software which observed several Twitter streams and enhanced this information by querying the Twitter REST APIs for the network data (friends and followers) of each author. Stream volume which exceeded the capacity of the REST requests was discarded. 
* The keyword stream monitored the terms: [#harvey, #harveystorm, #hurricaneharvey, #corpuschristi]
* The GPS stream used the bounding box: [-99.9590682, 26.5486063, -93.9790001, 30.3893434]
* The collection period ran from 2017-08-26 01:32:18 until 2017-09-02 10:30:52 
* 55,605 Tweets by 33,585 unique authors were recorded

Data was coded using an interface built into the collection software by a primary coder. A secondary coder coded a sub-set of coded users for validation of the coding schema. User instances were coded by whether they 'appeared to be in the affected area'.

These notebooks access the data directly from the database using standard Django query syntax.

# Data Cleaning & Enrichment
In this chapter, we will investigate the data which was collected by the custom software. It is assessed for suitability for machine learning approaches, and enriched with synthesised features. The final result is a dataframe which is ready for statistical techniques which are presented in later chapters.

First we get all the coding instances made by the primary and secondary coders, and check the total codings of each class. There may be multiple coding dimensions (sets of coding schema), in which case the code requires adjustment to constrain to one.


```python
### Initialisation ###
import os
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = [6, 4]

# Location of data files
DIR = './data/harvey_user_location/'

EVENT_NAME = Event.objects.all()[0].name.replace(' ', '')
DF_FILENAME = 'df_users.csv'

# Confirm correct database is set in Django settings.py
if 'Harvey' not in EVENT_NAME:
    raise Exception('Event name mismatch -- check database set in Django')
```


```python
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

# Confirm correct database is set in Django settings.py
print('Dataset: ', EVENT_NAME)
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

    Dataset:  Hurricane Harvey
    Coding Dimension:  Local
    Subject:  user
    Class Totals (primary / secondary): 
    	 Unsure: 	31 	/ 5
    	 Non-Witness: 	1083 	/ 94
    	 Witness: 	386 	/ 52
    1500 Accounts coded by primary coder, 151 by secondary coder.


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




    utc_offset                           920
    time_zone                            920
    undirected_eigenvector_centrality    890
    closeness_centrality                 890
    degree_centrality                    890
    eigenvector_centrality               890
    load_centrality                      890
    betweenness_centrality               890
    url                                  808
    old_screen_name                       98
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
* Various 0 value fields had been added to the database schema but were not implemented at the time of collection. These can be safely dropped.


```python
# Dropping empty columns
empty_cols = users_df.columns[users_df.isnull().all()]
users_df.drop(empty_cols, axis=1, inplace=True)
```

We can also drop columns which have a single value, as they provide no differentiation:


```python
# Drop columns with only one unique value:
for col in users_df.columns:
    if len(users_df[col].value_counts()) <= 1:
        print('Dropping columns: ', col)
        users_df = users_df.drop([col], axis=1)
```

    Dropping columns:  protected
    Dropping columns:  ratio_media
    Dropping columns:  user_class


## Adding Location Data
There are a number of options which can represent the ground truth location of the user.
* Location listed on a user's profile
* User Timezone (deprecated)
* Manual Coding
* Location data derived from Tweet stream
    * GPS tagged Tweets
    * Mention of location in Tweet body
    
### Parsing user-set location in profile field
The location the user sets in their profile as a string is evaluated and a locality decision made. In this instance, a location is considered 'local' if its coordinates (supplied by the Google geolocation API or parsed directly from the location string) fall within the bounding box used for geographic Twitter data collection, or if it contains the string 'houston' or 'christi' (representing the town Corpus Christi). Both of these locations fall within the bounding box, and are used here as a time-saving operation.

Note that as this field can be set manually, it is unverifiable and therefore not a perfect representation of location, even where it exists. Users may neglect to update their location after moving, and some observations were made of users setting their location to that of a disaster event as a 'show of solidarity'.


```python
## This block supports manual coding of locations as local or non-local.
## It has been superceded by the next block which uses the Googlemaps API

#location_list = users_df.location.unique()
#with open('data/harvey_user_location/location_list_all_profile_locations.txt', 'w') as f:
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
        geocode_result = gmaps.geocode(location)
        #geocode_result = False
        #print('WARNING -- API DISABLED')
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
users_df["location"] = users_df["location"].str.lower().str.strip()
location_list = users_df.location.dropna().unique()

# Create sublist of local/non-local locations (non-local only for manual verification)
print("Running is_local() for {} strings...".format(len(location_list)))
local_location_list = [loc for loc in location_list if is_local(loc, boxes, known_localities)]
non_local_location_list = [loc for loc in location_list if loc not in local_location_list]

# Create column for users with local location listed in profile
users_df['local_profile_location'] = users_df.location.str.lower().isin(local_location_list)

# Write lists to file to save calling API on kernel restart:
with open(DIR + 'location_list_from_api_local.txt', 'w') as f:
    for item in local_location_list:
        f.write("%s\n" % item)
with open(DIR + 'location_list_from_api_non_local.txt', 'w') as f:
    for item in non_local_location_list:
        f.write("%s\n" % item)
```


```python
# Use cached locations instead of querying API:
#users_df["location"] = users_df["location"].str.lower().str.strip()
#local_location_list_cached = []
#with open('data/harvey_user_location/location_list_from_api_local.txt', 'r') as f:
#    for line in f:
#        local_location_list_cached.append(line.rstrip('\n'))
#users_df['local_profile_location'] = users_df.location.str.lower().isin(local_location_list)
```

### Timezone Field
Timezone data provided by Twitter when capturing the user objects is less specific than other methods, but may be useful as a supplementary source.
As this data field has been deprecated by Twitter, it will not be available in new data sets.


```python
# View most prevalent time zones:
print(users_df['time_zone'].value_counts().head())

# Create column for profiles in relevant time zone (chosen manually):
relevant_timezone = 'Central Time (US & Canada)'
users_df['local_timezone'] = users_df.time_zone == relevant_timezone
users_df = users_df.drop(['time_zone', 'utc_offset'], axis=1)
```

    Central Time (US & Canada)     341
    Eastern Time (US & Canada)     197
    Pacific Time (US & Canada)     177
    Mountain Time (US & Canada)     37
    Quito                           25
    Name: time_zone, dtype: int64


### Manual Coding
Accounts were manually coded as 'local' or 'non-local'.

Coders were shown the user account details as well as the Twitter stream of the user. The coders were instructed to determine whether the user account was in an area affected by the hurricane at any point during the data collection period. Therefore, the term 'local' may be misleading to the reader, as the definition given to the coders will include anyone visiting the area as, for example, a responder or aid worker. This larger set of 'on the ground' users is a more useful target for classification.


```python
# Create column to represent manual coding:
users_df['coded_as'] = \
    users_df['screen_name'].apply(lambda x: account_codings.get(user__screen_name = x).data_code.name)

# Convert to one-hot encoding
users_df['coded_as_witness'] = users_df['coded_as'] == 'Witness'
users_df['coded_as_non_witness'] = users_df['coded_as'] == 'Non-Witness'

# Remove original column:
users_df = users_df.drop(['coded_as'], axis=1)
```

The 'Unsure' code is represented as `False` values in both the `coded_as_witness` and `coded_as_non_witness` columns. If the 'Unsure' rows are removed, we can also remove the `coded_as_non_witness` column (which is now represented as `False` in the `coded_as_witness` column).
<br /><br />

### GPS from Tweet stream
While the Tweets detected by the system may not contain GPS data, the author may have made other GPS-enabled Tweets during the collection period from which we can infer location. We create a column representing whether the user made any geolocated Tweets within the bounding box during the collection period.


```python
# Check whether any of a user's Tweets fall within the bounding box and update column:
# This will take several minutes to run

users_df['tweet_from_locality'] = False
users = users_df.screen_name.tolist()

for i in range(len(users)):
    if i%100 == 0:
        print('Progress: {} of {}: {}'.format(i, len(users)))
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

    1500
    User 10 of 1500: ChristiWilliams
    User 20 of 1500: OliviaFett
    User 30 of 1500: UHhousing
    User 40 of 1500: SmallTownDicks
    User 50 of 1500: WmBrockschmidt
    User 60 of 1500: marc_ahx
    User 70 of 1500: TamaraChanel
    User 80 of 1500: VotersDemand
    User 90 of 1500: MC_Halo687
    User 100 of 1500: TheBlazeKari
    User 110 of 1500: Ambersallin
    User 120 of 1500: Hamal
    User 130 of 1500: JMarkMcGinnis
    User 140 of 1500: deleon_sarita
    User 150 of 1500: meaneyreport
    User 160 of 1500: touchawe
    User 170 of 1500: Devil_dog_71
    User 180 of 1500: 2020Jobs
    User 190 of 1500: stevenmaislin
    User 200 of 1500: bquentin3
    Error with user:  TCAIS
    User 210 of 1500: Oscarluism_
    User 220 of 1500: hillarybeth
    Error with user:  SEFLCareers
    User 230 of 1500: homesteadraised
    User 240 of 1500: ArmyBtownAD
    User 250 of 1500: MagentaMelee
    User 260 of 1500: thepaigelewis
    User 270 of 1500: meVschristina
    User 280 of 1500: AvvBrosisky
    User 290 of 1500: JeremiahWheele1
    User 300 of 1500: KurtLKrieger
    User 310 of 1500: TrueDumbBlonde
    User 320 of 1500: MrAbdelLHS
    User 330 of 1500: LandlordLinks
    User 340 of 1500: gfbakery2014
    User 350 of 1500: JCP803
    User 360 of 1500: GinoMerlot
    User 370 of 1500: MeSSiaH_808
    User 380 of 1500: callinlexie
    User 390 of 1500: nikkinik528
    User 400 of 1500: JNLIII
    User 410 of 1500: IntentionallyKB
    User 420 of 1500: ezoptical
    User 430 of 1500: CabriLuigi
    User 440 of 1500: Bo_Wright
    User 450 of 1500: itzshelleybell
    User 460 of 1500: RWonMaui
    User 470 of 1500: Felixcalvince
    User 480 of 1500: PatMateluna
    User 490 of 1500: petermlotzemd1
    User 500 of 1500: IAmRobWu
    User 510 of 1500: TeviTroy
    User 520 of 1500: lovelyalica_
    User 530 of 1500: BizarroLuthor
    User 540 of 1500: austintindle
    User 550 of 1500: 2GrkGrls
    User 560 of 1500: weapon83
    User 570 of 1500: corgli
    User 580 of 1500: Heat975Action
    User 590 of 1500: SequoiaRE_EB
    User 600 of 1500: guss813
    User 610 of 1500: fatrat282
    User 620 of 1500: WendyFinneyWood
    User 630 of 1500: JohnnyKey_AR
    User 640 of 1500: SNSPP_PATS_FO_
    User 650 of 1500: NJS4EVER
    User 660 of 1500: ntebdwa
    User 670 of 1500: y81de
    User 680 of 1500: JackieSGolf
    User 690 of 1500: ThingsToShea
    User 700 of 1500: dallasreese
    User 710 of 1500: TurkWarfield
    User 720 of 1500: KaruniaAgung
    User 730 of 1500: catsterle
    User 740 of 1500: Jonathan4743
    User 750 of 1500: joal1969
    User 760 of 1500: youremysonshine
    User 770 of 1500: s_mellors
    User 780 of 1500: iamluvlady
    User 790 of 1500: MerrisBadcock
    User 800 of 1500: VictoriaBBurns
    User 810 of 1500: Margaret8OK
    User 820 of 1500: vicentearenastv
    User 830 of 1500: golivent1
    User 840 of 1500: ConmayS
    User 850 of 1500: EthanEmeryWX
    User 860 of 1500: LeahKWilliams
    User 870 of 1500: sameolg713
    User 880 of 1500: yanaincali
    User 890 of 1500: ChannelE2E
    User 900 of 1500: DJDLuxTx
    User 910 of 1500: SWELGL
    User 920 of 1500: abestesq
    User 930 of 1500: kyle_reidy
    User 940 of 1500: dontgruberme
    User 950 of 1500: BaneSaysTrump
    User 960 of 1500: dariameetsworld
    User 970 of 1500: SharpstownTX
    User 980 of 1500: RogueSquadronMC
    User 990 of 1500: shortmotivation
    User 1000 of 1500: DanielMoralesTV
    User 1010 of 1500: rcantu
    User 1020 of 1500: ZestyOrangePic
    User 1030 of 1500: ElenaRsv
    User 1040 of 1500: BeyondBlunt
    User 1050 of 1500: biominer86
    User 1060 of 1500: MikeImken
    User 1070 of 1500: Rosa_Sherrod
    User 1080 of 1500: wifeofJW
    User 1090 of 1500: AndreaROMANIN1
    User 1100 of 1500: TheBarkingPigTX
    User 1110 of 1500: lesleymesser
    User 1120 of 1500: Renate651
    User 1130 of 1500: BarryMyhawginya
    User 1140 of 1500: PepperBurns1
    User 1150 of 1500: RedPoliticalMan
    User 1160 of 1500: MrsBoothSays
    User 1170 of 1500: Foundry_HB
    User 1180 of 1500: VenatoreMedia
    User 1190 of 1500: KimiKentMusic
    User 1200 of 1500: SociologyofCC
    User 1210 of 1500: mlweldon5
    User 1220 of 1500: _ItEndsNow_
    User 1230 of 1500: LexeyJohnson
    User 1240 of 1500: not_DonnyTrump
    User 1250 of 1500: plsimmo
    User 1260 of 1500: lillamb1997
    User 1270 of 1500: JayleenHeft
    User 1280 of 1500: PrayerPictures
    User 1290 of 1500: HoustonISDGov
    User 1300 of 1500: BrownsteinHyatt
    User 1310 of 1500: olivesjoy
    User 1320 of 1500: bellinissima
    User 1330 of 1500: SUSANIRELAND4
    User 1340 of 1500: onlndtng_sucks
    User 1350 of 1500: DiscoverDior
    User 1360 of 1500: SMLaughna
    User 1370 of 1500: TweetsfromMsB
    User 1380 of 1500: ConveyerOfCool
    User 1390 of 1500: ocuellar10
    User 1400 of 1500: BusinessMoney5
    User 1410 of 1500: Ejwhite0
    User 1420 of 1500: Justin_Horne
    User 1430 of 1500: t3hasian
    User 1440 of 1500: palomaresjl
    User 1450 of 1500: SMUTexasMexico
    User 1460 of 1500: WeAreGoLocal
    User 1470 of 1500: ScoopALoop3
    User 1480 of 1500: JennyWCVB
    User 1490 of 1500: GalvHistory
    User 1500 of 1500: acman2k6


###  Temporary Export
The dataframe is exported to a csv file before further manipulation (and column dropping) to avoid repeating the expensive tasks above.


```python
import pandas as pd
filename = 'df_users_interim.csv'

# Sanitise description field:
users_df["description"] = users_df["description"].str.replace("\r", " ")
users_df.to_csv(DIR + filename)
```


```python
# Re-import and check rows match:
orig_shape = users_df.shape
users_df_temp = pd.read_csv(path, index_col=0)
if users_df_temp.shape == orig_shape:
    users_df = users_df_temp
else:
    print("Shape mis-match. Check string sanitisation")
```

## Data Enrichment
New features are synthesised from existing data.


Columns are added which represent the age of the user account (at the time of original collection) and at which point during the event the account was first detected by the stream.


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

An error in data collection allowed some `in_degree` and `out_degree` values to become negative. These are adjusted to 0. 

Note: It is possible that other entries have values one lower than what they should be.


```python
# Fix negative values in in_degree and out_degree: an error from data collection:
users_df.loc[users_df['in_degree'] < 0, 'in_degree'] = 0
users_df.loc[users_df['out_degree'] < 0, 'out_degree'] = 0
```

## Feature Encoding
Qualitative columns are converted into formats interpretable by a model.


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

We check for columns which should be represented categorically by counting the unique values in each column (under the assumption that categorical variables will have fewer than 20 unique values):


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
    [1 3] 
    
    day_of_detection
    [3 5 1 2 6 4 7 8] 
    


`day_of_detection` is an ordinal feature and is therefore untouched.

`data_source` is a categorical feature, so we re-encode it as a binary value. Here, the values of 1 and 3 are arbitrary (the 2 value was not implemented in the collection process).


```python
# Encoding categorical columns as one-hot:
# data_source==1 is not encoded as we only need n-1 columns to represent n categories.
users_df['is_data_source_3'] = users_df['data_source'] == 3.
users_df = users_df.drop(['data_source'], axis=1)
```

True/False columns are converted to 1/0 values for model compatibility


```python
# Convert True/False columns to 1/0
print('Converting columns from boolean to binary:\n')
for col in users_df.columns:
    if (len(users_df[col].value_counts()) == 2 and 
            True in users_df[col].values and 
            False in users_df[col].values):
        print(col)
        users_df[col] = users_df[col].astype(int)
```

    Converting columns from boolean to binary:
    
    default_profile
    default_profile_image
    geo_enabled
    has_extended_profile
    is_translation_enabled
    verified
    local_profile_location
    local_timezone
    coded_as_witness
    coded_as_non_witness
    tweet_from_locality
    lang_is_en
    has_translator_type
    has_url
    changed_screen_name
    is_data_source_3


## Export to File


```python
users_df.to_csv(DIR + DF_FILENAME)

# Re-import and check rows match:
orig_shape = users_df.shape
users_df_temp = pd.read_csv(path, index_col=0)
if users_df_temp.shape == orig_shape:
    users_df = users_df_temp
    print('Dataframe exported to CSV.')
    print(users_df.shape)
else:
    print("Shape mis-match. Check string sanitisation")

```

    Dataframe exported to CSV.
    (1500, 45)

