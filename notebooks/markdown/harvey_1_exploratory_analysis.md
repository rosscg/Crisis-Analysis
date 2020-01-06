
# Exploratory Analysis of Hurricane Harvey Data

This notebook contains an initial exploration of the data collected from Twitter.


```python
from streamcollect.models import Tweet, User, Event, Hashtag, Url, Mention
```

Check Event details:


```python
e = Event.objects.all()[0]
print(e.name)
print(min(e.kw_stream_start, e.gps_stream_start))
td = e.kw_stream_end - e.kw_stream_start
print('Capture window (kw): {} days, {} hours, {} minutes'.format(td.days, td.seconds//3600, td.seconds//60%60))
td = e.gps_stream_end - e.gps_stream_start
print('Capture window (geo): {} days, {} hours, {} minutes'.format(td.days, td.seconds//3600, td.seconds//60%60))
tracked_kws = Event.objects.all()[0].keyword.all().values_list('keyword', flat=True)
print('Tracked keywords: {}'.format(tracked_kws))
```

Print fields for the Tweet model for reference:


```python
[f.name for f in Tweet._meta.get_fields()]
```

Count the Tweets by `data_source` value (i.e. whether they were detected by the stream or added).

* -1 = Identified as spam/irrelevant
* 0 = Sourced from rest API, or quotes/replied_to Tweets
* 1 = Low-priority keyword stream
* 2 = High-priority keyword stream
* 3 = Geo stream (contains coordinates)
* 4 = Geo stream (does not contain coordinates, but Place object falls within bounding box)

Note that some older datasets may not adhere strictly to the above. E.g. for the Hurricane Harvey dataset, all keyword streamed Tweets are `data_source = 1`.


```python
print("Total Tweets: {}".format(Tweet.objects.all().count()))
for n in range(-1,5):
    print("data_source = {}: {}".format(n, Tweet.objects.filter(data_source=n).count()))
```

## Tweet Content
First, we check the proportion of Tweets (by stream) which contain URLs, mentions, and hashtags. As media was not stored during the Harvey data collection, we cannot test this proportion.

Also consider testing hashtags excluding those already tracked.


```python
def print_proportions(queryset):
    total = queryset.count()
    print('Contains hashtag: {}'.format(queryset.filter(hashtags__isnull=False).distinct().count()))
    print('{0:.2%}'.format(queryset.filter(hashtags__isnull=False).distinct().count()/total))
    print('No hashtag: {}'.format(queryset.filter(hashtags__isnull=True).count()))
    print('Contains URL: {}'.format(queryset.filter(urls__isnull=False).distinct().count()))
    print('{0:.2%}'.format(queryset.filter(urls__isnull=False).distinct().count()/total))
    print('No URL: {}'.format(queryset.filter(urls__isnull=True).count()))
    print('Contains mention: {}'.format(queryset.filter(mentions__isnull=False).distinct().count()))
    print('{0:.2%}'.format(queryset.filter(mentions__isnull=False).distinct().count()/total))
    print('No mention: {}'.format(queryset.filter(mentions__isnull=True).count()))
    print('Total: {}'.format(total))

for data_source in [1,3]:
    print('\nChecking source = {}'.format(data_source))
    queryset = Tweet.objects.filter(data_source=data_source)
    print_proportions(queryset)

print('\nChecking Overall...')
queryset = Tweet.objects.filter(data_source__gte=1)
print_proportions(queryset)

print('\nSource = 1, excluding Instagram...')
queryset = Tweet.objects.filter(data_source=1).exclude(source='Instagram')
print_proportions(queryset)
print('\nSource = 3, excluding Instagram...')
queryset = Tweet.objects.filter(data_source=3).exclude(source='Instagram')
print_proportions(queryset)
print('\nOverall, excluding Instagram...')
queryset = Tweet.objects.filter(data_source__gte=1).exclude(source='Instagram')
print_proportions(queryset)
```

There is an issue where some Tweets from source=1 do not contain hashtag objects, which may have been an error during storage to database, or in the way the Tweet is formatted.

Examples:


```python
for t in Tweet.objects.filter(data_source=1).filter(hashtags__isnull=True)[:5]:
    print(t.text)
```

Check proportion of Tweets which include coordinates. It is expected that this is 100% for the data_source=3 stream.


```python
def gps_coordinates(queryset):
    print('Geo-tag count: {}'.format(queryset.filter(coordinates_type__isnull=False).count()))
    print('{0:.2%}'.format(queryset.filter(coordinates_type__isnull=False).count()/queryset.count()))

print('Number of Tweets including coordinates:')
print('\nData source = 1:')
queryset = Tweet.objects.filter(data_source=1)
gps_coordinates(queryset)
print('\nData source = 3:')
queryset = Tweet.objects.filter(data_source=3)
gps_coordinates(queryset)
print('\nOverall:')
queryset = Tweet.objects.filter(data_source__gte=1)
gps_coordinates(queryset)

print('\nData source = 3, exlcuding source = instagram:')
queryset = Tweet.objects.filter(data_source=3).exclude(source='Instagram')
gps_coordinates(queryset)
print('\nData source = 1, exlcuding source = instagram:')
queryset = Tweet.objects.filter(data_source=1).exclude(source='Instagram')
gps_coordinates(queryset)
print('\nOverall, exlcuding source = instagram:')
queryset = Tweet.objects.filter(data_source__gte=1).exclude(source='Instagram')
gps_coordinates(queryset)
```

Checking which mentions, urls and hashtags are most common amongst the dataset:

### Hashtags


```python
from django.db.models import Case, IntegerField, Sum, When

max_items = 20

# Get keywords which were tracked to exlude them from count.
keywords = Keyword.objects.all().values_list('keyword', flat=True)
print(keywords)
# Remove preceding hash, as hashes are excluded from hashtag objects.
keywords = [k[1:] for k in keywords]

# Count all Tweets:
hashtags_with_counts = Hashtag.objects.exclude(hashtag__in=keywords)\
    .annotate(tweet_count=Count('tweets__id')).order_by('-tweet_count')[:max_items]
# Count for filtered Tweets:
hashtags_with_counts_kw = Hashtag.objects.exclude(hashtag__in=keywords)\
    .annotate(tweet_count=Sum(Case(When(\
        tweets__data_source=1, 
    then=1), default=0, output_field=IntegerField()))).order_by('-tweet_count')[:max_items]
# Count for filtered Tweets:
hashtags_with_counts_geo = Hashtag.objects.exclude(hashtag__in=keywords)\
    .annotate(tweet_count=Sum(Case(When(\
        tweets__data_source=3, 
    then=1), default=0, output_field=IntegerField()))).order_by('-tweet_count')[:max_items]

print('\nKeyword Hashtag counts:')
for v in hashtags_with_counts_kw:
    print('{}: {}'.format(v.tweet_count, v))
print('\nGeo Hashtag counts:')
for v in hashtags_with_counts_geo:
    print('{}: {}'.format(v.tweet_count, v))
#print('\nOverall Hashtag counts:')
#for v in hashtags_with_counts:
#    print('{}: {}'.format(v.tweet_count, v))    
```

Of the hashtags above, most can be classified as 'regional / referring to the event'. There are a number of unrelated tags from the automated job adverts picked up by the geo stream. #trump and #climatechange come from discussion about the event.

#veterans and #repost are not obvious in their meaning. We can check some of these Tweets to understand the discussion around these tags.

The below output shows that #veterans is used by job advert platforms and #repost appears to be a convention on Instagram to represent what, on Twitter, would be called a Retweet.


```python
veteran_ts = Hashtag.objects.filter(hashtag='veterans')[0].tweets.all()
repost_ts = Hashtag.objects.filter(hashtag='repost')[0].tweets.all()

print('\n===========\n#Veterans:\n===========')
for t in veteran_ts[:5]:
    print(t.text)
    #print('https://twitter.com/{}/status/{}\n'.format(t.author.screen_name, t.tweet_id))

print('\n===========\n#Repost:\n===========')
for t in repost_ts[:5]:
    print(t.text)
    print('https://twitter.com/{}/status/{}'.format(t.author.screen_name, t.tweet_id))
```

### URLs


```python
# Aggregating URLs

# Attempt to unwind URLs before printing.
import requests

# Remove arguments from URL (exluded as arguments contain identifying information)
#import re
#prog = re.compile('(.*)(?=\?)')

# Count all Tweets:
#urls_with_counts = Url.objects.all()\
#    .annotate(tweet_count=Count('tweets__id')).order_by('-tweet_count')[:max_items]
# Count for filtered Tweets:
urls_with_counts_kw = Url.objects.all()\
    .annotate(tweet_count=Sum(Case(When(\
        tweets__data_source=1, 
    then=1), default=0, output_field=IntegerField()))).order_by('-tweet_count')[:max_items+10]
# Count for filtered Tweets:
urls_with_counts_geo = Url.objects.all()\
    .annotate(tweet_count=Sum(Case(When(\
        tweets__data_source=3, 
    then=1), default=0, output_field=IntegerField()))).order_by('-tweet_count')[:max_items]

print('\nKeyword URL counts:')
for v in urls_with_counts_kw:
    r = requests.get('http://' + str(v))
    url = r.url
    #result = prog.match(r.url)
    if not url:
        url = 'http://' + str(v)
    #url = result.group(0)
    print('{}: {} \t{}'.format(v.tweet_count, url, v))
    
print('\n Geo URL counts:')
for v in urls_with_counts_geo:
    r = requests.get('http://' + str(v))
    url = r.url
    if not url:
        url = 'http://' + str(v)
    print('{}: {} \t{}'.format(v.tweet_count, url, v))
```


```python
# Checking Tweets including a URL manually:
url = 'redcross.org'
ts = Url.objects.filter(url=url)[0].tweets.all()
for t in ts:
    print('\nAuthor: {}'.format(t.author))
    print(t)
```

### Mentions


```python
# Aggregating Mentions

# Count all Tweets:
mentions_with_counts = Mention.objects.all() \
    .annotate(tweet_count=Count('tweets__id')).order_by('-tweet_count')[:max_items]
# Count for filtered Tweets:
mentions_with_counts_kw = Mention.objects.all().annotate(tweet_count=Sum(Case(When(
        tweets__data_source=1, 
    then=1), default=0, output_field=IntegerField()))).order_by('-tweet_count')[:max_items]
# Count for filtered Tweets:
mentions_with_counts_geo = Mention.objects.all().annotate(tweet_count=Sum(Case(When(
        tweets__data_source=3, 
    then=1), default=0, output_field=IntegerField()))).order_by('-tweet_count')[:max_items]

print('\nKeyword URL counts:')
for v in mentions_with_counts_kw:
    print('{0}: {1:.0%} {2} \t\thttps://twitter.com/{3}'.format(v.tweet_count, v.tweets.filter(data_source=1).values("author").distinct().count()/v.tweet_count, v, v))
print('\n Geo URL counts:')
for v in mentions_with_counts_geo:
    print('{0}: {1:.0%} {2} \t\thttps://twitter.com/{3}'.format(v.tweet_count, v.tweets.filter(data_source=3).values("author").distinct().count()/v.tweet_count, v, v))
#print('\nOverall URL counts:')
#for v in mentions_with_counts:
#    print('{}: {} \t\thttps://twitter.com/{}'.format(v.tweet_count, v, v))
```


```python
# Checking Tweets including a Mention manually:
mention = 'texastoyzz'
ts = Mention.objects.filter(mention=mention)[0].tweets.all()
for t in ts:
    print('\nAuthor: {}'.format(t.author))
    print(t)

#Mention.objects.filter(mention=mention)[0].tweets.all().values("author").distinct().count()

```

## Tweet Source
The source of a Tweet may be a suitable proxy for some level of user classification, as some businesses may use automated platforms.

Check whether there are null values for source field, as these would have to be counted seperately:


```python
Tweet.objects.filter(data_source__gt=0).filter(source__isnull=True).count()
```

### Proportions by Source


```python
from django.db.models import Count
import pandas as pd
```


```python
# Get all streamed Tweets
tweets = Tweet.objects.filter(data_source__gt=0)

# Group tweets by 'source' and count totals
source_query = list(tweets.values('source').annotate(total_count=Count('source')).order_by('-total_count'))

# Turn query into dictionary and create dataframe
source_list = [x["source"] for x in source_query]
source_counts = [x["total_count"] for x in source_query]
source_dictionary = {"source" : source_list, "count" : source_counts}
df = pd.DataFrame.from_dict(source_dictionary)
df['proportion_of_total'] = df['count'] / tweets.count()

# Split tweets into Keyword and Geo streams and add similar columns:
geo_tweets = tweets.filter(data_source__gt=2)
geo_source_query = list(geo_tweets.values('source').annotate(total_count=Count('source')).order_by('-total_count'))
geo_dictionary = {k["source"]:k["total_count"] for k in geo_source_query}
geo_count_col = pd.Series([geo_dictionary.get(df["source"][i],0) for i in range(0, len(df))]) 
df.insert(2, "geo_count", geo_count_col) 
df['proportion_of_geo_stream'] = df['geo_count'] / geo_tweets.count()

kw_tweets = tweets.filter(data_source__lte=2)
kw_source_query = list(kw_tweets.values('source').annotate(total_count=Count('source')).order_by('-total_count'))
kw_dictionary = {k["source"]:k["total_count"] for k in kw_source_query}
kw_count_col = pd.Series([kw_dictionary.get(df["source"][i],0) for i in range(0, len(df))]) 
df.insert(2, "kw_count", kw_count_col) 
df['proportion_of_kw_stream'] = df['kw_count'] / kw_tweets.count()

df.head()
```

Note that `proportion_of_geo_stream` represents the proportion of the geo streamed tweets that has come from the given source, not the proportion of the source that is from the geo stream (and similarly for `proportion_of_kw_stream`.

For example, the above table clearly shows that the majority (76.6%) of Tweets returned by the geo stream came from Instagram.


```python
import matplotlib.pyplot as plt

df[:20].plot.bar(x='source', y='proportion_of_total', rot=85)
df.sort_values(by=['proportion_of_kw_stream'], ascending=False)[:20].plot.bar(x='source', y='proportion_of_kw_stream', rot=85)
df.sort_values(by=['proportion_of_geo_stream'], ascending=False)[:20].plot.bar(x='source', y='proportion_of_geo_stream', rot=85)

df[:20].plot.bar(x='source', y=['proportion_of_total', 'proportion_of_kw_stream', 'proportion_of_geo_stream'], rot=85)
```

Instagram makes up a large proportion of the geo-tagged Tweets. This appears to be a built-in function of the Instagram app when an Instagram user has auto-crossposting to Twitter enabled.

We can replot the geo chart excluding the Instagram content:


```python
df.sort_values(by=['proportion_of_geo_stream'], ascending=False)[1:21].plot.bar(x='source', y='proportion_of_geo_stream', rot=85)
```

We can also check what proportion of Instagram content is geo tagged:


```python
#query = Tweet.objects.filter(data_source=1)
#query = Tweet.objects.filter(data_source=1).exclude(source='Instagram')
query = Tweet.objects.filter(data_source=1).filter(source='Instagram')
total = query.count()
no_geo = query.filter(coordinates_type__isnull=True).count()
print("{0} of {1} Instagram posts from kw_stream do not include coordinates ({2:.2f}%).".format(no_geo, total, (no_geo / total * 100)))
```

### Type of Content by Source

We can manually check the collection of each source value for common behaviours (for example, whether a source platforms is automatically generating marketing Tweets).

First, we collect a list of sources which appear in the dataset over 100 times. 

We then take a look at some content from each source for a brief qualitative analysis. We also check whether a source is dominated by a small set of users by checking the proportional distribution by user. This will allow us to check whether certain sources are comprised of (e.g.) automated marketing or spam accounts.


```python
# Sources for qualitative analysis:
sources_for_analysis = df['source'].loc[df['count'] > 100]
sources_for_analysis
#list(sources_for_analysis.values)
```


```python
for source_name in list(df['source'].loc[df['count'] > 100].values):
    print('\n\n--------------------------------------')
    print("Tweets for source: {}".format(source_name))
    print('--------------------------------------\n')
    query = tweets.filter(source=source_name)
    for t in query[:min(query.count(), 10)]:
        print(t.text)
    #text_df = pd.DataFrame(list(tweets.filter(source=source_name).values('text')))
    #text_df[:10]

    author_count = list(query.values('author__screen_name').annotate(total_count=Count('author__screen_name')).order_by('-total_count'))

    # Turn query into dictionary and create dataframe
    source_list = [x["author__screen_name"] for x in author_count]
    source_counts = [x["total_count"] for x in author_count]
    source_dictionary = {"author__screen_name" : source_list, "count" : source_counts}
    df2 = pd.DataFrame.from_dict(source_dictionary)
    df2['total_proportion'] = df2['count'] / query.count()

    print('\n')
    print(df2.head())

```

From the above results, we are primarily interested in categorising a source as automated or otherwise irrelevant to our application.

* Typical Usage:
  * Instagram (cross-posting), Twitter for iPhone, Twitter Web Client, Twitter for Android, TweetDeck, Twitter for iPad, Twitter Lite, Facebook (cross-posting)
* Automated posting from other network (used in a different way to normal usage):
  * Foursquare, Untappd, (Instagram)
* Advertising/Spam:
  * Paper.li
* Automated Job Postings:
  * TweetMyJOBS, SafeTweet by TweetMyJOBS
* Media Outlets / content managers: 
  * SocialNewsDesk, Sprout Social, IFTTT, Hootsuite, Buffer
* Private Network (News Outlet): 
  * BubbleLife
* Private app: 
  * Error-log
  
Of the 'typical usage' sources, it may be worth considering the platform -- ie. 'Instagram', 'Twitter for iPhone' etc are likely to be from mobile devices, whereas 'Twitter Web Client' and 'TweetDeck' are more likely to be generated on computers.


```python
# Plotting distribution of authors
#df2.plot(y='total_proportion', x='author__screen_name', rot=85)
#plt.axis('off')
#plt.show()
```

The proportion of Tweets per-source that contain URLS, media, mentions and hashtags are calculated and added to the dataframe.

Note that the media column is excluded as this wasn't captured for the Harvey dataset.


```python
url_prop = []
hashtag_prop = []
mention_prop = []
media_prop = []

for source_value in sources_for_analysis:
    source_tweets = tweets.filter(source=source_value)
    total = source_tweets.count()
    
    url_count = source_tweets.filter(urls__isnull=False).distinct().count()
    url_prop.append(url_count / total)
    hashtag_count = source_tweets.filter(hashtags__isnull=False).distinct().count()
    hashtag_prop.append(hashtag_count / total)
    mention_count = source_tweets.filter(mentions__isnull=False).distinct().count()
    mention_prop.append(mention_count / total)
    #media_count = source_tweets.filter(media_files__isnull=False).count()
    #media_prop.append(media_count / total)

col_index = len(df.columns)
df.insert(col_index, 'url_prop', pd.Series(url_prop))
df.insert(col_index, 'hashtag_prop', pd.Series(hashtag_prop))
df.insert(col_index, 'mention_prop', pd.Series(mention_prop))
#df.insert(col_index, 'media_prop', pd.Series(media_prop))

df.head()
```


```python
df.sort_values(by=['url_prop'], ascending=False)[0:21].plot.bar(x='source', y='url_prop', rot=85)
df.sort_values(by=['hashtag_prop'], ascending=False)[0:21].plot.bar(x='source', y='hashtag_prop', rot=85)
df.sort_values(by=['mention_prop'], ascending=False)[0:21].plot.bar(x='source', y='mention_prop', rot=85)
#df.sort_values(by=['media_prop'], ascending=False)[0:21].plot.bar(x='source', y='media_prop', rot=85)
```

It is unsurprising to see that the sources identified as spam/automated are more likely to include URLs and Mentions. 

It is interesting to note that there is a significant difference in URL rates between the sources expected to be computer-based vs mobile, which supports the idea that the former are computer users and therefore more likely to be sharing content they are reading, rather than posting live thoughts while 'on-the-go'.

Of the above categories, we can safely disregard all but those classified as 'Typical Usage'. The content generated by users of the Foursquare and Untappd apps does contain some interesting content, but as most interactions are based on the use of the apps, it's generally unrelated to the event.


```python
df.sort_values(by=['url_prop'], ascending=False)[0:21]
```


```python
#eliminated_sources = ['Foursquare', 'Untappd', 'Paper.li', 'TweetMyJOBS', 'SafeTweet by TweetMyJOBS', 'SocialNewsDesk', 'Sprout Social', 'IFTTT', 'BubbleLife', 'Error-log']
#original_tweet_count = tweets.count()

#tweets = Tweet.objects.filter(data_source__gt=0).exclude(source__in=eliminated_sources)
#print('{0} Tweets eliminated from dataset of {1} ({2:.2f}%).'.format(original_tweet_count - tweets.count(), original_tweet_count, ((original_tweet_count - tweets.count()) / original_tweet_count * 100)))
```



# Data Processing
Exploration of data which has been hand-coded.

Tweets and accounts have been coded, with a secondary coder doing a proportion of objects which were coded by the main coder.

Dimensions refer to different coding schema.


```python
import pandas as pd

# Codings for Tweets:
# Note: we exclude data_code__data_code_id=0 as this is a 'to be coded' class, which is an artefact of the software.
tweet_codings = Coding.objects.filter(coding_id=1).filter(tweet__isnull=False).filter(data_code__data_code_id__gt=0)
tweet_codings_secondary = Coding.objects.filter(coding_id=2).filter(tweet__isnull=False)
# Codings for accounts:
account_codings = Coding.objects.filter(coding_id=1).filter(user__isnull=False).filter(data_code__data_code_id__gt=0)
account_codings_secondary = Coding.objects.filter(coding_id=2).filter(user__isnull=False)

# Note these totals combine all user or Tweet codes, so can be misleading if more than one dimension is used.
print("{} Tweets coded by primary coder, {} by secondary coder.".format(tweet_codings.count(), tweet_codings_secondary.count()))
print("{} Accounts coded by primary coder, {} by secondary coder.".format(account_codings.count(), account_codings_secondary.count()))
print('\n')

# Check available coding schema:
dimensions = DataCodeDimension.objects.all()
print('Classes by Dimension:\n')
for d in dimensions:
    print('\'{}\', type: {}\n{}\n'.format(d.name, d.coding_subject, list(d.datacode.values_list('name', flat=True))))
    
```

    2180 Tweets coded by primary coder, 225 by secondary coder.
    931 Accounts coded by primary coder, 151 by secondary coder.
    
    
    Classes by Dimension:
    
    'Information Type', type: tweet
    ['Aid Request', 'Ground Truth', 'Info for Affected', 'Info for Non-Affected', 'Emotion - Affected', 'Emotion - Unaffected', 'Unrelated', 'Unclassified']
    
    'Local', type: user
    ['Unsure', 'Non-Witness', 'Witness']
    


## Evaluation of coder agreement
First, we check the agreement between the two coders to assess the choice and definition of codes.

The primary coder's choices are displayed as rows, the secondary coder's choices are columns.

### Tweet Codes


```python
# Note the following code will need to be updated if there is more than one dimension for 
# Tweet or User coding.

# Create Matrix as dataframe
classes = DataCodeDimension.objects.get(coding_subject='tweet').datacode.values_list('name', flat=True)
class_df = pd.DataFrame(index=classes, columns=classes)
class_df = class_df.fillna(0)

# Get all Tweets that have been coded by both users:
double_coded_tweets = Tweet.objects.filter(coding_for_tweet__coding_id=1).filter(coding_for_tweet__coding_id=2)

for t in double_coded_tweets:
    coding1 = t.coding_for_tweet.filter(coding_id=1)[0]
    coding2 = t.coding_for_tweet.filter(coding_id=2)[0]
    class_df.loc[coding1.data_code.name, coding2.data_code.name] += 1

class_df
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
      <th>Aid Request</th>
      <th>Ground Truth</th>
      <th>Info for Affected</th>
      <th>Info for Non-Affected</th>
      <th>Emotion - Affected</th>
      <th>Emotion - Unaffected</th>
      <th>Unrelated</th>
      <th>Unclassified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aid Request</th>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ground Truth</th>
      <td>0</td>
      <td>33</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Info for Affected</th>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Info for Non-Affected</th>
      <td>0</td>
      <td>7</td>
      <td>7</td>
      <td>33</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Emotion - Affected</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Emotion - Unaffected</th>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>7</td>
      <td>8</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Unrelated</th>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Unclassified</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Calculate disagreement percentages for rows and columns.

The 'error' represents the error of the subject coder if the alternate coder represents the correct class. It is better interpreted as a count of disagreements, per class chosen by the subject coder.


```python
agreement_counts = [row[index] for index, row in class_df.iterrows()]

col_count = class_df.sum(axis=0).subtract(agreement_counts)
col_proportion = col_count.divide(class_df.sum(axis=0)/100).round(1)

row_count = class_df.sum(axis=1).subtract(agreement_counts)
row_proportion = row_count.divide(class_df.sum(axis=1)/100).round(1)

prop_df = pd.concat([row_count, row_proportion, col_count, col_proportion], axis=1)
prop_df.columns = ['Primary\'s Errors', '%', 'Second\'s Errors', '%']
print(prop_df)
```

                           Primary's Errors     %  Second's Errors     %
    Aid Request                           1  16.7                1  16.7
    Ground Truth                          3   8.3               18  35.3
    Info for Affected                     0   0.0               12  46.2
    Info for Non-Affected                17  34.0               10  23.3
    Emotion - Affected                    7  53.8               12  66.7
    Emotion - Unaffected                 24  37.5                7  14.9
    Unrelated                            11  26.2                3   8.8
    Unclassified                          0   NaN                0   NaN


The categories for which there is a large imbalance in the error rate are interesting as they may represent categories for which each coder has a fundamentally different interpretation. For example, the 'Info for Affected' class: every Tweet coded in this class by the primary coder was also coded as such by the secondary, however the secondary also categorised a further 12 Tweets in this class that the primary did not. This suggests the primary coder had a much more narrow definition of this class.

These values must be considered with respect to their proportion of the total set. For example, there may be a high error rate in a code that appears rarely in the dataset while the majority of the data falls in one class with a high agreement rating. 

We may also be interested in weighting particular classes when calculating accuracy, if they are more important than others to correctly classify.

##### Cohen's Kappa
Instead of using a basic accuracy measurement, we calculate the Cohen's Kappa, which accounts for the probability of the users agreeing on a code by chance.


```python
# Cohen's Kappa
def calc_cohen(df):
    total = class_df.sum().sum()

    pr_list = class_df.sum(axis=1).divide(total)
    sec_list = class_df.sum(axis=0).divide(total)
    pe = sum(pr_list*sec_list)
    po = sum(agreement_counts)/total

    kappa = (po - pe) / (1 - pe)

    return kappa

calc_cohen(class_df)
```




    0.659000697635257



The Kappa is .66

A score between .61-80 suggests 'substantial agreement' (Landis, J.R.; Koch, G.G. (1977). "The measurement of observer agreement for categorical data").

However, as number of classes increases, the Kappa should also naturally increase (as pe will naturally become smaller).

### Account Codes


```python
# Note the following code will need to be updated if there is more than one dimension for 
# Tweet or User coding.

# Create Matrix as dataframe
classes = DataCodeDimension.objects.get(coding_subject='user') \
            .datacode.values_list('name', flat=True)
class_df = pd.DataFrame(index=classes, columns=classes)
class_df = class_df.fillna(0)

# Get all Users that have been coded by both users:
double_coded_users = User.objects.filter(coding_for_user__coding_id=1).filter(coding_for_user__coding_id=2)

for u in double_coded_users:
    coding1 = u.coding_for_user.filter(coding_id=1)[0]
    coding2 = u.coding_for_user.filter(coding_id=2)[0]
    class_df.loc[coding1.data_code.name, coding2.data_code.name] += 1

class_df
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
      <th>Unsure</th>
      <th>Non-Witness</th>
      <th>Witness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unsure</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Non-Witness</th>
      <td>5</td>
      <td>89</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Witness</th>
      <td>0</td>
      <td>5</td>
      <td>41</td>
    </tr>
  </tbody>
</table>
</div>




```python
agreement_counts = [row[index] for index, row in class_df.iterrows()]

col_count = class_df.sum(axis=0).subtract(agreement_counts)
col_proportion = col_count.divide(class_df.sum(axis=0)/100).round(1)

row_count = class_df.sum(axis=1).subtract(agreement_counts)
row_proportion = row_count.divide(class_df.sum(axis=1)/100).round(1)

prop_df = pd.concat([row_count, row_proportion, col_count, col_proportion], axis=1)
prop_df.columns = ['Primary\'s Errors', '%', 'Second\'s Errors', '%']
print(prop_df)
```

                 Primary's Errors      %  Second's Errors      %
    Unsure                      3  100.0                5  100.0
    Non-Witness                13   12.7                5    5.3
    Witness                     5   10.9               11   21.2


##### Cohen's Kappa


```python
calc_cohen(class_df)
```




    0.706551915602443



As both Kappas indicate 'substantial agreement', we may proceed with using them in supervised learning methods. 

## Preparing Data Set

We will create a dataset each for the coded Tweets and user accounts.

### Tweet Dataframe

We start by adding all coded Tweets to a dataframe, then checking column names for those which should be encoded or dropped.


```python
import pandas as pd
```


```python
# Get all Tweets coded by primary coder:
tweets = Tweet.objects.filter(coding_for_tweet__coding_id=1, coding_for_tweet__data_code__data_code_id__gt=0)
tweet_df = pd.DataFrame(list(tweets.values()))

tweet_df.head()
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
      <th>author_id</th>
      <th>coordinates_lat</th>
      <th>coordinates_lon</th>
      <th>coordinates_type</th>
      <th>created_at</th>
      <th>data_source</th>
      <th>favorite_count</th>
      <th>id</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_user_id</th>
      <th>...</th>
      <th>media_files</th>
      <th>media_files_type</th>
      <th>place_id</th>
      <th>quoted_status_id</th>
      <th>quoted_status_id_int</th>
      <th>replied_to_status_id</th>
      <th>retweet_count</th>
      <th>source</th>
      <th>text</th>
      <th>tweet_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1338848</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>2017-08-26 12:57:50+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>4852</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>0</td>
      <td>Twitter for iPhone</td>
      <td>Great reporting on #Harvey @FOXNews @SHarrigan...</td>
      <td>901443573475946496</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5827896</td>
      <td>29.7629</td>
      <td>-95.3832</td>
      <td>Point</td>
      <td>2017-08-31 16:49:46+00:00</td>
      <td>3</td>
      <td>0</td>
      <td>45580</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>0</td>
      <td>Instagram</td>
      <td>I distinctly remember packing for Houston in a...</td>
      <td>903313880188948481</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8965229</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>2017-08-29 23:55:27+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>32393</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>0</td>
      <td>Twitter Lite</td>
      <td>starting the school year off with a flooded ho...</td>
      <td>902696231780196352</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7507026</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>2017-08-29 14:02:27+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>26193</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>0</td>
      <td>Pardot</td>
      <td>As #HurricaneHarvey continues, we hope these t...</td>
      <td>902546996799520768</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3432737</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>2017-08-31 01:05:54+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>41262</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>0</td>
      <td>Twitter for iPhone</td>
      <td>Please help my Aunt Grace get back on her feet...</td>
      <td>903076349149106176</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
list(tweet_df.columns)
```




    ['author_id',
     'coordinates_lat',
     'coordinates_lon',
     'coordinates_type',
     'created_at',
     'data_source',
     'favorite_count',
     'id',
     'in_reply_to_status_id',
     'in_reply_to_user_id',
     'is_deleted',
     'is_deleted_observed',
     'lang',
     'media_files',
     'media_files_type',
     'place_id',
     'quoted_status_id',
     'quoted_status_id_int',
     'replied_to_status_id',
     'retweet_count',
     'source',
     'text',
     'tweet_id']



Certain columns are converted into a numeric or binary representation and others are removed as irrelevant to classification.

Language is encoded as binary column for english / non-english. English represents over 90% of the data.

One-hot encoding uses values 0 and 1 instead of True and False to enable ML integration.


```python
# Change author_id, which is the SQL user id, to the Twitter user id of the author:
tweet_df['author_id'] = tweet_df['author_id'].apply(lambda x: User.objects.get(id=x).user_id)


# 'Has Coordinates' column:
d = {'Point': 1, None: 0}
tweet_df['has_coords'] = tweet_df['coordinates_type'].replace(d)
# Alternate methods:
#x = tweet_df['coordinates_type'].map(d)
#tweet_df['coordinates_type'].replace(d, inplace=True)

# 'Is a Reply' column:
tweet_df['is_reply'] = 0
tweet_df.loc[tweet_df['in_reply_to_user_id'].isnull() == False, 'is_reply'] = 1

# 'Is Quoting' column:
tweet_df['is_quoting'] = 0
tweet_df.loc[tweet_df['quoted_status_id_int'].isnull() == False, 'is_quoting'] = 1

#tweet_df[tweet_df.quoted_status_id_int.notnull()].head()

```


```python
# Encode languages as integers

# Encode each language as a seperate integer (not appropriate for ML):
#langs = tweet_df.lang.unique()
#lang_encoding = list(range(len(langs)))
#lang_encoding_dict = dict(zip(langs, lang_encoding))
#tweet_df['lang'].replace(lang_encoding_dict, inplace=True)
#lang_encoding_dict

# Single language encoding:
tweet_df['lang_en'] = 0
tweet_df.loc[tweet_df['lang'] == 'en', 'lang_en'] = 1

# Check proportion of each language in dataset to justify choice of only english:
tweet_df['lang'].value_counts()/tweets.count()
```




    en     0.906422
    und    0.047706
    es     0.025229
    fr     0.005505
    pt     0.002752
    ar     0.001835
    de     0.001835
    da     0.001376
    nl     0.001376
    ja     0.000917
    it     0.000917
    ht     0.000459
    et     0.000459
    pl     0.000459
    cy     0.000459
    fi     0.000459
    tl     0.000459
    ru     0.000459
    vi     0.000459
    hi     0.000459
    Name: lang, dtype: float64



Encode sources as one-hot columns for those that made up at least 1% of the overall dataset identified in the exploratory analysis:

Note: TweetMyJOBS and SafeTweet by TweetMyJOBS are combined into one column.


```python
source_list = ['Instagram', 'Twitter for iPhone', 'Twitter Web Client', 'Twitter for Android', 'Paper.li', 
 'Hootsuite', 'TweetMyJOBS', 'SafeTweet by TweetMyJOBS', 'IFTTT', 'Facebook', 'TweetDeck', 
 'Twitter for iPad', 'BubbleLife', 'Twitter Lite']

tweet_df['source_other'] = 1
for s in source_list:
    col_name = 'source_' + s.replace(" ", "")
    tweet_df[col_name] = 0
    tweet_df.loc[tweet_df['source'] == s, col_name] = 1
    tweet_df.loc[tweet_df['source'] == s, 'source_other'] = 0
    
# Merge columns:
tweet_df.loc[tweet_df['source_SafeTweetbyTweetMyJOBS'] == 1, 'source_TweetMyJOBS'] = 1
tweet_df.drop(columns=['source_SafeTweetbyTweetMyJOBS'], inplace=True)
    
#tweet_df.head()
```


```python
# Drop unecessary columns:
drop_columns = ['coordinates_lat', 'coordinates_lon', 'coordinates_type', 
                'favorite_count', 'id', 'in_reply_to_status_id', 'in_reply_to_user_id', 
                'is_deleted_observed', 'media_files', 'media_files_type', 'place_id', 
                'quoted_status_id', 'quoted_status_id_int', 'replied_to_status_id', 
                'retweet_count', 'is_deleted', 'source', 'lang']

tweet_df.drop(columns=drop_columns, inplace=True)

tweet_df.head()
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
      <th>author_id</th>
      <th>created_at</th>
      <th>data_source</th>
      <th>text</th>
      <th>tweet_id</th>
      <th>has_coords</th>
      <th>is_reply</th>
      <th>is_quoting</th>
      <th>lang_en</th>
      <th>source_other</th>
      <th>...</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>877507343826604034</td>
      <td>2017-08-26 12:57:50+00:00</td>
      <td>1</td>
      <td>Great reporting on #Harvey @FOXNews @SHarrigan...</td>
      <td>901443573475946496</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>126695871</td>
      <td>2017-08-31 16:49:46+00:00</td>
      <td>3</td>
      <td>I distinctly remember packing for Houston in a...</td>
      <td>903313880188948481</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3872916612</td>
      <td>2017-08-29 23:55:27+00:00</td>
      <td>1</td>
      <td>starting the school year off with a flooded ho...</td>
      <td>902696231780196352</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>289365657</td>
      <td>2017-08-29 14:02:27+00:00</td>
      <td>1</td>
      <td>As #HurricaneHarvey continues, we hope these t...</td>
      <td>902546996799520768</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1228146372</td>
      <td>2017-08-31 01:05:54+00:00</td>
      <td>1</td>
      <td>Please help my Aunt Grace get back on her feet...</td>
      <td>903076349149106176</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



Enrich the data by adding columns from linked data:


```python
def get_hashtag_count(tweet_id):
    return tweets.get(tweet_id=tweet_id).hashtags.count()
def get_url_count(tweet_id):
    return tweets.get(tweet_id=tweet_id).urls.count()
def get_mention_count(tweet_id):
    return tweets.get(tweet_id=tweet_id).mentions.count()

tweet_df['hashtag_count'] = tweet_df['tweet_id'].map(get_hashtag_count)
tweet_df['url_count'] = tweet_df['tweet_id'].map(get_url_count)
tweet_df['mention_count'] = tweet_df['tweet_id'].map(get_mention_count)

#list(tweet_df.columns)
```

We then add the code, which is the target outcome, as an integer.


```python
def get_tweet_code(tweet_id):
    return Coding.objects.filter(coding_id=1).get(tweet__tweet_id=tweet_id) \
        .data_code.data_code_id

tweet_df['data_code_id'] = tweet_df['tweet_id'].map(get_tweet_code)

tweet_df.head()
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
      <th>author_id</th>
      <th>created_at</th>
      <th>data_source</th>
      <th>text</th>
      <th>tweet_id</th>
      <th>has_coords</th>
      <th>is_reply</th>
      <th>is_quoting</th>
      <th>lang_en</th>
      <th>source_other</th>
      <th>...</th>
      <th>source_IFTTT</th>
      <th>source_Facebook</th>
      <th>source_TweetDeck</th>
      <th>source_TwitterforiPad</th>
      <th>source_BubbleLife</th>
      <th>source_TwitterLite</th>
      <th>hashtag_count</th>
      <th>url_count</th>
      <th>mention_count</th>
      <th>data_code_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>877507343826604034</td>
      <td>2017-08-26 12:57:50+00:00</td>
      <td>1</td>
      <td>Great reporting on #Harvey @FOXNews @SHarrigan...</td>
      <td>901443573475946496</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>126695871</td>
      <td>2017-08-31 16:49:46+00:00</td>
      <td>3</td>
      <td>I distinctly remember packing for Houston in a...</td>
      <td>903313880188948481</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3872916612</td>
      <td>2017-08-29 23:55:27+00:00</td>
      <td>1</td>
      <td>starting the school year off with a flooded ho...</td>
      <td>902696231780196352</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>289365657</td>
      <td>2017-08-29 14:02:27+00:00</td>
      <td>1</td>
      <td>As #HurricaneHarvey continues, we hope these t...</td>
      <td>902546996799520768</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1228146372</td>
      <td>2017-08-31 01:05:54+00:00</td>
      <td>1</td>
      <td>Please help my Aunt Grace get back on her feet...</td>
      <td>903076349149106176</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
#tweet_df.to_csv(r'tweet_df.csv', index = None, header=True);
```

The tweet_df is now ready to be used in classification algorithms, however we can also add data from the Tweets' authors.


```python
# Get all authors from tweet_df:
authors = User.objects.filter(tweet__in=tweets).distinct()
author_df = pd.DataFrame(list(authors.values()))

# Check available fields:
print(list(author_df.columns))

author_df.head()
```

    ['added_at', 'betweenness_centrality', 'closeness_centrality', 'created_at', 'data_source', 'default_profile', 'default_profile_image', 'degree_centrality', 'description', 'eigenvector_centrality', 'favourites_count', 'followers_count', 'friends_count', 'geo_enabled', 'has_extended_profile', 'id', 'in_degree', 'is_deleted', 'is_deleted_observed', 'is_translation_enabled', 'katz_centrality', 'lang', 'listed_count', 'load_centrality', 'location', 'name', 'needs_phone_verification', 'old_screen_name', 'out_degree', 'protected', 'ratio_detected', 'ratio_media', 'ratio_original', 'screen_name', 'statuses_count', 'suspended', 'time_zone', 'translator_type', 'tweets_per_hour', 'undirected_eigenvector_centrality', 'url', 'user_class', 'user_followers', 'user_followers_update', 'user_following', 'user_following_update', 'user_id', 'user_network_update_observed_at', 'utc_offset', 'verified']





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
      <th>url</th>
      <th>user_class</th>
      <th>user_followers</th>
      <th>user_followers_update</th>
      <th>user_following</th>
      <th>user_following_update</th>
      <th>user_id</th>
      <th>user_network_update_observed_at</th>
      <th>utc_offset</th>
      <th>verified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-08-26 00:37:31.831480+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012-02-18 00:37:53+00:00</td>
      <td>3</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td></td>
      <td>NaN</td>
      <td>...</td>
      <td>None</td>
      <td>2</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>495469195</td>
      <td>None</td>
      <td>None</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-08-27 20:58:57.505146+00:00</td>
      <td>4.006556e-08</td>
      <td>0.131911</td>
      <td>2011-03-03 07:24:09+00:00</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>0.000182</td>
      <td>Ginger guy in Houston, Texas. I might talk pol...</td>
      <td>3.298253e-07</td>
      <td>...</td>
      <td>https://t.co/Uzn7bzMxoE</td>
      <td>2</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>260126165</td>
      <td>None</td>
      <td>None</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-08-31 15:53:00.939210+00:00</td>
      <td>4.457059e-04</td>
      <td>0.185294</td>
      <td>2010-06-16 15:40:52+00:00</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>0.001762</td>
      <td>WELCOME to Student Housing and Residential Lif...</td>
      <td>2.660015e-04</td>
      <td>...</td>
      <td>http://t.co/Pg6DkCwrZA</td>
      <td>2</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>156321797</td>
      <td>None</td>
      <td>-18000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-08-26 16:56:17.089665+00:00</td>
      <td>2.123291e-04</td>
      <td>0.185398</td>
      <td>2009-04-08 22:15:18+00:00</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>0.003037</td>
      <td>-- Extreme Storm Chaser "Michael Koch" -- @sky...</td>
      <td>3.014993e-02</td>
      <td>...</td>
      <td>https://t.co/tJTP1PVqi6</td>
      <td>2</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>29851333</td>
      <td>None</td>
      <td>-14400</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-08-28 23:17:23.168775+00:00</td>
      <td>1.840374e-04</td>
      <td>0.117098</td>
      <td>2012-02-16 17:57:20+00:00</td>
      <td>3</td>
      <td>True</td>
      <td>False</td>
      <td>0.000607</td>
      <td>Toystore with a lot more!  Pop Culture fanatic...</td>
      <td>7.147106e-08</td>
      <td>...</td>
      <td>http://t.co/kw5DR7prpB</td>
      <td>2</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>494251206</td>
      <td>None</td>
      <td>None</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>




```python
#### Fields from User object:
# 'user_id',
# 'favourites_count', 'followers_count', 'friends_count', 'statuses_count',
# 'listed_count',
# 'default_profile',
# 'default_profile_image',
# 'has_extended_profile',
# 'geo_enabled',
# 'verified',
#### Fields added/calculated during data collection:
# 'user_class',
# 'data_source',
# 'in_degree','out_degree',
# 'betweenness_centrality', 'closeness_centrality','degree_centrality', 
# 'eigenvector_centrality', 'katz_centrality', 'load_centrality'
# 'undirected_eigenvector_centrality'
# 'ratio_detected','ratio_media','ratio_original' 'tweets_per_hour', 

#### Columns to encode:
# 'lang',
# 'time_zone',    # TODO
# 'utc_offset',   # TODO
# 'url', 
# 'created_at',   # TODO
# 'location',

# Columns not relevant to classification:
drop_columns =  ['added_at', 'id', 'old_screen_name', 'is_deleted_observed', 
                 'is_deleted', 'user_followers', 'user_followers_update', 
                 'user_following', 'user_following_update', 
                 'user_network_update_observed_at', 'needs_phone_verification',
                 'suspended', 'protected', 'translator_type', 'is_translation_enabled',
                 'description', 'name', 'screen_name']
author_df.drop(columns=drop_columns, inplace=True)

# Rename conflict with Tweet dataframe column name:
author_df.rename(columns={'data_source':'user_data_source'}, inplace=True)
```


```python
# 'has_url' column:
author_df['profile_has_url'] = 0
author_df.loc[author_df['url'].isnull() == False, 'profile_has_url'] = 1

# Profile has a local location listed, as matched with the list:
local_locations = ['tx', 'houston', 'texas']
author_df['profile_has_local_location'] = 0
author_df.loc[author_df['location'].str.contains(
    '|'.join(local_locations), case=False), 'profile_has_local_location'] = 1

# Convert True/False to 1/0
d = {True: 1, False: 0}
author_df['default_profile'] = author_df['default_profile'].replace(d)
author_df['default_profile_image'] = author_df['default_profile_image'].replace(d)
author_df['geo_enabled'] = author_df['geo_enabled'].replace(d)
author_df['has_extended_profile'] = author_df['has_extended_profile'].replace(d)
author_df['verified'] = author_df['verified'].replace(d)


# Encode languages as boolean: is 'en':
author_df['user_lang_en'] = 0
author_df.loc[author_df['lang'] == 'en', 'user_lang_en'] = 1
# Check proportion of each language in dataset (to justify choice of only encoding 'en'):
author_df['lang'].value_counts()/authors.count()

# TODO: categorise and encode the following three
# 'utc_offset', 
# 'time_zone',
# 'created_at',
##author_df['time_zone'].value_counts()
```




    en       0.9460
    es       0.0215
    fr       0.0065
    de       0.0045
    pt       0.0035
    it       0.0035
    ar       0.0030
    en-gb    0.0030
    nl       0.0015
    fi       0.0015
    tr       0.0010
    sv       0.0010
    xx-lc    0.0005
    el       0.0005
    en-GB    0.0005
    ca       0.0005
    sk       0.0005
    ru       0.0005
    ja       0.0005
    Name: lang, dtype: float64




```python
drop_columns = ['lang', 'url', 'location', 'utc_offset', 'time_zone', 'created_at']
author_df.drop(columns=drop_columns, inplace=True)
```


```python
# Create new dataframe with authors at correct locations (and duplicated where necessary) 
## Replace with pd.dataframe.merge
row_list = []
for x in tweet_df['author_id']:
    row_list.append(author_df.loc[author_df['user_id'] == x].iloc[0])
author_arranged_df = pd.DataFrame(row_list)
author_arranged_df.reset_index(inplace=True, drop=True)

# Combine dataframes
if author_arranged_df.shape[0] == tweet_df.shape[0]:
    tweet_df = pd.concat([tweet_df, author_arranged_df], axis=1)
else:
    print('Error joining frames.')
```


```python
tweet_df.drop(columns=['text'], inplace=True) # Dropping to avoid unsanitised data breaking csv.

tweet_df.to_csv(r'data/harvey_tweet_df.csv', index = None, header = True);
```

### User Dataframe


```python
# Get all users coded by primary coder:
#users = User.objects.filter(coding_for_user__coding_id=1, coding_for_user__data_code__data_code_id__gt=0)
#user_df = pd.DataFrame(list(users.values()))
```
