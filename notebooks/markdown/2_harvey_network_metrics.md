
# User Location Classification in Hurricane Harvey
This is the second notebook in a series which are written primarily as a research logbook for the author. They are therefore not to be considered complete and do not represent the final analysis. For this -- see the final published papers and thesis, or contact the author directly.
The goal of this analysis is to evaluate methods by which users Tweeting about Hurricane Harvey may be classified as in the area or otherwise.

Data was collected with custom software which observed several Twitter streams and enhanced this information by querying the Twitter REST APIs for the network data (friends and followers) of each author. Stream volume which exceeded the capacity of the REST requests was discarded. 
* The keyword stream monitored the terms: [#harvey, #harveystorm, #hurricaneharvey, #corpuschristi]
* The GPS stream used the bounding box: [-99.9590682, 26.5486063, -93.9790001, 30.3893434]
* The collection period ran from 2017-08-26 01:32:18 until 2017-09-02 10:30:52 
* 55,605 Tweets by 33,585 unique authors were recorded

Data was coded using an interface built into the collection software by a primary coder. A secondary coder coded a sub-set of coded users for validation of the coding schema. User instances were coded by whether they 'appeared to be in the affected area'.

These notebooks access the data directly from the database using standard Django query syntax.


# Testing Community Detection Data

This section investigates the value of social network data in classifying users by the target class (witness/non-witness). The hypothesis, loosely, is that local (or witness) users are more likely to follow one another, and therefore the reflection of this behaviour within the network structure provides a metric which can be implemented within classification models. Collecting this network data is significantly more difficult than the other User and Tweet features, therefore the associative qualities are rarely tested.

The presence of this structure within the Hurricane Harvey dataset is visually established. The graph is then partitioned using a suite of community detection algorithms which provide metrics which are compatible with standard classification algorithms. These metrics are then tested for their correlation with the target feature to validate the use of community features in modelling, and to select the most appropriate algorithm. Finally, the algorithms are tested over subgraphs of the final dataset which represent the dataset at various points during the collection process.

<img src="./data/harvey_user_location/img/harvey-network-structure.png" alt="network-structure" style="width: 600px;"/>

A visual inspection of the follower/friend network of detected users (pictured above) shows a clear community structure. This suggests that there are certain user features that influence the propensity for a user to follow another user with similar features. These may be features which we have observed and recorded, or other features which we cannot predict. This can be tested by using community detection algorithms to partition the nodes of the graph into these communities, and then comparing these partitions to features with which they may be associated. Primarily, we are interested in whether these communities are related to whether a user is a 'witness' or not, as per our manual coding. If this is the case, the community metrics can be a useful feature in our models.

The image below shows a representation of the same network structure, where only the 1500 coded nodes are displayed in colours which represent their codes. The general grouping of orange nodes towards one section of the graph which corresponds with a community cluster shown in the original graph structure suggests that this community contains a greater proportion of witness nodes and therefore is a dependent feature. A preliminary hypothesis is that local users are more likely to follow one another (thus forming this community) and are also more likely to be witnesses. If true, this community could accurately predict which users are local and therefore, more likely to be witnesses.

<img src="./data/harvey_user_location/img/harvey-network-structure-coded.png" alt="network-structure-coded" style="width: 600px;"/>

The following image shows the output of a community detection algorithm which has partitioned the graph. In this example, the pink community appears to represent the witness group shown in the previous image.

<img src="./data/harvey_user_location/img/harvey-network-structure-community.png" alt="network-structure-community" style="width: 600px;"/>

## Calculating Community Labels
In the following section, the original dataframe is enhanced with modularity metrics. These are features which are calculated based upon the graph structure of the user friend/follower network. There are a number of community detection algorithms, so a set of these have been calculated to be tested and compared for association to the target (witness) class.

Community metrics were calculated using the `networkx` implementations of `greedy_modularity_communities`, `label_propagation_communities`, `asyn_lpa_communities`, `asyn_fluidc` and `community_louvain` from the `community` package. These were chosen due to their common use for these applications as well as the availability of their implmentations within the package.

The graph includes all detected users (i.e. not their followers/friends unless they were also detected as authors) for a total of 31,496 nodes and 99,602 edges. Community detection was performed on subgraph representing the largest connected component of 17,958 nodes and 75,203 edges. 

As most algorithms require an undirected graph, the direction of relationships was ignored.

Communities are labelled as numbers according to their ranking in size (where 0 is the largest), thus the labels have some level of ordinality.


```python
### Initialisation ###
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = [6, 4]

# Location of data files
DIR = './data/harvey_user_location/'
GRAPH_DIR = DIR + 'graph_objs/'

EVENT_NAME = Event.objects.all()[0].name.replace(' ', '')
DF_FILENAME = 'df_users.csv'

# Confirm correct database is set in Django settings.py
if 'Harvey' not in EVENT_NAME:
    raise Exception('Event name mismatch -- check database set in Django')
```


```python
def get_graph_object():
    ''' 
    Gets the graph object for the current event.
    Imports gexf file if extant, otherwise builds from database
    and saves as gexf file.
    '''
    FILENAME = 'network_data_{}_v1.gexf'.format(EVENT_NAME)
    try:
        # Load cached file if available
        G = nx.read_gexf(GRAPH_DIR + FILENAME)
        #print('Importing existing graph object...')
    except:
        print('Creating new graph object from database...')
        classed_users = User.objects.filter(user_class__gte=1)
        edges = Relo.objects.filter(target_user__in=classed_users, source_user__in=classed_users, end_observed_at=None)
        G=nx.DiGraph()
        for node in classed_users:
            try:
                user_code = node.coding_for_user.filter(coding_id=1).exclude(data_code__name='To Be Coded')[0].data_code.name
            except:
                user_code = ''
            G.add_node(node.screen_name, user_class=node.user_class, user_code=user_code)
        edge_list = [(edge.source_user.screen_name, edge.target_user.screen_name) for edge in edges]
        G.add_edges_from(edge_list)
        # Write to file and re-import to bypass unresolved issue with community algorithms
        nx.write_gexf(G, GRAPH_DIR + FILENAME, prettyprint=True)
        G = nx.read_gexf(GRAPH_DIR + FILENAME)
    return G
```


```python
# Modularity:   https://scholar.google.com/scholar?q=Finding+community+structure+in+very+large+networks
# Label Prop:   https://neo4j.com/docs/graph-algorithms/current/algorithms/label-propagation/#algorithms-label-propagation-sample
# Louvain:      https://github.com/taynaud/python-louvain/blob/master/docs/index.rst

#import networkx as nx
from networkx.algorithms.community import (greedy_modularity_communities, 
                                            label_propagation_communities, 
                                            asyn_lpa_communities, 
                                            asyn_fluidc, girvan_newman)
import community as community_louvain
from collections import Counter

def calc_community_metrics(G, filename=False):
    ''' 
    Returns a graph object enhanced with various community 
    metrics added as node attributes.
    If a filename is provided, imports a cached gexf file 
    with community values if extant, otherwise calculates 
    them and and saves as gexf file.
    '''
    
    FLIUD_COMMUNITIES_NUMBER = 8
    RETURN_GIANT_COMPONENT = True # Only return giant component as graph object
    
    # Load cached file if available
    if filename:
        try:
            G = nx.read_gexf(GRAPH_DIR + filename)
            #print('Importing existing community graph object...')
            return G
        except:
            pass
    
    print('Calculating community metrics for graph. {} nodes and {} edges...'
              .format(len(G), G.number_of_edges()))

    # Create undirected graph (required for community detection):
    H = nx.Graph(G)

    # Get largest component
    Hcc = max(nx.connected_components(H), key=len)
    H0 = H.subgraph(Hcc)
    #H0 = nx.connected_component_subgraphs(H)[0]
    print('Largest component has {} nodes and {} edges.'
              .format(len(H0), H0.number_of_edges()))

    # Discard other components:
    if RETURN_GIANT_COMPONENT:
        G = G.subgraph(Hcc)

    # Get communities
    print('Calculating c_modularity...')
    c_modularity = list(greedy_modularity_communities(H0))
    print('Calculating c_label_prop...')
    c_label_prop = list(label_propagation_communities(H0))
    c_label_prop = sorted(c_label_prop, key=len, reverse=True)
    print('Calculating c_label_prop_asyn...')
    c_label_prop_asyn = list(asyn_lpa_communities(H0))
    c_label_prop_asyn = sorted(c_label_prop_asyn, key=len, reverse=True)
    print('Calculating c_fluid...')
    c_fluid = list(asyn_fluidc(H0, FLIUD_COMMUNITIES_NUMBER))
    c_fluid = sorted(c_fluid, key=len, reverse=True)
    # TOO SLOW:
    # print('Calculating c_girvan_newman...')
    # c_girvan_newman = list(girvan_newman(H0))
    print('Calculating c_louvain...')
    partition = community_louvain.best_partition(H0)
    # Rank indices by size
    counter = Counter(partition.values())
    ranking = sorted(counter, key=counter.get, reverse=True)
    partition_sorted = {k: ranking.index(v) for k, v in partition.items()}

    print('Adding data as node attributes...')
    # Add communities to node attributes:
    community_output_dict = {'c_modularity': c_modularity,
                            'c_label_prop': c_label_prop,
                            'c_label_prop_asyn': c_label_prop_asyn,
                            'c_fluid': c_fluid,
                            # 'c_girvan_newman': c_girvan_newman
                            }
    for key in community_output_dict:
        community_dict = {}
        for i, c in enumerate(community_output_dict[key]):
            for name in c:
                community_dict[name] = i
        nx.set_node_attributes(G, community_dict, key)
    # (Louvain package returns a different format):
    nx.set_node_attributes(G, partition_sorted, 'c_louvain')

    if filename:
        print('Writing to file...')
        nx.write_gexf(G, GRAPH_DIR + filename, prettyprint=True)
        
    return G
```


```python
e = Event.objects.all()[0]
filename = 'network_data_{}_comm.gexf'.format(e.name.replace(' ', ''))

G = get_graph_object()
G = calc_community_metrics(G, filename)

# Create datagrame from graph node attributes
nodes = G.nodes(data=True)
df_comm = pd.DataFrame.from_dict(dict(nodes), orient='index')
df_comm.drop(['user_class', 'user_code', 'label'], axis=1, inplace=True)
#df_comm = df_comm.reset_index(drop=True)
df_comm.head()
```

    Creating new graph object from database...
    Calculating community metrics for graph. 31931 nodes and 101096 edges...
    Largest component has 18409 nodes and 76341 edges.
    Calculating c_modularity...
    Calculating c_label_prop...
    Calculating c_label_prop_asyn...
    Calculating c_fluid...
    Calculating c_louvain...
    Adding data as node attributes...
    Writing to file...





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>c_modularity</th>
      <th>c_label_prop</th>
      <th>c_label_prop_asyn</th>
      <th>c_fluid</th>
      <th>c_louvain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0044Tamil</th>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>007rogerbmoore</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>01fmoreira</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0KTOBR</th>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0x41_0x48</th>
      <td>241</td>
      <td>885</td>
      <td>816</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Open original Dataframe
users_df = pd.read_csv(DIR + DF_FILENAME, index_col=0)
users_df.shape
```




    (1500, 45)




```python
# Create list of community algorithm column names
comm_cols = list(df_comm.columns)

# Merge dataframes
users_df = pd.merge(left=users_df, right=df_comm, how='left', left_on='screen_name', right_index=True)

users_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>added_at</th>
      <th>betweenness_centrality</th>
      <th>closeness_centrality</th>
      <th>created_at</th>
      <th>default_profile</th>
      <th>default_profile_image</th>
      <th>degree_centrality</th>
      <th>description</th>
      <th>eigenvector_centrality</th>
      <th>favourites_count</th>
      <th>...</th>
      <th>has_url</th>
      <th>changed_screen_name</th>
      <th>account_age</th>
      <th>day_of_detection</th>
      <th>is_data_source_3</th>
      <th>c_modularity</th>
      <th>c_label_prop</th>
      <th>c_label_prop_asyn</th>
      <th>c_fluid</th>
      <th>c_louvain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-08-28 20:42:59.273657+00:00</td>
      <td>0.000043</td>
      <td>0.135798</td>
      <td>2013-03-01 19:23:11+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>0.000304</td>
      <td>If You Want To Live A Happy Life ❇ change your...</td>
      <td>3.905631e-07</td>
      <td>2030</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1645</td>
      <td>3</td>
      <td>0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-08-30 13:58:20.296918+00:00</td>
      <td>0.000015</td>
      <td>0.122066</td>
      <td>2014-01-20 00:34:57+00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0.000243</td>
      <td>Employee Giving PM @Microsoft.A daydreamer w/ ...</td>
      <td>1.785776e-07</td>
      <td>1015</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1321</td>
      <td>5</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1368.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-08-26 19:51:45.107222+00:00</td>
      <td>0.000000</td>
      <td>0.077120</td>
      <td>2012-07-24 13:47:47+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>0.000061</td>
      <td>Making an impact isn’t something reserved for ...</td>
      <td>8.518251e-14</td>
      <td>12</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1865</td>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>697.0</td>
      <td>1054.0</td>
      <td>3.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-08-26 11:13:05.769123+00:00</td>
      <td>0.000383</td>
      <td>0.167070</td>
      <td>2010-12-16 17:30:04+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>0.000668</td>
      <td>Eyeing global entropy through a timeline windo...</td>
      <td>4.315565e-05</td>
      <td>347</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>2451</td>
      <td>1</td>
      <td>0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-08-26 14:19:23.604361+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009-04-24 12:08:14+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>Producer. Show Control Designer. Project Coord...</td>
      <td>NaN</td>
      <td>25</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3052</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>



### Giant Component Comparison
Community detection is applied only to the largest component within the graph. This is a limitation of some of the algorithms and is used for all cases for consistency. Coded users which are not part of this largest sub-graph are not assigned a community label. The subgraphs which are not the largest component may isolated nodes or graphs too small to be worth evaulating, or there could be a single second subgraph containing 49% of the nodes. In the latter case, we would need to consider evaluating the strucutre of this graph too. We can check the proportion of nodes within the giant component, the size of the secondary components, and whether non-witness nodes are more likely to be unconnected to the giant component:


```python
u_label =  sum(users_df[comm_cols[0]].value_counts())
#u_nolabel =  sum(users_df[comm_cols[0]].isna())
print('{} of {} ({:.1f}%) coded users are part of the giant component and therefore have community labels.'
          .format(u_label, users_df.shape[0], u_label/users_df.shape[0]*100))

u_label_pos = sum(users_df.loc[users_df['coded_as_witness'] == 1][comm_cols[0]].value_counts())
u_total_pos = users_df.loc[users_df['coded_as_witness'] == 1].shape[0]
print('{} of {} ({:.1f}%) positively coded users are part of the giant component.'
          .format(u_label_pos, u_total_pos, u_label_pos/u_total_pos*100))

u_label_neg = sum(users_df.loc[users_df['coded_as_witness'] == 0][comm_cols[0]].value_counts())
u_total_neg = users_df.loc[users_df['coded_as_witness'] == 0].shape[0]
print('{} of {} ({:.1f}%) negatively coded users are part of the giant component.'
          .format(u_label_neg, u_total_neg, u_label_neg/u_total_neg*100))

print('Difference in positive and negative proportions: {:.1f}%'.format((u_label_pos/u_total_pos-u_label_neg/u_total_neg)*100))


G_temp = get_graph_object()
H = nx.Graph(G_temp)
Hcc = max(nx.connected_components(H), key=len)
H0 = H.subgraph(Hcc)
print('\n{} of {} ({:.1f}%) nodes are part of the giant component and therefore have community labels.'
          .format(len(H0), len(G_temp), len(H0)/len(G_temp)*100))
print('{} of {} ({:.1f}%) edges are part of the giant component.'
          .format(H0.number_of_edges(), G_temp.number_of_edges(), H0.number_of_edges()/G_temp.number_of_edges()*100))


cc_lens = sorted([len(x) for x in nx.connected_components(H)], reverse=True)
print('\nThe second largest component has {} nodes. {:.2f}% of giant component.'.format(cc_lens[1], cc_lens[1]/cc_lens[0]*100))

u_label = u_label_pos = u_total_pos = u_label_neg = u_total_neg = G_temp = H = Hcc = H0 = None

```

    903 of 1500 (60.2%) coded users are part of the giant component and therefore have community labels.
    285 of 386 (73.8%) positively coded users are part of the giant component.
    618 of 1114 (55.5%) negatively coded users are part of the giant component.
    Difference in positive and negative proportions: 18.4%
    
    18409 of 31931 (57.7%) nodes are part of the giant component and therefore have community labels.
    76341 of 101096 (75.5%) edges are part of the giant component.
    
    The second largest component has 10 nodes. 0.05% of giant component.


Here we can see that the giant component encapsulates 58% of the total nodes, and 60% of the classed nodes. Community detection labels are therefore not applied to 597 of the classed nodes. Positive-classed nodes are more likely to be within the giant component (74% compared to 56%), which aligns with our hypothesis that witness nodes form communities together. Therefore, whether a node is not in the giant component (and therefore has no community label) is a useful observation, in addition to the labels and metrics calculated for  the nodes within the component.

As the second-largest component consists of only 10 nodes, we can safely ignore the structure of these disconnected subgraphs and effectively consider all nodes outside of the giant component as equally disconnected: essentially giving them a common community label: `-1`.

## Evaluating Community Metrics

Some algorithms create many small communities of 1-10 members, which will not be useful in generalisation of the model. Therefore discarding these is useful.

Chi-square tests are typically recommended to require contingency tables with values of at least 5, which is addressed (in part) by this step. Note that this does not elimnate all relevant cells in the confusion matrix -- e.g. a row with 5 positive and 0 negative should be excluded from chi-square statistics as the negative cell has a frequency <5. This is therefore properly sorted below.


```python
# # Ignore small communities where observed cases are too low (required for chi-square tests)
# # This is now handled instead in the chi-sq function's implementation
# MIN_COMMUNITY_SIZE = 10

# for col in comm_cols:
#     s = users_df[col].value_counts()
#     #s = users_df.loc[users_df['coded_as_witness']==1, col].value_counts()
#     users_df.loc[~users_df[col].isin(s.index[s >= MIN_COMMUNITY_SIZE]), col] = np.NaN

```


```python
X = users_df[comm_cols]
y = users_df['coded_as_witness']
```

As a preliminary visual investigation into the relationship between community and code, we can check the proportion of nodes within a community that are coded as positive cases (witnesses). In an independent case, we would expect these ratios to be similar, and reflect the overall ratio of cases, which is:
$$\frac{386}{1500} = 0.257$$
This is shown as a dotted line on the plots below.


```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [6, 4]


def create_plot_grid(df_list, df_list_secondary=None, titles=None, axhline=None, kind='bar'):
    '''
    Plots a list of dataframes or series in a single figure 
    with two columns.
    
    Plots a secondary value on a different scale if 
    df_list_secondary is passed. Either as a matching list,
    or list containing a single series.
    
    Can add a horizontal line at a value provided as axhline.
    '''

    # If df_list are series, use series names as titles
    if not titles:
        try:
            titles = [x.name for x in df_list]
        except:
            pass
    
    # Plot single chart without grid:
    if len(df_list) == 1:
        if titles:
            df_list[0].plot(kind=kind, colormap='Spectral', title=titles[0], rot=45)
            #ax.set_title(titles[0])
        else:
            df_list[0].plot(kind=kind, colormap='Spectral', rot=45)
        if not df_list_secondary == None:
            df_list_secondary[0].plot(kind=kind, secondary_y=True, legend=True)
        return
    
    # Create grid structure:
    ncol = 2
    nrow = int((len(df_list)+1)/2)
    # Temporarilty increase figsize:
    plt.rcParams['figure.figsize'] = [plt.rcParams['figure.figsize'][0] * 2, 
                                      plt.rcParams['figure.figsize'][1] * nrow]
    fig, axs = plt.subplots(nrow, ncol)
    fig.tight_layout(pad=4.0)
    # Delete last cell if odd number:
    if len(df_list) % 2 != 0:
        fig.delaxes(axs[nrow-1,1])
    
    # Populate grid with plots:
    for r in range(nrow):
        for c in range(ncol):
            count = r*2+c
            # Prevent trying last cell if odd length
            if count == len(df_list):
                break
            # Handle missing index dimensions for 2x1 grid
            if len(df_list) == 2:
                ax = axs[c]
            else:
                ax = axs[r,c]

            df_list[count].plot(kind=kind, ax=ax, colormap='Spectral', rot=45)
            if not df_list_secondary == None:
                try:
                    df_list_secondary[count].plot(kind=kind, ax=ax, secondary_y=True, legend=True)
                except: # One secondary series passed for all plots
                    df_list_secondary[0].plot(kind=kind, ax=ax, secondary_y=True, legend=True)
            
            if titles:
                #ax.set_title(titles[count])
                ax.text(.5, .9, titles[count], 
                            horizontalalignment='center', 
                            transform=ax.transAxes, 
                            bbox=dict(facecolor='Grey', alpha=0.5))
            if axhline:
                axs[r,c].axhline(y=axhline, color='b', linestyle='dotted', lw=2)
    
    # Reset figsize:
    plt.rcParams['figure.figsize'] = [plt.rcParams['figure.figsize'][0] / 2, 
                                      plt.rcParams['figure.figsize'][1] / nrow]
    return
```


```python
# Create dataframe for ratio of positive cases per community class, for each community algorithm
df_list = [users_df.loc[users_df['coded_as_witness']==1, col].dropna().value_counts() / 
               users_df[col].dropna().value_counts() 
               for col in comm_cols]

# Calculate expected proportion of positive cases given independence:
exp_pos_proportion = users_df['coded_as_witness'].value_counts()[1] / users_df.shape[0]

create_plot_grid(df_list, axhline=exp_pos_proportion)
```


![png](2_harvey_network_metrics_files/2_harvey_network_metrics_15_0.png)


From inspecting the graphs above, there appears to be a disproportionate amount of positive cases in certain communities, suggesting some association between community (as detected by a given algorithm) and the classification. Therefore, it is likely that including these metrics will increase the information available to the predictive models.

The charts shown above suggest that the highest proportion of positive classes appear in the largest, or second-largest communities (as the labels have been ranked in order of size). This is significant -- a model cannot be trained on community label as a feature, as the labels are qualitative and will be different each time an algorithm runs on a network. Therefore these features cannot generalise to new datasets. The feature that is supplied must therefore be something which is generalisable; in this case, the ranking of the community by size my be appropriate (for example, a feature which represents whether a user is in the largest detected community). Alternatively, communities may exhibit different characteristics such as connectedness. This will be explored later. The higher proportions seen in some of the later communities are less relevant as these are of a much smaller size. Thus the high proportions are 'easier' to achieve, and as smaller communities are more likely to represent unique cases, they are less likely to generalise.

The next steps in the analyis of the validity of this approach is the calculate whether the disparities observed above are statistically significant. That is, whether these associations could have been observed by chance.

Formally, for each community detection algorithm, we are testing the hypotheses:
$$H_0: \text{There is no association between the community label and witness label}$$
$$H_A: \text{There is an association between the community label and witness label}$$

### Chi-Square
A chi-squre analysis is performed on the output of each detection algorithm with the target class. Note that communities with a size below 5 are removed as per the recommendation for chi-square analysis.


```python
import scipy.stats as scs

def chi_square(X, y):
    '''
    Calculate chi-square statistic between two rows.
    Eliminates rows where cell value is below threshold
    as per chi-square recommendation.
    Note: Automatically drops NaN rows
    '''
    
    min_cell_val = 2
    confusion_matrix = pd.crosstab(X, y)
    
    if confusion_matrix.shape == (2,2):
        min_cell_val = 5
    
    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[0] < 2:
        print('WARNING: Single row or column provided: r={}, c={}'
                  .format(confusion_matrix.shape[0], confusion_matrix.shape[1]))
        return None
        
    # Eliminate rows where observed cases are below threshold:
    confusion_matrix = confusion_matrix[(confusion_matrix >= min_cell_val).all(1)]
    # Check dimensions, as eliminating shorter rows is preferred.
    if confusion_matrix.shape[0] < confusion_matrix.shape[1]:
        print('WARNING: check passed series order -- cell elimination performed row-wise. r={}, c={}'
                  .format(confusion_matrix.shape[0], confusion_matrix.shape[1]))
    
    # Eliminate rows where expected values are below threshold:
    exp_matrix = pd.DataFrame(scs.chi2_contingency(confusion_matrix)[3])
    exp_matrix.index = confusion_matrix.index
    
    # Repeat in case new cells become < threshold once rows are dropped
    while (exp_matrix >= min_cell_val).all(1).all() == False:
        confusion_matrix = confusion_matrix[(exp_matrix >= min_cell_val).all(1)]
        exp_matrix = pd.DataFrame(scs.chi2_contingency(confusion_matrix)[3])
        exp_matrix.index = confusion_matrix.index
    
    return scs.chi2_contingency(confusion_matrix)

```


```python
def get_chi2_df(X, y):
    cols = X.columns
    data = [list(chi_square(X[col], y)[:2]) for col in cols]
    chi2a_df = pd.DataFrame(data=data, columns=['chi-sq-a', 'chi-sq-a p-val'], index = cols)
    return chi2a_df

chi2a_df = get_chi2_df(X, y)
chi2a_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chi-sq-a</th>
      <th>chi-sq-a p-val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c_modularity</th>
      <td>211.186118</td>
      <td>7.455104e-40</td>
    </tr>
    <tr>
      <th>c_label_prop</th>
      <td>94.493169</td>
      <td>2.372655e-20</td>
    </tr>
    <tr>
      <th>c_label_prop_asyn</th>
      <td>56.978430</td>
      <td>2.597272e-12</td>
    </tr>
    <tr>
      <th>c_fluid</th>
      <td>143.881668</td>
      <td>7.807233e-28</td>
    </tr>
    <tr>
      <th>c_louvain</th>
      <td>200.477682</td>
      <td>4.387605e-35</td>
    </tr>
  </tbody>
</table>
</div>



For all features, the analysis produced a significant $\chi^2$ value, well beyond our $\alpha=0.05$. Therefore we can reject the null hypothesis and accept that the alternative: there is an association between community and witness status.

In the following cells, a number of other measures of association are tested, including the `sklearn` implementation of chi-square.

We are now interested in selecting the community algorithm which is most useful in class prediction.

### Chi-Square (sklearn)


```python
from sklearn.feature_selection import chi2

# This method drops NaN rows for any column, so more data is lost
# temp_df = users_df[comm_cols + ['coded_as_witness']].dropna()

# X = temp_df[comm_cols]
# y = temp_df['coded_as_witness']

# chi2b_df = pd.DataFrame(data={'feature': comm_cols, 'chi-sq-b': chi2(X,y)[0], 'chi-sq-b p-val': chi2(X,y)[1]})
# #chi2b_df.plot(kind='bar', y='chi2', x='feature', rot=45)
# chi2b_df

# This method calculates each metric separately, so fewer rows are dropped.
def get_chi2_sklearn_df(X, y):
    cols = X.columns    
#     data = [[k[0] for k in chi2(
#                                 X[col].dropna().to_frame(), 
#                                 y[X[col].notna()]
#                                 )
#             ] for col in cols]
    
    data = []
    for col in cols:
        data.append( [k[0] for k in chi2(
                                X[col].dropna().to_frame(), 
                                y[X[col].notna()])])
    chi2b_df = pd.DataFrame(data=data, columns=['chi-sq-b', 'chi-sq-b p-val'], index=cols)
    return chi2b_df

chi2b_df = get_chi2_sklearn_df(X, y)
chi2b_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chi-sq-b</th>
      <th>chi-sq-b p-val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c_modularity</th>
      <td>116.916736</td>
      <td>2.993758e-27</td>
    </tr>
    <tr>
      <th>c_label_prop</th>
      <td>5309.547802</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>c_label_prop_asyn</th>
      <td>7235.591424</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>c_fluid</th>
      <td>12.520518</td>
      <td>4.025072e-04</td>
    </tr>
    <tr>
      <th>c_louvain</th>
      <td>277.749038</td>
      <td>2.323415e-62</td>
    </tr>
  </tbody>
</table>
</div>



`sklearn.feature_selection.chi2` uses a different implementation of the chi-square formula, therefore these result differ from the results returned by `scipy.stats.scs.chi2_contingency`. Regardless, the each of the p-values is sufficiently small to reject the null hypothesis and accept that there is an association.

 A p-value close to zero means that our variables are very unlikely to be completely unassociated in some population. However, this does not mean the variables are strongly associated; a weak association in a large sample size may also result in p = 0.000.
 
 ### Cramer's V
 
Given that we would like to choose the algorithm which results in the communities with the strongest associations to the target label, we need another statistical measure. Cramérs $\phi$ (or V) coefficient is based upon the $\chi^2$ value and measures the strength the association between two categorical variables. Here, a score of $0$ represents no association, and $1$ represents a perfect association.

$${\phi}_c = \sqrt{\frac{\chi^2}{N(k-1)}}$$


```python
import numpy as np

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    #chi2 = scs.chi2_contingency(confusion_matrix)[0]
    chi2 = chi_square(x, y)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
```


```python
def get_cramers_df(X, y):
    cols = X.columns
    data = [cramers_v(X[col].dropna(), y[X[col].notna()]) for col in cols]
    cramers_df = pd.DataFrame(data=data, columns=['cramers_v'], index=cols)
    return cramers_df

cramers_df = get_cramers_df(X, y)
cramers_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cramers_v</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c_modularity</th>
      <td>0.402479</td>
    </tr>
    <tr>
      <th>c_label_prop</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>c_label_prop_asyn</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>c_fluid</th>
      <td>0.389545</td>
    </tr>
    <tr>
      <th>c_louvain</th>
      <td>0.426970</td>
    </tr>
  </tbody>
</table>
</div>



### Theil's U
While Cramer's Phi measures the strength of the association, Theil's U is a conditional measure. That is, it is able to measure how well we can predict one variable, given the other. Therefore, it is a more suitable statistic when evaluating features to use in prediction models.

In essence: 'given Y, what fraction of the bits of X can we predict?'



```python
import math
from collections import Counter

def conditional_entropy(x, y, log_base: float = math.e):
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy, log_base)
    return entropy

def theils_u(x, y):
    '''given Y, what fraction of the bits of X can we predict?'''
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = scs.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
```


```python
def get_theils_df(X, y):
    cols = X.columns
    # Swap X and y positions to match Theil format:
    data = [theils_u(y[X[col].notna()], X[col].dropna()) for col in cols]
    theils_df = pd.DataFrame(data=data, columns=['theils_u'], index=cols)
    return theils_df

theils_df = get_theils_df(X, y)
theils_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>theils_u</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c_modularity</th>
      <td>0.266495</td>
    </tr>
    <tr>
      <th>c_label_prop</th>
      <td>0.322568</td>
    </tr>
    <tr>
      <th>c_label_prop_asyn</th>
      <td>0.264045</td>
    </tr>
    <tr>
      <th>c_fluid</th>
      <td>0.132357</td>
    </tr>
    <tr>
      <th>c_louvain</th>
      <td>0.246875</td>
    </tr>
  </tbody>
</table>
</div>



### Random Forest Feature Selection
Decision Trees (and their ensemble random forest counterparts) can rank features on their importance, where their importance represents how much reduction of the (gini) uncertainty measure each feature contributes to the model. Note that due to how decision trees learn, these importance values can vary each time a model is trained, though this variance should be minimised in ensemble methods.


```python
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings

# Ignore FutureWarning from RF classifier
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_rf_importance_df(X, y):
    '''
    Returns importances of X variables as determined by
    random forest model.
    
    X: dataframe
    Y: series
    '''
    nan_rows = X.isna().any(1)
    y = y[~nan_rows]
    X = X[~nan_rows]

    #tree = DecisionTreeRegressor().fit(X, y)
    #tree = DecisionTreeClassifier().fit(X, y)
    tree = RandomForestClassifier(n_estimators=1000).fit(X, y)

    rf_df = pd.DataFrame(tree.feature_importances_, 
                   columns =['importance'], index=comm_cols)
    return rf_df


rf_df = get_rf_importance_df(users_df[comm_cols], users_df['coded_as_witness'])
#rf_df.plot(kind='bar', x='feature', rot=45)
rf_df 
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c_modularity</th>
      <td>0.262228</td>
    </tr>
    <tr>
      <th>c_label_prop</th>
      <td>0.130405</td>
    </tr>
    <tr>
      <th>c_label_prop_asyn</th>
      <td>0.123066</td>
    </tr>
    <tr>
      <th>c_fluid</th>
      <td>0.170965</td>
    </tr>
    <tr>
      <th>c_louvain</th>
      <td>0.313336</td>
    </tr>
  </tbody>
</table>
</div>



We can now compare the algorithms across each association measure. Theil's U and perhaps the random forest feature importance measures are more significant here. The Louvain and modularity measures appear to be the most promising. The asynchronous implementation of label propogation appears to outperform its synchronous counterpart, though it's overall relevance requires further investigation.


```python
df_list = [chi2a_df, chi2b_df, cramers_df, theils_df, rf_df]
# Plot only first column:
df_list = [x.iloc[:,0] for x in df_list]

create_plot_grid(df_list)
```


![png](2_harvey_network_metrics_files/2_harvey_network_metrics_30_0.png)


## Evaluation on Temporal Sub-Graphs

While the analysis above demonstrates the predictive power of these algorithms at the end point of an event, once the graph data is most rich, the purpose of this research is to classify user objects as they are detected. Therefore, we need to test the efficacy of these algorithms on partial graphs, as they appeared throughout the data collection process.

Data was collected over the course of several days for this event, therefore we can create subgraphs at intervals an re-calculate the metrics discussed above. This function creates a subgraph as it existed at every 24-hour interval from the event start time. Therefore, for the Hurricane Harvey event, which was recorded over 7 days and 10 hours, 7 sub-graphs are generated. With the final graph, there are a total of 8 graphs. Each graph contains all the data of the graph preceding it plus the newly observed users and their common relationships.

The same transormations as above are then applied and the association measures calculated. It is expected that the measures increase over time, as the network structure is revealed. We are therefore interested in observing the rate at which this happens.


```python
import networkx as nx
from datetime import timedelta

def get_graph_objects_time_sliced():
    """
    Creates subgraphs of the main event's graph every 24 hours 
    from start date.
    
    Sub-graphs are returned in a dictionary keyed by their
    slice index, where t_n = k*24h. The dictionary will not
    include the final, complete graph.
    """
    
    # Get event and calculate duration:
    e = Event.objects.all()[0]
    end = max(e.time_end, e.kw_stream_end, e.gps_stream_end)
    start = min(e.time_start, e.kw_stream_start, e.gps_stream_start)
    duration_days = (end - start).days
    print('Days in event: {}: {} '.format(e.name, duration_days))
    # Create timestamps every 24 hours:
    interim_time_slices = [start + timedelta(days=t+1) for t in range(duration_days)]
    # Reverse list to support progressively paring down subset queries:
    interim_time_slices.reverse()
    
    graph_dict = {}
    print('Getting user and edge list for full graph...')
    classed_users = User.objects.filter(user_class__gte=1)
    edges = Relo.objects.filter(target_user__in=classed_users, source_user__in=classed_users, end_observed_at=None)
    
    # Create subgraphs for each time stamp in reverse chronological order:
    for i in range(len(interim_time_slices)):
        filename = 'network_data_{}_tslice-{}.gexf'.format(EVENT_NAME, len(interim_time_slices)-i)
        try:
            # Load cached file if available
            G = nx.read_gexf(GRAPH_DIR + filename)
            print('Importing existing graph object for time slice {} ...'.format(i))
        except:
            print('Creating new graph object for time slice {}...'.format(i))
            # Subset classed users for those added prior to slice point:
            classed_users = classed_users.filter(added_at__lt=interim_time_slices[i])
            G=nx.DiGraph()
            for node in classed_users:
                try:
                    user_code = (node.coding_for_user.filter(coding_id=1)
                                .exclude(data_code__name='To Be Coded')[0]
                                .data_code.name)
                except:
                    user_code = ''
                G.add_node(node.screen_name, user_class=node.user_class, user_code=user_code)
            # Subset edges to those associated with subset of users
            edges = edges.filter(target_user__in=classed_users, 
                                    source_user__in=classed_users)
            edge_list = [(edge.source_user.screen_name, edge.target_user.screen_name) for edge in edges]
            G.add_edges_from(edge_list)
            # Write to file and re-import to bypass issue with community algorithms
            nx.write_gexf(G, GRAPH_DIR + filename, prettyprint=True)
            G = nx.read_gexf(GRAPH_DIR + filename)
            
        graph_dict[len(interim_time_slices)-i] = G
        
    return graph_dict
```


```python
g_dict = get_graph_objects_time_sliced()
```

    Days in event: Hurricane Harvey: 7 
    Getting user and edge list for full graph...
    Creating new graph object for time slice 0...
    Creating new graph object for time slice 1...
    Creating new graph object for time slice 2...
    Creating new graph object for time slice 3...
    Creating new graph object for time slice 4...
    Creating new graph object for time slice 5...
    Creating new graph object for time slice 6...


Calculate the community metrics for each subgraph and create time slice dataframes for each metric:


```python
# TODO: check this cell, move into function?

g_dict_comm = {}

tslice_chi2a_df = pd.DataFrame()
tslice_chi2b_df = pd.DataFrame()
tslice_cramers_df = pd.DataFrame()
tslice_theils_df = pd.DataFrame()
tslice_rf_df = pd.DataFrame()

    
# Calculate the community metrics for each subgraph:
for key in g_dict:
    print('Processing subgraph: ', key)
    e = Event.objects.all()[0]
    filename = 'network_data_{}_comm_tslice-{}.gexf'.format(e.name.replace(' ', ''), key)
    g2 = g_dict[key]
    
    # Write and reimport to avoid issue with community calculation:
    nx.write_gexf(g2, GRAPH_DIR + 'temp.gexf', prettyprint=True)
    g2 = nx.read_gexf(GRAPH_DIR + 'temp.gexf')
    os.remove(GRAPH_DIR + 'temp.gexf')
    
    # Calc community metrics for graph
    g2 = calc_community_metrics(g2, filename)
    g_dict_comm[key] = g2
    
    # Create community dataframe
    nodes = g2.nodes(data=True)
    df_comm_temp = pd.DataFrame.from_dict(dict(nodes), orient='index')
    df_comm_temp.drop(['user_class', 'user_code', 'label'], axis=1, inplace=True)
    
    comm_cols = list(df_comm_temp.columns)
    # Create copy of df and add comm data from slice:
    users_df_temp = users_df.copy().drop(columns=comm_cols)
    users_df_temp = pd.merge(left=users_df_temp, right=df_comm_temp, how='left', left_on='screen_name', right_index=True)

    # Ignore small communities where observed cases are too low (required for chi-square tests)
    MIN_COMMUNITY_SIZE = 5
    for col in comm_cols:
        s = users_df_temp[col].value_counts()
        users_df_temp.loc[~users_df_temp[col].isin(s.index[s >= MIN_COMMUNITY_SIZE]), col] = np.NaN

    X2 = users_df_temp[comm_cols]
    y2 = users_df_temp['coded_as_witness']
    # Chi-square
    tslice_chi2a_df[key] = get_chi2_df(X2, y2).iloc[:,0]
    # Chi-square sklearn
    tslice_chi2b_df[key] = get_chi2_sklearn_df(X2, y2).iloc[:,0]
    # Cramer's V
    tslice_cramers_df[key] = get_cramers_df(X2, y2).iloc[:,0]
    # Theil's U
    tslice_theils_df[key] = get_theils_df(X2, y2).iloc[:,0]
    # Random Forest
    tslice_rf_df[key] = get_rf_importance_df(X2, y2).iloc[:,0]
    
    users_df_temp = None
```

    Processing subgraph:  7
    Calculating community metrics for graph. 31390 nodes and 98838 edges...
    Largest component has 18067 nodes and 74594 edges.
    Calculating c_modularity...
    Calculating c_label_prop...
    Calculating c_label_prop_asyn...
    Calculating c_fluid...
    Calculating c_louvain...
    Adding data as node attributes...
    Writing to file...
    Processing subgraph:  6
    Calculating community metrics for graph. 28899 nodes and 84765 edges...
    Largest component has 16179 nodes and 63914 edges.
    Calculating c_modularity...
    Calculating c_label_prop...
    Calculating c_label_prop_asyn...
    Calculating c_fluid...
    Calculating c_louvain...
    Adding data as node attributes...
    Writing to file...
    Processing subgraph:  5
    Calculating community metrics for graph. 24625 nodes and 65207 edges...
    Largest component has 13163 nodes and 48919 edges.
    Calculating c_modularity...
    Calculating c_label_prop...
    Calculating c_label_prop_asyn...
    Calculating c_fluid...
    Calculating c_louvain...
    Adding data as node attributes...
    Writing to file...
    Processing subgraph:  4
    Calculating community metrics for graph. 19892 nodes and 47670 edges...
    Largest component has 10031 nodes and 35545 edges.
    Calculating c_modularity...
    Calculating c_label_prop...
    Calculating c_label_prop_asyn...
    Calculating c_fluid...
    Calculating c_louvain...
    Adding data as node attributes...
    Writing to file...
    Processing subgraph:  3
    Calculating community metrics for graph. 14021 nodes and 30094 edges...
    Largest component has 6407 nodes and 22153 edges.
    Calculating c_modularity...
    Calculating c_label_prop...
    Calculating c_label_prop_asyn...
    Calculating c_fluid...
    Calculating c_louvain...
    Adding data as node attributes...
    Writing to file...
    Processing subgraph:  2
    Calculating community metrics for graph. 9194 nodes and 17107 edges...
    Largest component has 3723 nodes and 12529 edges.
    Calculating c_modularity...
    Calculating c_label_prop...
    Calculating c_label_prop_asyn...
    Calculating c_fluid...
    Calculating c_louvain...
    Adding data as node attributes...
    Writing to file...
    Processing subgraph:  1
    Calculating community metrics for graph. 6616 nodes and 10268 edges...
    Largest component has 2288 nodes and 7405 edges.
    Calculating c_modularity...
    Calculating c_label_prop...
    Calculating c_label_prop_asyn...
    Calculating c_fluid...
    Calculating c_louvain...
    Adding data as node attributes...
    Writing to file...


We can now inspect the rate of growth over the collection period. Edges are only included if they are connected to an existing user, therefore their growth should accelerate as the available users with which to connect increases.


```python
node_counts = pd.Series([len(g) for g in g_dict_comm.values()] + [0]).iloc[::-1].reset_index(drop=True)
node_counts.name = 'nodes'
edge_counts = pd.Series([g.number_of_edges() for g in g_dict_comm.values()] + [0]).iloc[::-1].reset_index(drop=True)
edge_counts.name = 'edges'
dff = pd.concat([node_counts, edge_counts], axis=1)

dff.loc['final'] = [len(G), G.number_of_edges()]
dff.plot(title='Graph Growth per 24h')
dff
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nodes</th>
      <th>edges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2288</td>
      <td>9990</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3723</td>
      <td>16733</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6407</td>
      <td>29516</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10031</td>
      <td>46996</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13163</td>
      <td>64469</td>
    </tr>
    <tr>
      <th>6</th>
      <td>16179</td>
      <td>83982</td>
    </tr>
    <tr>
      <th>7</th>
      <td>18067</td>
      <td>98005</td>
    </tr>
    <tr>
      <th>final</th>
      <td>18409</td>
      <td>100257</td>
    </tr>
  </tbody>
</table>
</div>




![png](2_harvey_network_metrics_files/2_harvey_network_metrics_37_1.png)



```python
# Sort columns chronologically:
dfs = [tslice_chi2a_df, tslice_chi2b_df, tslice_cramers_df, tslice_theils_df, tslice_rf_df]

for df in dfs:
    df.sort_index(axis=1, inplace=True)
    
# Append the final graph values to each dataframe:
tslice_chi2a_df['final'] = chi2a_df.iloc[:,0]
tslice_chi2b_df['final'] = chi2b_df.iloc[:,0]
tslice_cramers_df['final'] = cramers_df.iloc[:,0]
tslice_theils_df['final'] = theils_df.iloc[:,0]
tslice_rf_df['final'] = rf_df.iloc[:,0]
```

The association measures can now be viewed over the time slice periods. Their growth over time will illustrate the emergence of the graph structure as seen in the final graph. Where association measures are poor for early values if `t`,  modularity measures may not provide adequate predictive power in earlier graphs and therefore using the metrics may not be feasible in a live classification application. Ideally, the higher association measures will emerge early.

We expect the chi-square values to generally increase as the graph size increases; this does not represent an increasing association but is a function of the formula. They are presented here for reference.


```python
colnames = tslice_chi2a_df.columns

print('chi-square a')
series_list = [tslice_chi2a_df.iloc[i] for i in range(len(tslice_chi2a_df))]
create_plot_grid(series_list)
```

    chi-square a



![png](2_harvey_network_metrics_files/2_harvey_network_metrics_40_1.png)



```python
print('chi-square b')
series_list = [tslice_chi2b_df.iloc[i] for i in range(len(tslice_chi2b_df))]
create_plot_grid(series_list)
```

    chi-square b



![png](2_harvey_network_metrics_files/2_harvey_network_metrics_41_1.png)



```python
print('Cramer\'s v')
series_list = [tslice_cramers_df.iloc[i] for i in range(len(tslice_cramers_df))]
create_plot_grid(series_list)
```

    Cramer's v



![png](2_harvey_network_metrics_files/2_harvey_network_metrics_42_1.png)



```python
print('Theil\'s U')
series_list = [tslice_theils_df.iloc[i] for i in range(len(tslice_theils_df))]
create_plot_grid(series_list)
```

    Theil's U



![png](2_harvey_network_metrics_files/2_harvey_network_metrics_43_1.png)



```python
print('RF Feature Importance')
series_list = [tslice_rf_df.iloc[i] for i in range(len(tslice_rf_df))]
create_plot_grid(series_list)
```

    RF Feature Importance



![png](2_harvey_network_metrics_files/2_harvey_network_metrics_44_1.png)


The stability in the Theil's U and Random Forest metrics suggest that the associations are strong even in smaller graphs. Therefore using these community data throughout the collection process in live classification is a viable strategy. Theil's U measures the fraction of Y (witness labels) we can predict using the community measures. The highest and most stable over time algorithms therefore appear to be c_modularity, c_label_prop_async, and c_louvain. Each of these shows a value of approximately .20-.25 over the course of the collection period.

As community labels are qualitative, we now need to convert them to quantifiable metrics so that we can train a model that is able to generalise to new data (rather than simply learning the community label name). 

## Quantifying Community Structure

The analysis above has measured whether there is an association between the (qualitative) community label, and the target class. This association has been shown to exist, meaning 'witness' users tend to co-appear in certain communities. To be able to use community data in a machine-learning model, the communities labels must be converted into a value which can be generalised to new datasets. A clustering algorithm may label communities arbitrarily and this will be different on each dataset, therefore these labels must be converted into some measure that can be generalised between datasets.

In the current structure, communities are ranked by size. This could be a suitable metric: perhaps witness accounts are most common in the largest community. This has been observed in the output of, for example, the `c_fluid` algorithm, but is not consistant with the other algorithms' labels.

In this section, various measures by which to characterise the communities are calculated, and their correlation to the target label is evaluated. In essence, we are hoping that the network structure within the communities with more 'witness' nodes is significantly different to the other communities.


```python
def calc_network_metrics(G):
    ''' Calculates various metrics summarising a graph object '''
    # TODO: Consider evaluating reciprocal relationship rate.

    result_dict = {}
    
    # Create undirected graph:
    G = nx.Graph(G)
    #### TESTING ####
    if not nx.is_connected(G):
        print('ERROR: undirected graph not connected! Taking largest component from lengths:')
        print([len(x) for x in nx.connected_components(G)])
        Gcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(Gcc)
    #################
    n = len(G)
    
    result_dict['nodes'] = n
    result_dict['edges'] = G.number_of_edges()
    
    avg_degree = G.number_of_edges() / n
    result_dict['avg_degree'] = avg_degree
    
    avg_shortest_path_length = nx.average_shortest_path_length(G)
    result_dict['avg_shortest_path_length'] = avg_shortest_path_length
    
    # Generate Erdős-Rényi graph or a binomial graph.
    max_edges = n*(n-1)/2
    pr_edge = G.number_of_edges() / max_edges
    R = nx.gnp_random_graph(n, pr_edge)
    
    # Max shortest path (diameter) and expected diameter:
    result_dict['diameter'] = nx.diameter(G)
    # result_dict['ex_diameter'] = nx.diameter(nx.connected_component_subgraphs(R)[0])
    Rcc = max(nx.connected_components(R), key=len)
    result_dict['ex_diameter'] = nx.diameter(R.subgraph(Rcc))
    result_dict['diameter_diff'] = result_dict['diameter'] - result_dict['ex_diameter']
    
    # Transitivity: fraction of all possible triangles present in G. (global clustering)
    transitivity = nx.transitivity(G)
    result_dict['transitivity'] = transitivity
    ex_transitivity = nx.transitivity(R)
    result_dict['ex_transitivity'] = ex_transitivity
    result_dict['transitivity_diff'] = transitivity - ex_transitivity
    
    # Clustering: fraction of possible triangles through a node that exist
    avg_clustering = nx.average_clustering(G)
    result_dict['avg_clustering'] = avg_clustering
    
    # The degree centrality for a node v is the fraction of nodes connected to it
    degree_centrality = nx.degree_centrality(G)
    avg_degree_centrality = sum([v for k, v in degree_centrality.items()]) / n
    result_dict['avg_degree_centrality'] = avg_degree_centrality
    
    # Eigenvector centrality computes the centrality for a node based on the centrality of its neighbors.
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    avg_eigenvector_centrality = sum([v for k, v in eigenvector_centrality.items()]) / n
    result_dict['avg_eigenvector_centrality'] = avg_eigenvector_centrality
    
    # Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v
    betweenness_centrality = nx.betweenness_centrality(G)
    avg_betweenness_centrality = sum([v for k, v in betweenness_centrality.items()]) / n
    result_dict['avg_betweenness_centrality'] = avg_betweenness_centrality
    
    # The load centrality of a node is the fraction of all shortest paths that pass through that node. (Load centrality is slightly different than betweenness)
    load_centrality = nx.load_centrality(G)
    avg_load_centrality = sum([v for k, v in load_centrality.items()]) / n
    result_dict['avg_load_centrality'] = avg_load_centrality
    
    # Closeness centrality [1] of a node u is the reciprocal of the sum of the shortest path distances from u to all n-1 other nodes
    closeness_centrality = nx.closeness_centrality(G)
    avg_closeness_centrality = sum([v for k, v in closeness_centrality.items()]) / n
    result_dict['avg_closeness_centrality'] = avg_closeness_centrality

    return result_dict
    
```


```python
def get_network_metrics_df(subgraphs_dict, title=None):
    '''
    Calculate network metrics for each subgraph in dict.
    
    Saves output to file, which is returned if available
    to avoid recalculation
    '''

    # Return cached file if available.
    if title:
        filename = 'network_data_{}_comm_metrics_{}.csv'.format(EVENT_NAME, title)
        try:
            results_df = pd.read_csv(DIR + filename, index_col=0)
            print('Returning cached file...')
            return results_df
        except:
            pass
    
    results_df = pd.DataFrame()

    print('Total community subgraphs: ', len(subgraphs_dict))
    for k, v in subgraphs_dict.items():
        ############ TODO: If subgraph too small, return NANS ?? ############
        print('Calculating metrics for community ID: ', k)
        results = calc_network_metrics(v)    
        results_df[k] = results.values()

    results_df.index = results.keys()
    results_df = results_df.T

    if not results_df['nodes'].is_monotonic_decreasing:
        print('WARNING: Communities not sorted by size')

    # Save dataframe to cached file:
    if title:
        results_df.to_csv(DIR + filename)
        
    return results_df
```


```python
def get_subgraphs_by_attr(G, attr, min_subgraph_size=0):
    '''
    Returns a dictionary of subgraphs of graph G
    based on the attribute passed.
    Dictionary is keyed by attribute label.
    '''
    subgraph_dict = {}
    node_data = list(nx.get_node_attributes(G, attr).values())
    labels = set(node_data)

    for label in labels:
        #TODO: Remove this threshold and the min_subgraph_size argument above.
#         # Exclude subgraphs below provided threshold.
#         count = sum([1 for x in node_data if x == label])
#         if count < min_subgraph_size:
#             continue
        # Node generator:
        nodes = ( node for node, data in G.nodes(data=True) if data.get(attr) == label )
        subgraph = G.subgraph(nodes)
        subgraph_dict[label] = subgraph
    return subgraph_dict
```

For now, we can focus on the most promising community algorithm labels identified above: `c_modularity`, `c_label_prop_asyn`, `c_fluid`, `c_louvain`.

We exclude communities smaller than 50.

For each algorithm we do the following:
* The complete graph is partitioned by the algorithm's labels, and the network metrics for each subgraph are then calculated and stored in a dataframe.

* We then calculate the ratio of positive-classed 'witness' cases against the total classed cases. As coding was done at random, some communities will have very few classed cases and therefore their ratios may not be statistically significant.

* Once these ratios are added to the community results dataframe, we can sort by their value and inspect the various metrics for possible relationships -- ie. which network metrics are associated with the ratio of positive cases.


```python
def get_community_metrics_df(G, comm_name, users_df):
    
    # Split graph by comm_name labels, and calculate network metrics per sub-graph:
    subgraphs_dict = get_subgraphs_by_attr(G, comm_name)
    # Ignore subgraphs with no coded user members:
    orig_len = len(subgraphs_dict)
    comm_ids_with_coded = users_df[comm_name].value_counts().index
    subgraphs_dict = {k: subgraphs_dict[k] for k in comm_ids_with_coded}
    print('Processing {} subgraphs containing coded users from total: {}. '.format(len(subgraphs_dict), orig_len))
    
    results_df = get_network_metrics_df(subgraphs_dict, comm_name)

    # Calculate ratio of positive cases per community label and add to dataframe:
    users_df_for_comm = users_df.loc[users_df[comm_name].notna() == True]
    ratio_list = []
    total_list = []
    for comm_id in results_df.index:
        tot = users_df_for_comm.loc[users_df_for_comm[comm_name] == comm_id]
        pos = tot.loc[tot['coded_as_witness'] == 1]
        if tot.shape[0] > 0:
            ratio_list.append(pos.shape[0]/tot.shape[0])
        else:
            ratio_list.append(None)
        total_list.append(tot.shape[0])
    results_df['pos_ratio'] = ratio_list
    results_df['total_coded'] = total_list

    # Sort dataframe by pos_ratio (high to low)
    results_df = results_df.sort_values(by=['pos_ratio'], ascending=False).reset_index()
    results_df.rename(columns={'index': 'size_rank'}, inplace=True)

    return results_df
```


```python
comm_names = ['c_modularity', 'c_label_prop', 'c_label_prop_asyn', 'c_fluid', 'c_louvain']
selected_comm_names = ['c_modularity', 'c_label_prop_asyn', 'c_fluid', 'c_louvain']
results_col_names = []
df_list_array = []
df_list_secondary_array = []

for comm_name in selected_comm_names:
    results_df = get_community_metrics_df(G, comm_name, users_df)
    results_col_names = results_df.columns

    # Trim by size of community and amount of cases coded within each:
    #min_community_size = 1000
    min_community_size = 0
    min_total_coded = 1
    results_sub_df = results_df.loc[
                                (results_df['nodes'] >= min_community_size) & 
                                (results_df['total_coded'] >= min_total_coded)
                                    ].reset_index(drop=True)
    
    # Create sub-df and save to file:
    users_df_temp = pd.merge(left=users_df, right=results_sub_df, how='left', left_on=comm_name, right_on='size_rank')
    users_df_temp.drop(columns=[x for x in comm_names if x != comm_names], inplace=True)
    filename = 'df_users_{}_comm_metrics_{}.csv'.format(EVENT_NAME, comm_name)
    users_df_temp.to_csv(DIR + filename)
    users_df_temp = None
    
    # Add to array for ploting:
    df_list_array.append( [results_sub_df[[c]] for c in results_sub_df.columns] )
    df_list_secondary_array.append( [results_sub_df['pos_ratio']] )
```

    Original dict length:  266
    New dict length:  66
    Returning cached file...
    Original dict length:  1485
    New dict length:  200
    Total community subgraphs:  200
    Calculating metrics for community ID:  0.0
    Calculating metrics for community ID:  1.0
    Calculating metrics for community ID:  2.0
    Calculating metrics for community ID:  12.0
    Calculating metrics for community ID:  3.0
    Calculating metrics for community ID:  38.0
    Calculating metrics for community ID:  4.0
    Calculating metrics for community ID:  6.0
    Calculating metrics for community ID:  19.0
    Calculating metrics for community ID:  43.0
    Calculating metrics for community ID:  48.0
    Calculating metrics for community ID:  17.0
    Calculating metrics for community ID:  656.0
    Calculating metrics for community ID:  27.0
    Calculating metrics for community ID:  5.0
    Calculating metrics for community ID:  317.0
    Calculating metrics for community ID:  155.0
    Calculating metrics for community ID:  26.0
    Calculating metrics for community ID:  45.0
    Calculating metrics for community ID:  14.0
    Calculating metrics for community ID:  32.0
    Calculating metrics for community ID:  15.0
    Calculating metrics for community ID:  987.0
    Calculating metrics for community ID:  550.0
    Calculating metrics for community ID:  31.0
    Calculating metrics for community ID:  1081.0
    Calculating metrics for community ID:  873.0
    Calculating metrics for community ID:  503.0
    Calculating metrics for community ID:  501.0
    Calculating metrics for community ID:  998.0
    Calculating metrics for community ID:  562.0
    Calculating metrics for community ID:  495.0
    Calculating metrics for community ID:  1230.0
    Calculating metrics for community ID:  560.0
    Calculating metrics for community ID:  461.0
    Calculating metrics for community ID:  1085.0
    Calculating metrics for community ID:  379.0
    Calculating metrics for community ID:  637.0
    Calculating metrics for community ID:  690.0
    Calculating metrics for community ID:  1387.0
    Calculating metrics for community ID:  1414.0
    Calculating metrics for community ID:  181.0
    Calculating metrics for community ID:  129.0
    Calculating metrics for community ID:  523.0
    Calculating metrics for community ID:  602.0
    Calculating metrics for community ID:  1020.0
    Calculating metrics for community ID:  574.0
    Calculating metrics for community ID:  350.0
    ERROR: undirected graph not connected! Taking largest component from lengths:
    [2, 2]
    Calculating metrics for community ID:  266.0
    Calculating metrics for community ID:  731.0
    Calculating metrics for community ID:  1384.0
    Calculating metrics for community ID:  327.0
    Calculating metrics for community ID:  496.0
    Calculating metrics for community ID:  239.0
    Calculating metrics for community ID:  249.0
    Calculating metrics for community ID:  980.0
    Calculating metrics for community ID:  1097.0
    Calculating metrics for community ID:  1455.0
    Calculating metrics for community ID:  655.0
    Calculating metrics for community ID:  697.0
    Calculating metrics for community ID:  7.0
    Calculating metrics for community ID:  240.0
    Calculating metrics for community ID:  106.0
    Calculating metrics for community ID:  880.0
    Calculating metrics for community ID:  832.0
    Calculating metrics for community ID:  944.0
    Calculating metrics for community ID:  86.0
    Calculating metrics for community ID:  164.0
    Calculating metrics for community ID:  40.0
    Calculating metrics for community ID:  152.0
    Calculating metrics for community ID:  24.0
    Calculating metrics for community ID:  144.0
    Calculating metrics for community ID:  55.0
    Calculating metrics for community ID:  448.0
    Calculating metrics for community ID:  63.0
    Calculating metrics for community ID:  35.0
    Calculating metrics for community ID:  11.0
    ERROR: undirected graph not connected! Taking largest component from lengths:
    [32, 2]
    Calculating metrics for community ID:  220.0
    Calculating metrics for community ID:  13.0
    Calculating metrics for community ID:  44.0
    Calculating metrics for community ID:  36.0
    Calculating metrics for community ID:  21.0
    Calculating metrics for community ID:  10.0
    ERROR: undirected graph not connected! Taking largest component from lengths:
    [32, 3]
    Calculating metrics for community ID:  122.0
    Calculating metrics for community ID:  20.0
    Calculating metrics for community ID:  148.0
    Calculating metrics for community ID:  124.0
    Calculating metrics for community ID:  47.0
    Calculating metrics for community ID:  9.0
    Calculating metrics for community ID:  88.0
    Calculating metrics for community ID:  384.0
    Calculating metrics for community ID:  1105.0
    Calculating metrics for community ID:  30.0
    Calculating metrics for community ID:  53.0
    Calculating metrics for community ID:  25.0
    Calculating metrics for community ID:  912.0
    Calculating metrics for community ID:  360.0
    Calculating metrics for community ID:  39.0
    Calculating metrics for community ID:  102.0
    Calculating metrics for community ID:  896.0
    Calculating metrics for community ID:  893.0
    Calculating metrics for community ID:  699.0
    Calculating metrics for community ID:  797.0
    Calculating metrics for community ID:  278.0
    Calculating metrics for community ID:  834.0
    Calculating metrics for community ID:  310.0
    Calculating metrics for community ID:  649.0
    Calculating metrics for community ID:  724.0
    Calculating metrics for community ID:  263.0
    Calculating metrics for community ID:  1368.0
    Calculating metrics for community ID:  1195.0
    Calculating metrics for community ID:  551.0
    Calculating metrics for community ID:  929.0
    Calculating metrics for community ID:  531.0
    Calculating metrics for community ID:  483.0
    Calculating metrics for community ID:  770.0
    Calculating metrics for community ID:  375.0
    ERROR: undirected graph not connected! Taking largest component from lengths:
    [2, 2]
    Calculating metrics for community ID:  1364.0
    Calculating metrics for community ID:  1068.0
    Calculating metrics for community ID:  619.0
    Calculating metrics for community ID:  305.0
    Calculating metrics for community ID:  526.0
    Calculating metrics for community ID:  721.0
    Calculating metrics for community ID:  309.0
    Calculating metrics for community ID:  1114.0
    Calculating metrics for community ID:  290.0
    Calculating metrics for community ID:  614.0
    Calculating metrics for community ID:  585.0
    Calculating metrics for community ID:  378.0
    Calculating metrics for community ID:  663.0
    Calculating metrics for community ID:  1429.0
    Calculating metrics for community ID:  291.0
    Calculating metrics for community ID:  437.0
    Calculating metrics for community ID:  1148.0
    Calculating metrics for community ID:  978.0
    Calculating metrics for community ID:  825.0
    Calculating metrics for community ID:  293.0
    Calculating metrics for community ID:  1022.0
    Calculating metrics for community ID:  1123.0
    Calculating metrics for community ID:  579.0
    Calculating metrics for community ID:  609.0
    Calculating metrics for community ID:  535.0
    Calculating metrics for community ID:  981.0
    Calculating metrics for community ID:  1169.0
    Calculating metrics for community ID:  916.0
    Calculating metrics for community ID:  347.0
    Calculating metrics for community ID:  366.0
    Calculating metrics for community ID:  145.0
    Calculating metrics for community ID:  986.0
    Calculating metrics for community ID:  179.0
    Calculating metrics for community ID:  827.0
    Calculating metrics for community ID:  657.0
    Calculating metrics for community ID:  1193.0
    Calculating metrics for community ID:  552.0
    Calculating metrics for community ID:  479.0
    Calculating metrics for community ID:  792.0
    Calculating metrics for community ID:  374.0
    Calculating metrics for community ID:  71.0
    ERROR: undirected graph not connected! Taking largest component from lengths:
    [6, 2]
    Calculating metrics for community ID:  1054.0
    Calculating metrics for community ID:  1207.0
    Calculating metrics for community ID:  611.0
    Calculating metrics for community ID:  559.0
    Calculating metrics for community ID:  1029.0
    Calculating metrics for community ID:  630.0
    Calculating metrics for community ID:  349.0
    Calculating metrics for community ID:  391.0
    Calculating metrics for community ID:  273.0
    Calculating metrics for community ID:  566.0
    Calculating metrics for community ID:  570.0
    Calculating metrics for community ID:  733.0
    Calculating metrics for community ID:  1180.0
    Calculating metrics for community ID:  866.0
    Calculating metrics for community ID:  671.0
    Calculating metrics for community ID:  267.0
    Calculating metrics for community ID:  1427.0
    Calculating metrics for community ID:  202.0
    Calculating metrics for community ID:  171.0
    Calculating metrics for community ID:  95.0
    Calculating metrics for community ID:  230.0
    Calculating metrics for community ID:  109.0
    Calculating metrics for community ID:  372.0
    Calculating metrics for community ID:  125.0
    ERROR: undirected graph not connected! Taking largest component from lengths:
    [3, 3]
    Calculating metrics for community ID:  276.0
    Calculating metrics for community ID:  174.0
    Calculating metrics for community ID:  214.0
    Calculating metrics for community ID:  292.0
    Calculating metrics for community ID:  584.0
    Calculating metrics for community ID:  246.0
    Calculating metrics for community ID:  194.0
    Calculating metrics for community ID:  146.0
    Calculating metrics for community ID:  492.0
    Calculating metrics for community ID:  1304.0
    Calculating metrics for community ID:  1328.0
    Calculating metrics for community ID:  242.0
    Calculating metrics for community ID:  876.0
    Calculating metrics for community ID:  170.0
    Calculating metrics for community ID:  778.0
    Calculating metrics for community ID:  85.0
    Calculating metrics for community ID:  203.0
    Calculating metrics for community ID:  1374.0
    WARNING: Communities not sorted by size
    Original dict length:  8
    New dict length:  8
    Total community subgraphs:  8
    Calculating metrics for community ID:  1.0
    ERROR: undirected graph not connected! Taking largest component from lengths:
    [2698, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1]
    Calculating metrics for community ID:  2.0
    ERROR: undirected graph not connected! Taking largest component from lengths:
    [2691, 1, 3, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1]
    Calculating metrics for community ID:  0.0
    ERROR: undirected graph not connected! Taking largest component from lengths:
    [2540, 3, 2, 2, 5, 2, 11, 1, 1, 2, 1, 1, 4, 2, 2, 5, 3, 9, 1, 3, 13, 1, 3, 5, 2, 1, 2, 6, 2, 6, 2, 1, 2, 4, 2, 1, 3, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 3, 1, 1, 2, 2, 4, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 4, 1, 3, 2, 1, 2, 3, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 3, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Calculating metrics for community ID:  3.0
    ERROR: undirected graph not connected! Taking largest component from lengths:
    [2459, 3, 3, 2, 1, 7, 3, 4, 1, 1, 1, 1, 1, 1, 1]
    Calculating metrics for community ID:  6.0
    ERROR: undirected graph not connected! Taking largest component from lengths:
    [1361, 2, 11, 1, 1, 1, 1, 8, 2, 13, 1, 10, 2, 1, 2, 1, 2, 6, 8, 1, 5, 2, 1, 5, 1, 6, 1, 4, 1, 1, 2, 1, 2, 2, 12, 2, 4, 4, 4, 11, 5, 1, 4, 1, 1, 1, 1, 6, 1, 3, 2, 1, 2, 2, 1, 5, 4, 6, 4, 4, 3, 1, 2, 2, 1, 3, 1, 1, 1, 6, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 4, 4, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 5, 1, 2, 1, 3, 1, 1, 1, 2, 1, 2, 3, 1, 1, 1, 3, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 3, 3, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 3, 1, 4, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Calculating metrics for community ID:  5.0
    ERROR: undirected graph not connected! Taking largest component from lengths:
    [684, 1, 33, 2, 155, 3, 2, 2, 4, 1, 19, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 18, 4, 3, 1, 5, 19, 3, 1, 6, 15, 2, 1, 1, 1, 1, 1, 2, 6, 6, 1, 1, 1, 1, 9, 6, 8, 3, 1, 1, 2, 2, 1, 2, 1, 10, 3, 2, 2, 4, 1, 1, 1, 2, 2, 6, 4, 7, 2, 1, 1, 3, 2, 1, 5, 5, 1, 2, 18, 1, 12, 1, 4, 2, 4, 1, 10, 4, 2, 1, 2, 3, 1, 11, 1, 5, 2, 3, 4, 2, 3, 4, 1, 3, 8, 3, 2, 1, 13, 1, 2, 2, 4, 2, 1, 7, 1, 1, 1, 4, 1, 9, 1, 4, 1, 1, 1, 2, 1, 9, 2, 3, 1, 1, 1, 2, 2, 1, 5, 3, 2, 2, 1, 1, 1, 7, 3, 2, 2, 3, 3, 3, 2, 1, 3, 2, 1, 3, 6, 4, 1, 3, 1, 12, 1, 4, 1, 3, 1, 2, 1, 3, 1, 3, 1, 1, 2, 1, 1, 4, 2, 1, 3, 1, 10, 2, 2, 1, 1, 4, 2, 1, 1, 1, 1, 3, 1, 3, 2, 1, 1, 5, 1, 3, 5, 2, 5, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 2, 5, 2, 3, 1, 5, 1, 3, 1, 1, 2, 1, 2, 1, 5, 6, 1, 5, 3, 1, 1, 1, 1, 1, 2, 6, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 6, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 5, 1, 1, 3, 1, 3, 4, 4, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 2, 3, 4, 2, 2, 1, 1, 1, 1, 4, 1, 5, 6, 1, 2, 2, 1, 3, 2, 1, 1, 2, 1, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Calculating metrics for community ID:  7.0
    ERROR: undirected graph not connected! Taking largest component from lengths:
    [1539, 1, 2, 1, 7, 1, 4, 1, 8, 12, 4, 3, 4, 1, 2, 6, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 5, 5, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 4, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1]
    Calculating metrics for community ID:  4.0
    ERROR: undirected graph not connected! Taking largest component from lengths:
    [2095, 3, 1, 1, 1, 1, 4, 1, 1, 4, 6, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 5, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    WARNING: Communities not sorted by size
    Original dict length:  73
    New dict length:  37
    Total community subgraphs:  37
    Calculating metrics for community ID:  0.0
    Calculating metrics for community ID:  2.0
    Calculating metrics for community ID:  1.0
    Calculating metrics for community ID:  3.0
    Calculating metrics for community ID:  4.0
    Calculating metrics for community ID:  6.0
    Calculating metrics for community ID:  7.0
    Calculating metrics for community ID:  5.0
    Calculating metrics for community ID:  8.0
    Calculating metrics for community ID:  10.0
    Calculating metrics for community ID:  9.0
    Calculating metrics for community ID:  11.0
    Calculating metrics for community ID:  15.0
    Calculating metrics for community ID:  22.0
    Calculating metrics for community ID:  12.0
    Calculating metrics for community ID:  17.0
    Calculating metrics for community ID:  14.0
    Calculating metrics for community ID:  18.0
    Calculating metrics for community ID:  19.0
    Calculating metrics for community ID:  16.0
    Calculating metrics for community ID:  26.0
    Calculating metrics for community ID:  13.0
    Calculating metrics for community ID:  20.0
    Calculating metrics for community ID:  36.0
    Calculating metrics for community ID:  33.0
    Calculating metrics for community ID:  23.0
    Calculating metrics for community ID:  35.0
    Calculating metrics for community ID:  25.0
    Calculating metrics for community ID:  32.0
    Calculating metrics for community ID:  31.0
    Calculating metrics for community ID:  63.0
    Calculating metrics for community ID:  29.0
    Calculating metrics for community ID:  58.0
    Calculating metrics for community ID:  55.0
    Calculating metrics for community ID:  27.0
    ERROR: undirected graph not connected! Taking largest component from lengths:
    [3, 3, 4, 3, 3]
    Calculating metrics for community ID:  44.0
    Calculating metrics for community ID:  69.0
    WARNING: Communities not sorted by size



```python
# Plot:
for c in len(df_list_array):
    create_plot_grid(df_list_array[c], df_list_secondary_array[c], kind='line')
```


![png](2_harvey_network_metrics_files/2_harvey_network_metrics_52_0.png)


The community network metrics have now been calculated and added to a copy of the users dataframe for each algorithm, creating a set of enhanced dataframes. The graphs above indicate a potential relationship between the positive ratio and certain centrality measures.

We can now check each dataframe for correlative properties (that is, a linear relationship between the calculated metric and the coded label).


```python
# TODO: unbalanced community size resulting in useless results here:\

from sklearn.linear_model import LogisticRegression

#### TESTING ####
import matplotlib.pyplot as plt
#################


comm_names = ['c_modularity', 'c_label_prop_asyn', 'c_fluid', 'c_louvain']
#comm_names = ['c_label_prop_asyn']

results_col_names = ['size_rank', 'nodes', 'edges', 'avg_degree',
                       'avg_shortest_path_length', 'diameter', 'ex_diameter', 'diameter_diff',
                       'transitivity', 'ex_transitivity', 'transitivity_diff',
                       'avg_clustering', 'avg_degree_centrality', 'avg_eigenvector_centrality',
                       'avg_betweenness_centrality', 'avg_load_centrality',
                       'avg_closeness_centrality']


score_df = pd.DataFrame(index = results_col_names)

#### TESTING ####
plot = True
#################

for comm_name in comm_names:
    results = []
    
    filename = 'df_users_{}_comm_metrics_{}.csv'.format(EVENT_NAME, comm_name)

    users_df_temp = pd.read_csv(DIR + filename, index_col=0)
    # Drop NaN rows: TODO: should be labelling these rows as their own disconnected community instead?
    users_df_temp = users_df_temp.loc[users_df_temp[results_col_names].notna().all(axis=1)]
    y = users_df_temp['coded_as_witness']
    #print('positive class ratio:', sum(y)/y.shape[0])
    #users_df_temp = users_df_temp[results_col_names]

    # Calculate logreg scores for each value and add column to dataframe
    for col in results_col_names:
        X = users_df_temp[col]
        X = X.values.reshape(-1,1)
        clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X, y)
        #print(col, ' Score: {:.5f}'.format(clf.score(X, y)))
        results.append(clf.score(X, y))
        
        #### TESTING ####
        if plot:
            print(col)
            plt.figure(1, figsize=(8, 6))
            #plt.clf()
            plt.scatter(X, y, color='black', zorder=20)
            #plt.plot(X, clf.coef_ * X + clf.intercept_, linewidth=1)
            plot = False
        #################

    score_df[comm_name] = results
    
#score_df
```

    size_rank



![png](2_harvey_network_metrics_files/2_harvey_network_metrics_54_1.png)



```python
# Stop running here

# Work out way to measure correlation between numeric and categorical
## point biserial correlation, logistic regression, and Kruskal Wallis H Test
## Measure corr for each col for each dataframe

# create basic ML notebook for testing
# TODO: Check visualisation of fluid network communities which appear v disconnected.

raise Exception
```

### Visual Inspection


```python
df_list = [results_df[[c]] for c in results_df.columns]
df_list_secondary = [results_df['pos_ratio']]

create_plot_grid(df_list, df_list_secondary, kind='line')
```

While there appears to be regular patterns in the data above, we must remember that many of the calculated metrics are influcenced by the size of the graph. As the size also influences the amount of coded examples within the graph, we cannot interpret these results meaningfully. For example, the four smallest graphs have ~60 nodes and no coded cases. Three graphs have zero positive cases, but only a single negative code, which isn't statistically significant. Therefore, we can drop communities below a certain size and re-examine the relationships.

As the network strutures appear as the size of the network grows, we can also exlude communities below a certain size:


```python
min_community_size = 1000
min_total_coded = 0

temp_df = tdf.loc[(tdf['nodes'] >= min_community_size) & (tdf['total_coded'] >= min_total_coded)].reset_index(drop=True)
df_list = [temp_df[[c]] for c in tdf.columns]
df_list_secondary = [temp_df[['pos_ratio']] for x in range(len(df_list))]

create_plot_grid(df_list, df_list_secondary, kind='line')
```


```python

```

### Adding to Dataframe
We can simply add all metrics to the DF for testing with various ML models


```python
#users_df.head()
#results_sub_df.head()
results_df.head()
```


```python
# userst_df = users_df.copy()
# userst_df.loc[userst_df['c_label_prop_asyn']==3].head()
userst_df = None
```


```python
userst_df = pd.merge(left=userst_df, right=results_sub_df, how='left', left_on='c_label_prop_asyn', right_on='size_rank')
userst_df.drop(columns=[x for x in comm_names if x != 'c_label_prop_asyn'], inplace=True)
```


```python
sum(userst_df['ex_transitivity'].value_counts())
```


```python

```
