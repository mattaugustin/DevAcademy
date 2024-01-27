## BRIEF   BRIEF  BRIEF   BRIEF  BRIEF 

# Using the linked dataset below produce a series of visualisations demonstrating the use of Matplotlib.
# You should produce at least one of the following: 
# Bar Graph. 
# Pie Chart. 
# Box Plot.  
# Line Chart.
# Scatter Plot.
# Once you have demonstrated use of the different plots you should look at the data and make some conclusions from the data set. 
# Can you find any links?

# Present your conclusion illustrated with appropriate visualisations.
# Submit your completed notebook.

###################################################################################################################################################################
###################################################################################################################################################################


from pandas_profiling import ProfileReport, profile_report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


earthquakes_df = pd.read_csv("all_month.csv")
earthquakes_df.info()
earthquakes_df.describe()

earthquakes_df.head(5)


##


# How long is the data covering ? Using the first and last two dates in the table 
earthquakes_df = earthquakes_df.sort_values("time")
pd.concat([ earthquakes_df.head(2) , earthquakes_df.tail(2) ], axis = 0)["time"]



### A  very useful tool to have quick information on multiple aspects of the table (missing items, distribution, extreme values etc...)
profile = ProfileReport(
    df = earthquakes_df
)
profile



######  1a  BAR GRAPH    WITH EARTHQUAKE TYPES    #######

# Are we only talking about earthquakes in the dataframe ?

quake_type = earthquakes_df['type'].value_counts()
percentage_values = (quake_type / len(earthquakes_df)) * 100
quake_type


# Let us display the share of event type (quakes, others...) in the data
plt.bar(quake_type.index, 
        quake_type.values, 
        color='green'
)
for index, value in enumerate(quake_type):
    plt.text(index, value + 0.5, f'{percentage_values.iloc[index]:.2f}%', ha='center', va='bottom')


# Give the takeaway of the information directly in the title
plt.title('There are mostly earthquakes populating the dataset')
plt.xlabel('Type of events')
plt.ylabel('Counts')



#######   1b  PIE chart WITH QUAKE TYPES   ########


# In the pie chart, display the figures above a certain % (eg 30%), otherwise dont display
def my_autopct(pct):
    return ('%1.1f%%' % pct) if pct > 30 else ''

pie = plt.pie(
    quake_type.values, 
    labels = None,
    autopct= my_autopct,
)
plt.legend(title='Categories', loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(pie[0],
           quake_type.index ,
           bbox_to_anchor=(1,0.5), loc="center right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)


plt.title('There are nearly ONLY earthquakes populating the dataset')
#plt.xlabel('Type of events')
#plt.ylabel('Counts')


# Add a new column with replaced values ?
type_corr_dict = {'ice quake': 'others', 'other event': 'others', "explosion": "others" }
earthquakes_df['type_corrected'] = earthquakes_df['type'].replace(type_corr_dict)
earthquakes_df.head(3)



#######    2a BAR chart WITH MAGNITUDE SCALE TYPES   ########

mag_type = earthquakes_df['magType'].value_counts()
mag_type

percentage_values = (mag_type / len(earthquakes_df)) * 100
for index, value in enumerate(mag_type):
    plt.text(index, value + 0.5, f'{percentage_values.iloc[index]:.2f}%', ha='center', va='bottom')
    
plt.bar(mag_type.index, 
        mag_type.values, 
        color='skyblue'
)

# Give the main takeaway in the title
plt.title('Events are mostly reported using the ML (Richter) magnitude ...')
plt.xlabel('Magnitude type')
plt.ylabel('Counts (and relative share in %)')



#######    2b  PIE chart WITH MAGNITUDE SCALE TYPES    ########

mag_type = earthquakes_df['magType'].value_counts()
mag_type

# Only display percentage info about 10%, otherwise dont display
def my_autopct(pct):
    return ('%1.1f%%' % pct) if pct > 10 else ''

pie_mag = plt.pie(
    mag_type.values, 
    labels = None,
    autopct =  my_autopct ,
    startangle=  90
)

plt.legend(pie_mag[0],
           mag_type.index ,
           bbox_to_anchor=(1,0.5), loc="center right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)

# Add title and labels
plt.title('Earthquakes are mostly reported using the ML (Richter) scale')
plt.xlabel('Magnitude types')
plt.ylabel('Percentage')


## So what are the 3 magnitude scales most reported in the dataset and their share ? 
percentage_values = ((mag_type / len(earthquakes_df)) * 100).round( decimals= 2)
percentage_values.name = "share"

# Collating absolute and relative share of magnitude scales
pd.concat([ mag_type , percentage_values ] , axis = 1  ) \
    .rename( columns = { "magType" : "absolute"} ) \
    .sort_values( by = "share" ,ascending= False).head(3)


# In the dataset, earthquake data are mainly reported using magnitude ML (Richter scale) with ~ 65%, followed by md and mb.
# Considering the data is from USGS, a USA geophysics study organisation, this is not surprising. 
# The UK BGS also favours this scale for moderate to low-strength earthquakes ( Mw < 6). 
# For larger events, the Richter magnitude saturates and the Mw scale (momment magnitude) is more suitable


#######     3a  BOX PLOTS  WITH MAGNITUDE AND DEPTH    ########


# Create a boxplot for the 'depth' column, without distinction
# Add vertical lines at x=25 km and x=50 km for reference purpose

plt.boxplot(earthquakes_df['depth'] , vert = False)
plt.axvline(x=25, color='red', linestyle='--', label='x=25')
plt.axvline(x=50, color='blue', linestyle='--', label='x=50')

plt.title('Earthquakes occur mostly within the first 100 km depth, with stretched outliers ?')
plt.xlabel('Earthquake depth (in km)')
plt.ylabel('')


# Is there a pattern between depth and magnitude ?
# Lets find out by distinguishing with magnitude bins
earthquakes_df['magnitude_bins'] = pd.cut(
    earthquakes_df['mag'],
    bins = range(int(earthquakes_df['mag'].min()-1), 
               int(earthquakes_df['mag'].max()) + 2)
)

earthquakes_df['magnitude_bins'] = earthquakes_df['magnitude_bins'].sort_values()


# Create a boxplot of 'depth' for each 'magnitude' bin, and add a reference vertical line at 40 km depth
plt.boxplot([earthquakes_df[earthquakes_df['magnitude_bins'] == bin]['depth'] for bin in earthquakes_df['magnitude_bins'].unique().sort_values() ],
            vert=False, labels=earthquakes_df['magnitude_bins'].unique())

plt.axvline(x=40, color='red', linestyle='--', label='x=25')

plt.title(' Across all quake categories, the median depth is less than 40km')
plt.xlabel('Earthquake depth (in km)')
plt.ylabel('Magnitude bins')

# the Seaborn version of this boxplot is more visually appealing, and the "magnitude-bins" are correctly ordered
plt.figure(figsize=(10, 6))
sns.boxplot(x='magnitude_bins', y='depth', 
            data=earthquakes_df, palette='viridis' ,
)
sns.title(' Across all quake categories, the median depth is less than 40km')
plt.xlabel('Earthquake depth (in km)')
plt.ylabel('Magnitude bins')


# In spite of regional variations, it is admitted earthquake magnitude and depths are correlated
# This dataset suggests similarly, albeit weakly. Regional variations occur, and even within one country
# It would be worth looking at only western US data (Eg Alaska, California) and Central Eastern USA 
# as tectonics within the US differ, with the West highly seismically active and the CEUSA being a "stable" region


#######     Line  PLOTS  CONNECTING dmin (min distance to the first recording station) AND rms (instrumentation signal error)   ########

# When you do NOT order your x-axis before plotting. ... 
earthquakes_df.head(3)
plt.figure(figsize=(8, 6))
plt.plot(earthquakes_df['mag'], earthquakes_df['magError'], c='skyblue', marker='.', label='Earthquake')

# Add title and labels
plt.title("A very erratic pattern")
plt.xlabel('magnitude')
plt.ylabel('error on magnitude characterisation')



# X should be ordered ideally before !
earthquakes_df = earthquakes_df.sort_values(by='mag')
# earthquakes_df.head(3)

plt.figure(figsize=(8, 6))
plt.plot(earthquakes_df['mag'], earthquakes_df['magError'], c='skyblue', marker='.', label='Earthquake')

# Add title and labels
plt.title("Ordered now ! Better?")
plt.xlabel('earthquake magnitude')
plt.ylabel('error on magnitude characterisation')


#######     SCATTER  PLOTS  CONNECTING Mag AND Mag Error   ########

# Scatter plots are not so strict about ordering your x-axis
plt.figure(figsize=(8, 6))
plt.scatter(earthquakes_df['mag'], earthquakes_df['magError'], c='skyblue', marker='.', label='Earthquake')

# Add title and labels
plt.title("Mag vs Magnitude Error shows unexpected pattern around M = 2")
plt.xlabel('Magnitude')
plt.ylabel('Error on magnitude characterisation')


# INSIGHT :
#  As earthquakes are stronger, it is easier to calculate the event magnitude, as many seismic monitoring stations have recorded it, 
# which allows to refine the measurement. Conversely, very low magnitudes are harder to calculate with precision, as maybe one station has recorded it,
# and the background noise (vibrations from human-induced activities such as train or construction) my pollute the signal.
# It is therefore expected to have larger errors at low magnitude and low errors at higher magnitude.
# What is unexpected is a cluster of (comparatively higher) errors around M2


#######     HISTOGRAM  PLOTS  TO UNDERSTAND MAGNITUDE DISTRIBUTION  ########

# We all know stonger magnitude earthquakes happen less often than weak ones. Does that stand true ?

plt.figure(figsize=(8, 6))
plt.hist(earthquakes_df['mag'], bins= 7 , color='skyblue', edgecolor='black')

# Add title and labels
plt.title('Stronger earthquakes happen less often')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')


###  Using more bins to refine the mesh may suggest other information.
##   Always change the bins in a histogram from its default value (i.e. 10)
plt.figure(figsize=(8, 6))
plt.hist(earthquakes_df['mag'], bins= 30 , color='skyblue', edgecolor='black')
plt.arrow(5 , 550,  -0.35, -120, 
          head_width = 1, 
            head_length = 50 ,
          width = 0.5
)

# Add title and labels
plt.title('Unexpected peak at magnitude M4.5 ?')
plt.xlabel('Magnitude')
plt.ylabel('Frequency of events')


# INSIGHT
# MOderate and large magnitude events (M4 and above) do happen more rarely than weak earthquakes (up to M4). The dataset confirms that in part.
# What is unexpected is a local peak of events around Mag 4.5. 
# Considering the data contains events at Mag 6 , maybe the 4.5 and M6 events cluster in space.
# Earthquake clustering is a common pattern in seismology. AFter strong events (M6 and above), smaller earthquakes occur in the region of the main shock.

# EXTRA TASK
# Looking at these particular events between M3.9 and M6 to know more ?
# earthquakes_filtered_df = earthquakes_df[(earthquakes_df['mag'] >= 3.9) & (earthquakes_df['mag'] <= 6)]


#######     SCATTER  PLOTS  CONNECTING LATITUDE AND LONGITUDE   ########

plt.figure(figsize=(8, 6))
plt.scatter(earthquakes_df['longitude'], earthquakes_df['latitude'], c='blue', marker='o', label='Earthquake')

# Add title and labels
plt.title("Lat vs Lon : this feels like a map")
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Drop rows where 'type' does not contain the word 'earthquake'
# earthquakes_df = earthquakes_df[earthquakes_df['type'].str.contains('earthquake', case=False)]

# INSIGHT
# The scatter plot shape and the range of latitude and longitude strongly suggests this is world data (although originating from USGS)
# When spotting latitude and longitude in a data, a reflex should be to think "map" and plot it.



######    PLOTTING A MAP        ########

import geopandas as gpd
from matplotlib.colors import ListedColormap

# Plot the data geocoordinates and lay it on a "world" map from the geopandas library
gdf = gpd.GeoDataFrame(earthquakes_df, geometry=gpd.points_from_xy(earthquakes_df['longitude'], earthquakes_df['latitude']))
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(figsize=(10, 6), color='lightgrey')

# Create a colormap from green to red
cmap = ListedColormap(['green', 'yellow', 'orange', 'red'])

# Plot the points with size and color based on magnitude
gdf.plot(ax=ax, cmap=cmap, marker='o', markersize=earthquakes_df['mag'] * 5, alpha=0.7, legend=True)



#######     CORRELATION  PLOTS  AS UNDERSTANDING THE RELATION   ########

#  Assess whether correlations exist in the dataset (reminder : correlation is NOT causation)
correlation_matrix_earthquakes = earthquakes_df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix_earthquakes, annot=True, cmap='coolwarm', fmt=".1f")

# Add title
plt.title('Understanding possible relationships between earthquake parameters')


# INSIGHT
# The matrix suggests stronger correlation (positive or negative) between some features, such as dmin and latitude or magnitude and latitude
# In earnest, the correlations are surprising and unexpected. One would expect correlation between Mag and Mag error, or Depth and Depth error,
# less so between longitude and mag (corr 0.6). Is this a data artefact ? Or a real thing ?


# WHAT I COULD HAVE DONE

# Time component
# The data has a time column. This calls for time distribution, and although the data is only one month worth,
# it would have interesting to look at how seismicity evolved across this last month, maybe by focusing on some
# specific regions as this data also incorporates world (strong) events.

# Special events
# The data contains mostly earthquakes but also other events such as explosions. It would have been interesting 
# to check where/when.
