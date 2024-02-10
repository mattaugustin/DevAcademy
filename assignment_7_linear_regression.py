

###  IMPORTANT : I run each section using a Jupyter interactive window (to activate in Settings/Send Selection to Interactive Window )


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
#import warnings
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics

import sweetviz as sv
import plotly.express as px



# Load dataset 
flights_df = pd.read_csv('c:\\ndsha_maps\\learning\\python\\DevAcademy\\Projects\\data\\jfk_airport_kaggle_data.csv')

# flights_df.info()
flights_df.head()


###### MY TRAIN OF THOUGHTS   #### 
# first find out the meaning of the variables and rename them (easier for later)
# plot the correlations immediately
# use your brain : looking at the variables : what would make sense to have an impact on it ?
# weather ? temperature ? windy ? nb of flights on the tarmac ? if you are delayed ?
# think time series if there is something like a time component
# if you are flight is delayed, does this have an impact on the taxi_out ? yeah, cuz u have to wait for authorisation
#
# what to use ? OHE , scaling on numerical ? Kfold 



# RENAME FEW COLUMNS TO MAKE IT MORE EXPLICIT

flights_df = flights_df.rename(
    columns=
    {
        "DAY_OF_MONTH"      : "day" ,
        "DAY_OF_WEEK"       : "weekday",
        "TAIL_NUM"          : "plane_id" ,
        "DEST"              : "destination" ,
        "CRS_ELAPSED_TIME"  : "sched_flight_duration" ,
        "CRS_DEP_M"         : "sched_dep_time"  , 
        "DEP_TIME_M"        : "actual_dep_time" ,
        "CRS_ARR_M"         : "sched_arr_time"  ,
        "sch_dep"           : "nb_flights_departing" ,
        "sch_arr"           : "nb_flights_arriving"   
    }
)


# What is Taxi-out Time ?
# The Taxi-Out Time is computed as the duration between gate out time and take off (wheels off) time. 
# A system value is obtained by averaging these durations over a period of time.


###### CREATE A SUMMARIZE FUNCTION DOING ".DESCRIBE()" & ".INFO" SIMULTANEOUSLY

def profile_data(data):
    
    """Panda Profiling Function
    
    Args:
        data (DataFrame): A data frame to profile
        
    Returns:
        DataFrame : a data frame with profiled data
    
    """
        
    return pd.concat(
        [
            pd.Series(data.dtypes , name = "Dtype") ,
            # counts
            pd.Series( data.count()        , name = "Count") ,
            pd.Series( data.isnull().sum() , name = "NA Count") ,
            pd.Series( data.nunique()      , name = "Count Unique") ,
            
            # Stats
            pd.Series( data.min()           , name = "Min" ) ,
            pd.Series( data.max()           , name = "Max" ) ,
            pd.Series( data.mean()          , name = "Mean") ,
            pd.Series( data.median()        , name = "Median") ,
            pd.Series( data.mode().iloc[0]  , name = "Mode") ,
        ] ,
        axis = 1
    )
    
# SHOW ALL COLUMNS AND NOT "..."
pd.set_option("display.max_columns", None)

profile_flights_df = profile_data( flights_df )

profile_flights_df


##### some questions we wanna ask #####

# Does weather condition really play a role ?
# Are there some special companies or flights more often doing taxing than others ? maybe some destinations more linked to taxing ?
# MEASURE scheduled vs actual departure time, as well as sched VS actual ARRIVAL time
# CREATE : time component to have a time series and see whether a certain day was worse than others

## Sweetviz report with target

report = sv.analyze(
    flights_df ,
    target_feat = "TAXI_OUT"
)

#?report.show_html

report.show_html(
    filepath = "flights_df_report.html"
)


# FIRST INSIGHTS
# destination , duration and distance all STRONGLY MUTUALLY correlated >> drop one or two columns and pick one for the model owing to collinearity
# Weather conditions have high degrees of correlation (wind and temp for instance are + correlated, pressure and temp are - correlated)
# Quite some outliers on "departure delay" : 1276 being max aka ~ 22h delay (exceptional event in the sweet viz report)

#       About TAXI_OUT :
# "flight carrier"    is   correlated. Are some companies more likely to experience TAXI_OUT than others ?
# "Destination"       is   correalted +    : maybe some destinations are more likely to have TAXI_OUT
# Wind, Dew  & Condition overall           : some positive correlation to TAXI_OUT
# "MONTH"             is weakly correlated :

# There is a hidden time component via the dep_time_m, crs_arr_m, as well as month/day_of_month/weekday.
# Lets re-engineer the table to extract the time series over last 3 months, possibly segmented by flight carrier (9 of them)


############ FEATURE ENGINEERING  ###########

flights_df["YEAR"] = flights_df['MONTH'].apply(lambda MONTH: 2019 if MONTH in [11, 12] else 2020)
flights_df = flights_df[['YEAR'] + [col for col in flights_df.columns if col != 'YEAR']]


flights_df

# CREATE "FLIGHT DATE" column
# flights_df["flight_date"] = pd.to_datetime(flights_df[["YEAR","MONTH","day"]])


# CREATE "FLIGHT_TIME" column 

# Functions to time components from the columns "CRS_DEP_M", "DEP_TIME_M", "CRS_ARR_M",  expressed in minute index of the day (ex : 109 min = 1h49 )

# get hour and minute from the relevant columns
def get_modulo(row, column_name):
    numerand, remainder = divmod(row[column_name], 60)
    return numerand, remainder

# correct for hour and day when hour = 24 (midnight )
def correct_hour_and_day(row):
    hour_corr = row['hour']
    day_corr = row['day']
    
    # Check if "hour" is equal to 24
    if row['hour'] == 24:
        hour_corr = 0
        day_corr += 1
    
    return pd.Series([hour_corr, day_corr], index=['hour_corr', 'day_corr'])



# Wrapping function to extract date time from any selected column
def apply_modulo_to_column(data_df, column_name, formatted_column_name):

    df = data_df.copy()
    
    df[["hour", "minute"]] = df.apply(get_modulo, axis=1, args=(column_name,), result_type='expand')
    
    
    # Apply the function to create new corrected columns
    df[['hour_corr', 'day_corr']] = df.apply(correct_hour_and_day, axis=1)

    # Create a datetime64 column
    df[formatted_column_name]    = pd.to_datetime(df["YEAR"].astype(str) + '-' + 
                                                    df['MONTH'].astype(str)+ '-' +
                                                    df['day_corr'].astype(str) + ' ' +
                                                    df['hour_corr'].astype(str)+ ':' +
                                                    df['minute'].astype(str) ,
                                                    format='%Y-%m-%d %H:%M')

    df = df.drop( ["day_corr","hour","hour_corr","minute"], axis = 1 )

    # Reorder the newly created column near its original 
    col_index = df.columns.get_loc(column_name)
    col_before_column_name    = df.columns[:col_index].tolist()
    columns_after_column_name = df.columns[col_index + 1:-1].tolist()
    
    df = df[ col_before_column_name + [column_name] + [formatted_column_name] + columns_after_column_name  ]
    
    return df


# Apply the function to the DataFrame for the selected columns
flights_df = apply_modulo_to_column( flights_df , "sched_dep_time"  , "sched_dep_time_formatted" )
flights_df = apply_modulo_to_column( flights_df , "actual_dep_time" , "actual_dep_time_formatted" )
flights_df = apply_modulo_to_column( flights_df , "sched_arr_time"  , "sched_arr_time_formatted" )

flights_df



####  HANDLING TIME SERIES #####

### not segmenting
flights_plot_df = flights_df \
    .set_index('sched_dep_time_formatted') \
    .resample('D', kind = 'period') \
    .agg(np.mean) \
    .reset_index() \
    .set_index("sched_dep_time_formatted")
    
## Segmenting across flight companies ####
# flights_df \
#     .set_index('sched_dep_time_formatted') \
#     .groupby('OP_UNIQUE_CARRIER') \
#     .resample('D', kind = 'period') \
#     .agg(np.mean) \
#     .unstack("OP_UNIQUE_CARRIER") \
#     .reset_index() \
#     .assign(order_date = lambda x: x['sched_dep_time_formatted'].dt.to_period()) \
#     .set_index("sched_dep_time_formatted")



######################################
######   DATA VISUALISATIONS  ########
######################################



######  PLOT DISTRIBUTION OF DISTANCE  #########

#bins = np.arange(0, 5001, 100)
plt.figure(figsize=(8, 6))
sns.displot( flights_df["DISTANCE"],
            #bins = 30 ,
            binwidth = 100 ,
            stat = "percent"          
)
plt.xticks(np.arange(0, 5000 + 1, 1000))
plt.gca().yaxis.set_major_formatter(PercentFormatter(100))

# Add title and labels
plt.title('Flight Distance distribution')
plt.xlabel('Distance (in km)')
plt.ylabel('Percentage')

### we can remove the flights above 2700 km as there are nearly none ?



######  PLOT DISTRIBUTION OF DESTINATION  #########

destination_df = pd.DataFrame(
    {
        "counts" : flights_df["destination"].value_counts() ,
        "percentage" : (flights_df['destination'].value_counts(normalize=True) * 100).round(2)
    } 
).reset_index() \
.rename(
    columns= {"index" : "destination"}
)

# destination_df.query("percentage > 1")["percentage"].sum()  # 88% of all dataset if all destinations accounting for more than 1% of dataset
# destination_df.head(20)["percentage"].sum()  # 63%% of all dataset if top 10 destination

## Categorical plot of Destination for the top 20
destination_df = flights_df \
.value_counts("destination", normalize= True) \
.mul(100 ) \
.round(2) \
.rename('percent') \
.to_frame()

destination_taxi_df = flights_df[["destination","TAXI_OUT"]] \
    .groupby("destination") \
    .mean() \
    .round(2) \
    .sort_values( by = "TAXI_OUT", ascending= False)


destination_plot_df = destination_df \
    .merge(
        destination_taxi_df ,
        how = "left" ,
        left_index = True ,
        right_index = True ,
    ) \
    .sort_values( by = "percent" , ascending = False) \
    .reset_index()


plt.figure(figsize=(10, 10))
# Plot bar chart for "percent" variable
sns.barplot(data= destination_plot_df.head(20), y='destination', x='percent', color='skyblue')

# Overlay line plot for "TAXI_OUT" variable
ax2 = plt.gca().twiny()  # Create secondary x-axis
sns.lineplot(data= destination_plot_df.head(20), 
             y='destination', 
             x='TAXI_OUT', 
             color='red', 
             marker='o', 
             ax=ax2 , 
             orient = "y"
)
 
# Set labels and title
plt.xlabel('Percent', color='blue')
#ax2.set_ylim(destination_plot_df['TAXI_OUT'].min() - 2, destination_plot_df['TAXI_OUT'].max() + 2)
ax2.set_xlabel('TAXI_OUT', color='red')
plt.ylabel('Destination')
plt.title('Destination vs Percent and TAXI_OUT')


######  PLOT DISTRIBUTION OF WEATHER CONDITIONS


weather_df = flights_df \
.value_counts("Condition", normalize= True) \
.mul(100 ) \
.round(2) \
.rename('percent') \
.to_frame()

weather_taxi_df = flights_df[["Condition","TAXI_OUT"]] \
    .groupby("Condition") \
    .mean() \
    .round(2) \
    .sort_values( by = "TAXI_OUT", ascending= False)


weather_plot_df = weather_df \
    .merge(
        weather_taxi_df ,
        how = "left" ,
        left_index = True ,
        right_index = True ,
    ) \
    .sort_values( by = "percent" , ascending = False) \
    .reset_index()


###  Plot "Condition" distribution in the dataset and its relation to "TAXI_OUT"
plt.figure(figsize=(10, 10))
sns.barplot(data= weather_plot_df.head(20), y='Condition', x='percent', color='skyblue')

# Overlay line plot for "TAXI_OUT" variable
ax2 = plt.gca().twiny()  # Create secondary x-axis
sns.lineplot(data= weather_plot_df.head(20), 
             y='Condition', 
             x='TAXI_OUT', 
             color='red', 
             marker='o', 
             ax=ax2 , 
             orient = "y"
)
 
# Set labels and title
plt.xlabel('Percent', color='blue')
#ax2.set_ylim(destination_plot_df['TAXI_OUT'].min() - 2, destination_plot_df['TAXI_OUT'].max() + 2)
ax2.set_xlabel('TAXI_OUT', color='red')
plt.ylabel('Condition')
plt.title('Condition vs Percent and TAXI_OUT')


#### PLOT DISTRIBUTION OF FLIGHT CARRIER


carrier_df = flights_df \
.value_counts("OP_UNIQUE_CARRIER", normalize= True) \
.mul(100 ) \
.round(2) \
.rename('percent') \
.to_frame()

carrier_taxi_df = flights_df[["OP_UNIQUE_CARRIER","TAXI_OUT"]] \
    .groupby("OP_UNIQUE_CARRIER") \
    .mean() \
    .round(2) \
    .sort_values( by = "TAXI_OUT", ascending= False)


carrier_plot_df = carrier_df \
    .merge(
        carrier_taxi_df ,
        how = "left" ,
        left_index = True ,
        right_index = True ,
    ) \
    .sort_values( by = "percent" , ascending = False) \
    .reset_index()

carrier_plot_df

###  Plot "OP_UNIQUE_CARRIER" distribution in the dataset and its relation to "TAXI_OUT"
plt.figure(figsize=(10, 10))
sns.barplot(data= carrier_plot_df.head(20), y='OP_UNIQUE_CARRIER', x='percent', color='skyblue')

# Overlay line plot for "TAXI_OUT" variable
ax2 = plt.gca().twiny()  # Create secondary x-axis
sns.lineplot(data= carrier_plot_df.head(20), 
             y='OP_UNIQUE_CARRIER', 
             x='TAXI_OUT', 
             color='red', 
             marker='o', 
             ax=ax2 , 
             orient = "y"
)
 
plt.xlabel('Percent', color='blue')
#ax2.set_ylim(destination_plot_df['TAXI_OUT'].min() - 2, destination_plot_df['TAXI_OUT'].max() + 2)
ax2.set_xlabel('TAXI_OUT', color='red')
plt.ylabel('OP_UNIQUE_CARRIER')
plt.title('OP_UNIQUE_CARRIER vs Percent and TAXI_OUT')




####   PLOT DISTRIBUTION OF DELAY  #########

plt.figure(figsize=(8, 6))
fig, ax = plt.subplots()
sns.histplot(data=flights_df, x="DEP_DELAY" , 
            stat = "percent" ,
            binwidth = 20, 
            binrange = [-20, 500] ,
            ax = ax         
)
ax.set_xlim(-25,1300)
ax.axvline(x = 60 , ymin = 0 , ymax = 0.5, color = "red" )
ax.text(70, 3.75, "1 h" , color = "red")
#ax.set_ylim(0,0.001)
plt.title('Most of the delay is contained within 60min ')
plt.xlabel('Delay at departure (in min)')
plt.ylabel('Percentage')

# plt.xticks(np.arange(0, 5000 + 1, 1000))
# plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
# max( flights_df["DEP_DELAY"] )


## ANY OUTLIERS IN DEP_DELAY ?
## >> YES !!! (hence the limit on x-axis up to 90min = 1h30)

flights_df["DEP_DELAY"].describe( percentiles= [0.25, 0.5 , 0.75 , 0.86 ,0.9 , 0.95 , 0.99] )

plt.figure(figsize=(10, 6))
ax = sns.boxplot( x='DEP_DELAY', 
            data=flights_df, palette='viridis' ,
)
ax.axvline(x = 14, ymin = 0.25, ymax = 0.75 , color = "darkgreen")
ax.text(16, 0.30, "25 min" , color = "darkgreen")
ax.axvline(x = 30, ymin = 0.25, ymax = 0.75 , color = "blue")
ax.text(32, 0.30, "30 min" , color = "blue")
ax.axvline(x = 60, ymin = 0.25, ymax = 0.75 , color = "red")
ax.text(62, 0.30, "60 min" , color = "red")
ax.set_xlim(-25,90)
ax.set_xticks(np.arange(-20, 100 + 1, 10))

ax.set_title(' ~ 86% of flights take off with less than 25min delay')
ax.set_xlabel('Delay at departure (in min)')



#######     CORRELATION  PLOTS  AS UNDERSTANDING THE RELATION   ########


#  Assess whether correlations exist in the dataset (reminder : correlation is NOT causation)
correlation_matrix_flights = flights_df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix_flights, 
            annot=True,  
            fmt=".2f",
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            # cmap='coolwarm',
            square=True
)
plt.title('Understanding possible relationships between flight take-off parameters')

# Alternatively for interaction purpose
px.imshow(
    correlation_matrix_flights ,
    width = 1000 ,
    height = 1000 ,
    template = "plotly_dark" ,
    title = "Taxi_out insights : Feature Correlations"
)


### USING SOME FUNCTIONS I CREATED IN A PREVIOUS PROJECT
from my_pandas_extensions.timeseries import summarize_by_time
from my_pandas_extensions.forecasting import plot_forecast_together


flights_df_time = flights_df \
    .summarize_by_time(
        date_column  = "sched_dep_time_formatted" ,
        value_column = "TAXI_OUT"   ,
        groups       = "OP_UNIQUE_CARRIER"   ,
        agg_func     = np.mean ,
        wide_format  = False ,
        kind = "period"
)

flights_df_time.reset_index() \
    .plot_forecast_together(
        date_column  = "sched_dep_time_formatted" ,
        value_column = "TAXI_OUT" ,
        id_column = "OP_UNIQUE_CARRIER" ,
        facet_ncol    = 3   ,
        date_labels='%D',
        date_breaks = "2 months" ,
        figure_size = (24,20) ,
        x_lab = "Date" ,
        wspace = 3 ,
        y_lab = "Taxi out duration (min)"
)




##############################################
########   REGRESSION ANALYSIS  ##############
##############################################



### DEFINE VECTORS
### Split data
### Feature scaling
### linear regression
### Kfold cross is with Ridge ?
# What is the order of Scaling/splitting when you have a mix of numerical/categorical


# X = flights_df.iloc[:,3].values.reshape(-1,1)
# X = flights_df[["destination","DEP_DELAY","Condition","nb_flights_departing","nb_flights_arriving"]].values
numerical_var = flights_df[["DEP_DELAY","nb_flights_departing","nb_flights_arriving"]]


# Standardize the features
scaler    = StandardScaler()
# X_scaled  = scaler.fit_transform(numerical_var)

### Handle categorical variables
categorical_var     = flights_df[["destination","Condition"]]
label_encoder       = LabelEncoder()
# Iterate over each categorical column and encode it
for column in categorical_var.columns:
    if categorical_var[column].dtype == 'object':
        categorical_var[column] = label_encoder.fit_transform(categorical_var[column])

categorical_var.values

X = pd.concat( [numerical_var , categorical_var] , axis=1)
X_scaled = scaler.fit_transform(X)

Y  = flights_df["TAXI_OUT"].values.reshape(-1,1)
Y

# NB Perform categorical encoding on the labels
# with Keras
# y_categorical = to_categorical(y_encoded)
# With Pandas
categorical_var_dum = pd.get_dummies(  flights_df[["destination", "Condition"]]   ) 


# Split the data into training and testing sets
X_train, X_test, y_train, y_test  =  train_test_split(X_scaled, Y, test_size=0.2, random_state=42)


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


## prediction
y_pred = lin_reg.predict(X_test)


meanAbError  = metrics.mean_absolute_error(y_test, y_pred)
meanSqError  = metrics.mean_squared_error(y_test, y_pred)
rootMeanSqrr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# print('R squared:{:.2f}'.format(lin_reg.score(X_test, y_test)))
# print('Mean Absolute Error:', meanAbError)
# print('Mean Square Error:', meanSqError)
# print('Root Mean Square Error:', rootMeanSqrr)

pd.DataFrame(
    { 
     "R squared" : [lin_reg.score(X_test, y_test)] ,
     "Mean Absolute Error" : [meanAbError] ,
     "Mean Square Error"   : [meanSqError] ,
     'Root Mean Square Error' : [rootMeanSqrr]
    }
).T


### TRYING NOW WITH One Hot Encoding instead of Label Encoder
scaler    = StandardScaler()

categorical_var_dum = pd.get_dummies(  flights_df[["destination", "Condition"]]   ) 

X = pd.concat( [numerical_var , categorical_var_dum] , axis=1)
X_scaled = scaler.fit_transform(X)

Y  = flights_df["TAXI_OUT"].values.reshape(-1,1)
Y


## data splitting
X_train, X_test, y_train, y_test = train_test_split( X_scaled , Y , test_size = 20 , random_state = 32)


## linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)



lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


## prediction
y_pred = lin_reg.predict(X_test)


meanAbError  = metrics.mean_absolute_error(y_test, y_pred)
meanSqError  = metrics.mean_squared_error(y_test, y_pred)
rootMeanSqrr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# print('R squared:{:.2f}'.format(lin_reg.score(X_test, y_test)))
# print('Mean Absolute Error:', meanAbError)
# print('Mean Square Error:', meanSqError)
# print('Root Mean Square Error:', rootMeanSqrr)

pd.DataFrame(
    { 
     "R squared" : [lin_reg.score(X_test, y_test)] ,
     "Mean Absolute Error" : [meanAbError] ,
     "Mean Square Error"   : [meanSqError] ,
     'Root Mean Square Error' : [rootMeanSqrr]
    }
).T



### WHAT I SHOULD HAVE DONE
# 
#  Verify assumptions for linear regression as these might not be satisfied everywhere ?

## WHAT I COULD HAVE DONE
#
# add more columns based on the correlation scores at the very beginning from the sweetviz report
# USe lasso and ridge, as well as LassoCV and RidgeCV
# Ensure you have the right scaling on both Train and Test
# Use a different algorithm for regression (Pycaret reg ? )