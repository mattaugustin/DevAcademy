import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

import sweetviz as sv

############### CUSTOM MADE FUNCTIONS   ##########

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
            pd.Series( data.duplicated().sum()   , name = "Count Duplicate"),
            
            # Stats
            pd.Series( data.min( skipna = True)   , name = "Min" ) ,
            pd.Series( data.max( skipna = True ) , name = "Max" ) ,
            pd.Series( data.mean()          , name = "Mean") ,
            pd.Series( data.median()        , name = "Median") ,
            pd.Series( data.mode().iloc[0]  , name = "Mode") ,
        ] ,
        axis = 1
    )



### PREPARE CATEGORICAL VS TARGET ###

def prepare_table_cat_and_target(
    data ,
    category_col ,
    target_col ,
    sort_order = False
):
    
    
    # category_col = "Sex"
    # target_col = "Survived"
    
    # data = titanic_df.copy()
    # sort_order = False
    
   # data[category_col] = data[category_col].astype("category")
    #data[target_col] = data[target_col].astype("category")
    
    data_category_df = data \
        .value_counts(category_col, normalize= True) \
        .mul(100 ) \
        .round(2) \
        .rename('Percentage_dataset') \
        .to_frame()
    
    data_target_df = data[[category_col,target_col]] \
        .groupby(category_col) \
        .mean() \
        .mul(100) \
        .round(2) \
        .sort_values( by = target_col , ascending= sort_order )
     
    data_catarget_df = data_category_df \
        .merge(
            data_target_df ,
            how = "left" ,
            left_index = True ,
            right_index = True ,
        ) \
        .sort_values( by = category_col , ascending = sort_order) \
        .reset_index()

    return data_catarget_df




### PLOT CATEGORICAL VS TARGET ###

def plot_cat_and_target(
    data ,
    category_col ,
    target_col ,
    figure_size = (10, 10) ,
    **kwargs
):

    data[category_col] = data[category_col].astype("category")
    #data = test_df.copy()
    #sort_order = False
    
    #data["Pclass"] = data["Pclass"].astype("category") 
    #category_col = "Pclass"
    #target_col   = "Survived"
    #figure_size = (10, 10)

    plt.figure(figsize= figure_size)
    # Plot bar chart for "percent" variable
    sns.barplot(data= data, y= category_col, x='Percentage_dataset', orient= "h" , **kwargs)

    # Overlay line plot for "target" variable
    ax2 = plt.gca().twiny()  
    sns.pointplot(
        data= data , 
        y = category_col,  
        x = target_col,    
        color='red', 
        linestyles= "-" ,
        markers= "o" ,
        ax=ax2 ,
        join= True ,
    )
    
    # Set labels and title, axes ...
    plt.xlabel('Percent', color='blue')
    ax2.set_xlabel(target_col, color='red')
    ax2.tick_params(axis='x', colors='red')
    
    plt.gca().xaxis.set_major_formatter(PercentFormatter(100))    
    plt.ylabel(category_col)
    plt.title(f'{category_col} vs {target_col} in proportion')
    


#######     PROJECT START       #######


titanic_df = pd.read_csv("c:\\ndsha_maps\\learning\\python\\DevAcademy\\Projects\\Data\\titanic\\train.csv")


# PassengerID : 

# Survivded : 0 = NO , 1 = YES

# PCLAS : passenger class : 1st / 2nd / 3rd

# SEX : Male or Female

# AGE : in years

# sibsp : nb of siblings / spouses aboard

# parch : nb of parents / children aboard

# TICKET : ticket number

# FARE : passenger fare in USD

# CABIN : Cabin number

# EMBARKED :  Port of Embarkation : C Cherbourg Q Queenstown  S Southhampton

# PassengerID and Cabin# probably wont be relevant if we cannot link the latter to a cabin location map and place of damage



profile_data(titanic_df)

# titanic_df.Age.isna().value_counts( normalize= True ).round( decimals=2)
# titanic_df.Cabin.isna().value_counts( normalize= True ).round( decimals=3)

##  Cabin       is  missing for about 77% of data
##  Age         is  missing _________ 20% of data
##  Embarked    is  missing _________0.2% of data


# Before the report, some data probably may not be useful : 
# PassengerID : index, repeating with real titanic_df.index
# Cabin (too many missing)
# Passenger Name (likely high cardinality) ??



report = sv.analyze(
    titanic_df ,
    target_feat = "Survived"
)

report.show_html(
    filepath = "titanic_df_report.html"
)


# Ovbious distribution to look at are :
#  %of survived vs not survived <> some imbalance
#  Survived vs UnSurvived :
#   by SEX
#   by CLASS (Social Economic Status)
#   by both SEX and CLASS (got it but wrong units : i want percentage)
#  
#   by Age :<> differentiating between less than 16-18 and the rest ? aka kids vs the rest
#   by Age and Class (if you are on lower class, do you die even if you are a kid ?)
#   Age and Class : Separate Adults and Class and Kids and Class




#############################################
####                                     ####
####    EXPLORATORY DATA ANALYSIS        ####
####                                     ####
#############################################


####    TARGET PROPORTION   ####
#### ---------------------  ####

# Calculate and plot the proportion of passengers who survived and who did not survive
survival_counts = (titanic_df['Survived'].value_counts(normalize=True) * 100 ).round(decimals=2)
survival_counts_abs = titanic_df['Survived'].value_counts()

survival_counts_df = pd.DataFrame({
    'Survived'  : survival_counts.index, 
    'Percentage': survival_counts.values , 
    'Count'     : survival_counts_abs.values ,
                                    })


plt.figure(figsize=(7 ,6))
ax = sns.barplot(
    data = survival_counts_df, 
    y    ='Count' , x='Survived', 
    orient='vertical', palette = ["red", "green"]
)


plt.ylabel('Count', color='black')
plt.xlabel('Outcome')
ax.set_xticklabels(['Died', 'Survived'])
plt.title('How many survived ?')

for i, row in survival_counts_df.iterrows():
    percentage = row["Percentage"]
    plt.text( i , row["Count"] , f'{percentage:.2f}%', ha='center', va='bottom' , color = "black")

# plt.savefig('c:\\DevAcademy\\Projects\\06_Classification\\pictures\\survival_chance.jpg', dpi = 150)


# print(list( enumerate(survival_counts_df) ))
# print(list(survival_counts_df.iterrows()))



# sns.catplot(
#     data=titanic_df, x="Pclass", y="Survived", hue="Sex",
#     palette={"male": "g", "female": "m"},
#     markers=["^", "o"], linestyles=["-", "--"],
#     kind="point"
# )



####    GENDER PROPORTION   ####
#### ---------------------  ####

# Count and assess proportion of survivors per gender
counts = titanic_df["Sex"].value_counts().sort_index()
percentages = counts / counts.sum() * 100


plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_df, y='Sex' , orient = "h" , 
             order = ["female","male"] , palette= ["darkorange","lightblue"]
)

plt.title('Gender demographic on board (pre-disaster)')
plt.xlabel('Number of people')
plt.ylabel('Gender')

# Insert percentage labels
total = len(titanic_df)
for i, count in enumerate(counts):
    percentage = percentages[i]
    plt.text(count - 0.1*count, i, f'{percentage:.2f}%', ha='center', va='bottom' , fontsize = 12 , color = "black")

# Save plot
# plt.savefig('c:\\DevAcademy\\Projects\\06_Classification\\pictures\\gender_demographic.jpg', dpi = 150)



## Gender proportion upon disaster (survivors and loss)

ax = sns.countplot(data=titanic_df, x='Sex' , hue = "Survived", palette = ["red","green"])
ax.set_axisbelow(True)
ax.grid(color = "gray", linestyle = "dashed" , axis = "y")
plt.title('Most men died and most women survived')
plt.xlabel('Gender')
plt.ylabel('Number')


####  GENDER vs TARGET   ####

titanic_sex_survived_df = prepare_table_cat_and_target(
    data  = titanic_df , 
    category_col =  "Sex" ,
    target_col   =  "Survived" ,
    sort_order = True
)

## Original category distribution pre-disaster AND proportion of category among survivors after disaster
plot_cat_and_target(
    data  = titanic_sex_survived_df ,
    category_col = "Sex"      ,
    target_col   = "Survived" ,
    figure_size = (10, 10) ,
    palette = ["darkorange","lightblue"]
)
for i, row in titanic_sex_survived_df.iterrows():
    percentage = row["Survived"]
    plt.text(percentage , i + 0.1, f'{percentage:.2f}%', ha='center', va='bottom' , color = "darkred")

# How to read : Among all survivors, women make up 74% whilst men make about 19%

# Save figure
# plt.savefig('c:\\DevAcademy\\Projects\\06_Classification\\pictures\\gender_demographic_survived.jpg', dpi = 150)


# print( list (titanic_sex_survived_df.iterrows()))
# print(list( enumerate(counts)  ))
# print(list( enumerate(test_df)  ))



####    CLASS vs SURVIVED       ####
####   -----------------------  ####


counts = titanic_df["Pclass"].value_counts().sort_index()
percentages = counts / counts.sum() * 100

# Plot the distribution using seaborn
plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_df.sort_values("Pclass" , ascending = False), y='Pclass' , orient = "h" , 
             order = [1, 2 , 3]
)

plt.title('Socio-demographic on board (pre-disaster)')
plt.xlabel('Number of passengers')
plt.ylabel('Passenger class')

# Annotate bars with percentages
total = len(titanic_df)
for i, count in enumerate(counts):
    percentage = percentages[i+1]
    plt.text(count - 30 , i, f'{percentage:.2f}%', ha='center', va='bottom')


# Save figure
plt.savefig('c:\\ndsha_maps\\learning\\python\\DevAcademy\\Projects\\06_Classification\\pictures\\socio_demographic.jpg', dpi = 150)



####  CLASS vs TARGET   ####

test_df = prepare_table_cat_and_target(
    data         = titanic_df ,
    category_col = "Pclass" ,
    target_col   = "Survived" ,
    sort_order   = False
)
# change otherwise plot will not
# test_df["Pclass"] = pd.Series( test_df["Pclass"]  , dtype = "category")

plot_cat_and_target(
    data         = test_df ,
    category_col = "Pclass" ,
    target_col   = "Survived" ,
    palette = ["green","blue","red"]
)

# Save figure
plt.savefig('c:\\python\\DevAcademy\\Projects\\06_Classification\\pictures\\socio_demographic_survived.jpg', dpi = 150)



#    CLASS vs GENDER vs SURVIVED
# ----------------------------

# proportion_sex_survived_df = titanic_df[["Sex","Survived"]] \
#     .groupby("Sex") \
#     .sum()


# sex_class_target_df = titanic_df[["Sex","Pclass","Survived"]] \
#     .groupby(["Sex","Pclass"]) \
#     .count() \
#     .merge(
#         right = proportion_sex_survived_df ,
#         how = "left" ,
#         left_index = True,
#         right_index = True
#     ) \
#     .assign(
#         prop_category  = lambda x:(x["Survived_x"]/x["Survived_y"]).round(decimals = 3) ,
#         prop_all_surv  = lambda x:(x["Survived_x"]/ x["Survived_x"].sum() ).round(decimals = 3) ,
#     ) \
#     .rename(
#         columns= {
#             "Survived_x" : "survived_category" , 
#             "Survived_y" : "survived_sex"
#         }
#     ).reset_index()

# ## Plot
# sns.pointplot(
#     data=sex_class_target_df, x="Pclass", y="prop_category", hue="Sex",
#     palette={"male": "g", "female": "m"},
#     markers=["^", "o"], linestyles=["-", "--"],
# ) 



###  SURVIVORS AND PROPORTION IN CATEGORY    ###
### -------------------------------------    ###

proportion_sex_class_df = titanic_df[["Sex","Pclass","Survived"]] \
    .groupby(["Sex","Pclass"]) \
    .count()

sex_class_prop_df = titanic_df[["Sex","Pclass","Survived"]] \
    .groupby(["Sex","Pclass"]) \
    .sum() \
    .merge(
        right = proportion_sex_class_df ,
        how = "left" ,
        left_index = True,
        right_index = True
    ) \
    .assign(
        prop_surv_category  = lambda x:(x["Survived_x"]/x["Survived_y"]).round(decimals = 3) ,
    ) \
    .rename(
        columns= {
            "Survived_x" : "survived_category" , 
            "Survived_y" : "total_category"
        }
    ).reset_index()


## Plot based on both categories
## For a given combined category (Class and Gender, aka Female 1st Class), 
#      whats the proportion of survivors in this category ?
# ex : 91 females out 94 survived in the first class (female) category

sns.pointplot(
    data    = sex_class_prop_df, x="Pclass", y="prop_surv_category", hue="Sex",
    palette ={"male": "blue", "female": "darkorange"},
    markers =["^", "o"], linestyles=["-", "--"]
)
plt.title('Socio-demographic & Gender who survived')
plt.xlabel('Passenger class')
plt.ylabel('Proportion of survivors in the category Class|Gender')

plt.savefig('c:\\ndsha_maps\\learning\\python\\DevAcademy\\Projects\\06_Classification\\pictures\\socio_demographic_gender_survived.jpg', dpi = 150)



####    AGE vs CLASS    ####
#### -----------------  ####

titanic_df["Age_bins"] = pd.cut( titanic_df["Age"] , bins = np.arange(0,100,10) )

titanic_df["Age_bins_2"] = pd.cut( 
    titanic_df["Age"] , 
    bins = [0,5,10,15,20, 30, 40, 50 , 60, 70 , 80 , 90] , 
    right= False ,
    include_lowest= True
)

sns.histplot(
    data  =  titanic_df ,
    x     =  "Age" ,
    stat = "percent" ,
    binwidth  = 5
)


# Different bins to look at different populations : e.g. kids up to 3y old ? Adult age starting from 18y old ?
#newBins = [0,3,10,15,18,20,30,40,50,60,70,80,90]
newBins = [0,5,10,15,20,30,40,50,60,70,80,90]

ax = sns.histplot(titanic_df["Age"].values,bins=newBins,kde=False , stat = "percent"  )
plt.axvline(x=18 , color = "red", lw = 2 , ls = "--" )  # 18y old separation line
plt.title('Mostly young adults onboard')
plt.xlabel('Passenger age on board')
plt.ylabel('Percentage of the dataset')

# Focusing on some parts of the histogram ?
# ax.set_xlim(0, 20)

plt.savefig('c:\\ndsha_maps\\learning\\python\\DevAcademy\\Projects\\06_Classification\\pictures\\age_passengers_onboard.jpg', dpi = 150)



####    SURVIVORS AND PROPORTION IN CATEGORY    ####
####  ----------------------------------------  ####


proportion_kid_class_df = titanic_df[["Age_bins_2","Pclass","Survived"]] \
    .groupby(["Age_bins_2","Pclass"]) \
    .count()

kid_class_prop_df = titanic_df[["Age_bins_2","Pclass","Survived"]] \
    .groupby(["Age_bins_2","Pclass"]) \
    .sum() \
    .merge(
        right = proportion_kid_class_df ,
        how = "left" ,
        left_index = True,
        right_index = True
    ) \
    .assign(
        prop_surv_category  = lambda x:(x["Survived_x"]/x["Survived_y"] * 100 ).round(decimals = 2) ,
    ) \
    .rename(
        columns= {
            "Survived_x" : "survived_category" , 
            "Survived_y" : "total_category"
        }
    ).reset_index()


sns.set_palette("bright")
my_palette = ["green", "blue" , "red"]
g = sns.FacetGrid(
    kid_class_prop_df, 
    row="Pclass" , hue = "Pclass" ,
    height = 3 , aspect = 3 , 
    palette =  my_palette
) 
sns.set_style(
    "whitegrid" , 
    { "grid.linestyle" : "--" ,
      "grid.color" : "black"}
)
g.map_dataframe(sns.pointplot, y = "prop_surv_category", x = "Age_bins_2" )
g.map(plt.axhline, y=0, ls='--', c='red')
plt.grid(True , axis = "y")
plt.gca().yaxis.set_major_formatter(PercentFormatter(100))    
g.set_ylabels("Survivors proportion by age & class")
g.set_xlabels("Age bins")

# How to read the graph ? In 3rd class, for the age range 5-10y old, only 36% of children survived

plt.savefig('c:\\ndsha_maps\\learning\\python\\DevAcademy\\Projects\\06_Classification\\pictures\\socio_demographic_age_survived.jpg', dpi = 150)




####    FARE CLASS SURVIVED
#### -------------------------

####    TOP10 OF WHO PAID A LOT ?    #####

#titanic_df.sort_values("Embarked", ascending = True).head(10)

(
titanic_df.
sort_values("Pclass")
#.query("Fare <= 100")
.pipe( 
    ( sns.catplot, "data") , 
     x      = "Fare"  , y    = "Pclass" , 
     orient = "h"     , kind = "box"    
    )
)
plt.title('Who paid $ 500 for a ticket ?')
plt.xlabel('Passenger ticket fare ($)')
plt.ylabel('Passenger class')


plt.savefig('c:\\ndsha_maps\\learning\\python\\DevAcademy\\Projects\\06_Classification\\pictures\\fare_class.jpg', dpi = 150)


ax = sns.catplot( 
     data =  titanic_df.sort_values("Pclass") ,
     x      = "Fare"  , y    = "Pclass" , 
     orient = "h"     , kind = "box"    
)
plt.xlim([0, 100])
plt.title('Focusing on tickets below the $ 100 mark ... ')
plt.xlabel('Passenger ticket fare ($)')
plt.ylabel('Passenger class')


plt.savefig('c:\\ndsha_maps\\learning\\python\\DevAcademy\\Projects\\06_Classification\\pictures\\fare_class_zoom.jpg', dpi = 150)


####    HOW TO FILL UP THE NAs for the AGE  ####

# average or median age by class ?


####    RELATION BETWEEN AGE, FARE and CLASS ?  ####
#### -----------------------------------------  ####

# Question : Based on your class and how much money you paid, could I find out your age ?

plt.figure(figsize= (7,7))
ax = sns.scatterplot(
    data = titanic_df , #.query("Pclass == 3") ,
    x = "Age" ,
    y = "Fare" ,
    hue = "Pclass" ,
    palette = ["green","blue","red"]
)
ax.set_xlim(0, 85)
ax.set_ylim(0, 300)


plt.savefig('c:\\ndsha_maps\\learning\\python\\DevAcademy\\Projects\\06_Classification\\pictures\\fare_class_age.jpg', dpi = 150)


# Fill NaN values in the "age" column with median age per passenger class

median_age_by_class = titanic_df.groupby("Pclass")["Age"].median()
titanic_df["Age"] = titanic_df.apply(
    lambda row: median_age_by_class[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"],
    axis=1
)

# Verifying that "Age" values do not have missing values 

profile_data(titanic_df)

# Re-defining the bins to solve the NaN created before the "Age" missing values were fixed
titanic_df["Age_bins"] = pd.cut( titanic_df["Age"] , bins = np.arange(0,100,10) )
titanic_df["Age_bins_2"] = pd.cut( 
    titanic_df["Age"] , 
    bins = [0,5,10,15,20, 30, 40, 50 , 60, 70 , 80 , 90] , 
    right= False ,
    include_lowest= True
)


## Alternative : determine Age based on polynomial regression using Fare and Pclass to find Age

# could do : boarding port ? siblings and parents how to exploit it . 
# Cabin info to extract from based on official info (but many are missing) , like deck, floor etc... ?
# Using Age (numerical) or using Age bin (categorical) in the model hereafter ?




#############################################
####                                     ####
####    MACHINE LEARNING ANALYSIS        ####
####                                     ####
#############################################


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, KBinsDiscretizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics,svm


# titanic_df.index     = titanic_df.iloc[:,0]
# titanic_df_to_use    = titanic_df[["Pclass","Sex","Fare"]]
# titanic_df_to_target = titanic_df["Survived"]


##### (1) Basic Version (basic encoding, not much processing)

#####   PRE-PROCESSING OF SELECTED FEATURES     ####

numerical_var = titanic_df[["Fare", "Pclass"]]

### Handle categorical variables
categorical_var     = titanic_df[["Sex"]]
label_encoder       = LabelEncoder()

# Iterate over each categorical column and encode it
for column in categorical_var.columns:
    if categorical_var[column].dtype == 'object':
        categorical_var[column] = label_encoder.fit_transform(categorical_var[column])

categorical_var.values


X = pd.concat( [numerical_var , categorical_var] , axis=1)

Y  = titanic_df["Survived"] # .reshape(-1,1)
Y



X = X.values
y = Y.values


#####   SPLIT-OUT TRAIN/TEST DATASET (non stratified! e.g. likely respecting data imbalance)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)


# Prepare ML Algorithms
models = []
models.append(('LogReg', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNeig', KNeighborsClassifier()))
models.append(('Dec Tree', DecisionTreeClassifier()))
models.append(('Gauss', GaussianNB()))
models.append(('SVM', svm.SVC(gamma='auto')))

# evaluate each model in turn, using 10 fold cross validation
results = []
names   = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.xlabel("algorithm")
plt.ylabel("Accuracy")

plt.savefig('c:\\ndsha_maps\\learning\\python\\DevAcademy\\Projects\\06_Classification\\pictures\\5_algo_scores.jpg', dpi = 150)


##### (2) Improved Version (label encoding + standard scaling)


#####  ENGINEERING ALL FEATURES ####

numerical_var = titanic_df[["Fare"]]
std_scaler = StandardScaler()

## FIT and TRANSFORM the DATA
scaler_Z = std_scaler.fit_transform(numerical_var)
# scaler_Z[:5]


### Handle categorical variables
categorical_var     = titanic_df[["Sex","Pclass"]]
label_encoder       = LabelEncoder()
# Iterate over each categorical column and encode it
for column in categorical_var.columns:
    if categorical_var[column].dtype == 'object':
        categorical_var[column] = label_encoder.fit_transform(categorical_var[column])

categorical_var.values


# Reshape array1 to match the number of columns in array2
scaler_Z_reshaped = np.reshape(scaler_Z, (891, 1))

# Concatenate arrays along the second axis (columns)
X = np.concatenate((scaler_Z_reshaped, categorical_var.values), axis=1)

Y  = titanic_df["Survived"] # .reshape(-1,1)
Y


#X = X.values
y = Y.values


# Data Split : not stratified !
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# Prepare ML Algorithms
models = []
models.append(('LogReg', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNeig', KNeighborsClassifier()))
models.append(('Dec Tree', DecisionTreeClassifier()))
models.append(('Gauss', GaussianNB()))
models.append(('SVM', svm.SVC(gamma='auto')))

# evaluate each model in turn
results = []
names   = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.xlabel("algorithm")
plt.ylabel("Accuracy")
#plt.show()

plt.savefig('c:\\ndsha_maps\\learning\\python\\DevAcademy\\Projects\\06_Classification\\pictures\\5_algo_scores_label_scaled.jpg', dpi = 150)



### SELECTING THE BEST MODEL FROM ABOVE AND APPLYING TO TEST DATASET (from previous split)

# Make predictions on validation dataset
model = svm.SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_test)


# Evaluate predictions
print(metrics.accuracy_score(Y_test, predictions))
print(metrics.confusion_matrix(Y_test, predictions))
print(metrics.classification_report(Y_test, predictions))


