Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
Import Dataset
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
sub = pd.read_csv('sample_submission.csv')
train.head()
id	Gender	Age	Driving_License	Region_Code	Previously_Insured	Vehicle_Age	Vehicle_Damage	Annual_Premium	Policy_Sales_Channel	Vintage	Response
0	1	Male	44	1	28.0	0	> 2 Years	Yes	40454.0	26.0	217	1
1	2	Male	76	1	3.0	0	1-2 Year	No	33536.0	26.0	183	0
2	3	Male	47	1	28.0	0	> 2 Years	Yes	38294.0	26.0	27	1
3	4	Male	21	1	11.0	1	< 1 Year	No	28619.0	152.0	203	0
4	5	Female	29	1	41.0	1	< 1 Year	No	27496.0	152.0	39	0
test.head()
id	Gender	Age	Driving_License	Region_Code	Previously_Insured	Vehicle_Age	Vehicle_Damage	Annual_Premium	Policy_Sales_Channel	Vintage
0	381110	Male	25	1	11.0	1	< 1 Year	No	35786.0	152.0	53
1	381111	Male	40	1	28.0	0	1-2 Year	Yes	33762.0	7.0	111
2	381112	Male	47	1	28.0	0	1-2 Year	Yes	40050.0	124.0	199
3	381113	Male	24	1	27.0	1	< 1 Year	Yes	37356.0	152.0	187
4	381114	Male	27	1	28.0	1	< 1 Year	No	59097.0	152.0	297
train.shape
(381109, 12)
Checking null values in data
train.isnull().sum()
id                      0
Gender                  0
Age                     0
Driving_License         0
Region_Code             0
Previously_Insured      0
Vehicle_Age             0
Vehicle_Damage          0
Annual_Premium          0
Policy_Sales_Channel    0
Vintage                 0
Response                0
dtype: int64
Hence proved, there is no missing data in out train data.

Check Columns (Categorical or Numeric)
It gives the names of categorical, numerical, and categorical but cardinal variables in the data set.

Note: Categorical variables with numerical appearance are also included in categorical variables.

def grab_col_names(dataframe, cat_th=10, car_th=20):


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(train)
Observations: 381109
Variables: 12
cat_cols: 6
num_cols: 6
cat_cols,num_cols
(['Gender',
  'Vehicle_Age',
  'Vehicle_Damage',
  'Driving_License',
  'Previously_Insured',
  'Response'],
 ['id',
  'Age',
  'Region_Code',
  'Annual_Premium',
  'Policy_Sales_Channel',
  'Vintage'])
Removing id & Policy_Sales_Channel column from num_cols because these columns are not providing any useful insights for modeling
num_cols = [col for col in num_cols if "id" not in col]
num_cols = [col for col in num_cols if "Policy_Sales_Channel" not in col]
num_cols
['Age', 'Region_Code', 'Annual_Premium', 'Vintage']
Data Visualizations
Plotting categorical data with ratio
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for i in cat_cols:
    cat_summary(train, i, plot=True)
        Gender     Ratio
Male    206089  54.07613
Female  175020  45.92387

           Vehicle_Age      Ratio
1-2 Year        200316  52.561341
< 1 Year        164786  43.238549
> 2 Years        16007   4.200111

     Vehicle_Damage      Ratio
Yes          192413  50.487656
No           188696  49.512344

   Driving_License      Ratio
1           380297  99.786938
0              812   0.213062

   Previously_Insured      Ratio
0              206481  54.178988
1              174628  45.821012

   Response      Ratio
0    334399  87.743664
1     46710  12.256336

Checking numerical columns for data distribution to check it data is skewed or having outlier
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
#     print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

#     print("#####################################")

for col in num_cols:
    num_summary(train, col, plot=True)




Getting mean data for all numerical columns
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
    
for col in num_cols:
    target_summary_with_num(train, "Response", col)
                Age
Response           
0         38.178227
1         43.435560


          Region_Code
Response             
0           26.336544
1           26.762963


          Annual_Premium
Response                
0           30419.160276
1           31604.092742


             Vintage
Response            
0         154.380243
1         154.112246


Polting box plot for all numerical columns to for outlier checking
sns.boxplot(x=train["Age"]);

As per Annual Premium distribution we can see outliers, but we can't use normal quantile range to remove those because these are not actual outliers and can help us in modeling. Therefore, will set Q1 to 0.05 and Q3 to 0.95
sns.boxplot(x=train["Annual_Premium"]);

There is no outlier in vintage
sns.boxplot(x=train["Vintage"]);

Gender-Response Visualization
To understand insurance ratio for each gender, countplot is givien below
df=train.groupby(['Gender','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
g = sns.catplot(x="Gender", y="count",col="Response",
                data=df, kind="bar",
                height=4, aspect=.7);

Driving license-Gender Visualization
Gender male is more dominant in insurance policy
df=train.groupby(['Gender'])['Driving_License'].count().to_frame().reset_index()
df
Gender	Driving_License
0	Female	175020
1	Male	206089
sns.catplot(x="Gender", y="Driving_License",
                data=df, kind="bar");

Response-Vehicle age Visualization
Further diging into data came to know more people opt for insuranc who's vehicle age is between 1-2 years
df=train.groupby(['Vehicle_Age','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
df
Vehicle_Age	Response	count
0	1-2 Year	0	165510
1	1-2 Year	1	34806
2	< 1 Year	0	157584
3	< 1 Year	1	7202
4	> 2 Years	0	11305
5	> 2 Years	1	4702
g = sns.catplot(x="Vehicle_Age", y="count", col="Response",
                data=df, kind="bar",
                height=4, aspect=.7);

Damage Vehicle-Response Visualization
df=train.groupby(['Vehicle_Damage','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
g = sns.catplot(x="Vehicle_Damage", y="count",col="Response",
                data=df, kind="bar",
                height=4, aspect=.7);

DATA PREPROCESSING
OUTLIERS
Data distribution is used to find outlier, as per out visualisation insights not much data points are out of range.
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 3 * interquantile_range
    low_limit = quartile1 - 3 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
    
for col in num_cols:
    print(f"{col} : {check_outlier(train,col)}")
Age : False
Region_Code : False
Annual_Premium : True
Vintage : False
Method below will return data points which are out of Inter Quartile Range
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    print("\n===========", col_name,"==========")
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
    
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    grab_outliers(train, col)
=========== Age ==========
Empty DataFrame
Columns: [id, Gender, Age, Driving_License, Region_Code, Previously_Insured, Vehicle_Age, Vehicle_Damage, Annual_Premium, Policy_Sales_Channel, Vintage, Response]
Index: []

=========== Region_Code ==========
Empty DataFrame
Columns: [id, Gender, Age, Driving_License, Region_Code, Previously_Insured, Vehicle_Age, Vehicle_Damage, Annual_Premium, Policy_Sales_Channel, Vintage, Response]
Index: []

=========== Annual_Premium ==========
          id  Gender  Age  Driving_License  Region_Code  Previously_Insured  \
1412    1413  Female   41                1         28.0                   0   
11319  11320  Female   50                1         46.0                   1   
13426  13427  Female   47                1         28.0                   0   
15024  15025  Female   32                1         28.0                   0   
25532  25533    Male   50                1         28.0                   0   

      Vehicle_Age Vehicle_Damage  Annual_Premium  Policy_Sales_Channel  \
1412     1-2 Year            Yes        267698.0                 124.0   
11319    1-2 Year             No        508073.0                  26.0   
13426    1-2 Year            Yes        301762.0                 124.0   
15024    1-2 Year            Yes        315565.0                 155.0   
25532    1-2 Year            Yes        229935.0                 122.0   

       Vintage  Response  
1412        63         1  
11319      192         0  
13426       22         0  
15024      150         0  
25532       64         1  

=========== Vintage ==========
Empty DataFrame
Columns: [id, Gender, Age, Driving_License, Region_Code, Previously_Insured, Vehicle_Age, Vehicle_Damage, Annual_Premium, Policy_Sales_Channel, Vintage, Response]
Index: []
Removing out of range outlier and printing columns name
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    
for col in num_cols:
    replace_with_thresholds(train, col)
    
for col in num_cols:
    print(f"{col} : {check_outlier(train,col)}")
Age : False
Region_Code : False
Annual_Premium : False
Vintage : False
CORRELATION ANALYSIS
df.corrwith(train["Response"]).sort_values(ascending=False)
corr_df = train.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr_df, annot=True, xticklabels=corr_df.columns, yticklabels=corr_df.columns, cmap="Blues");
plt.xticks(rotation=45);

FEATURE ENGINEERING
# train['Gender'] = train['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
train=pd.get_dummies(train,drop_first=True)
train = train.rename(columns={'Gender_Male':'Gender'})
train
id	Age	Driving_License	Region_Code	Previously_Insured	Annual_Premium	Policy_Sales_Channel	Vintage	Response	Gender	Vehicle_Age_< 1 Year	Vehicle_Age_> 2 Years	Vehicle_Damage_Yes
0	1	44.0	1	28.0	0	40454.0	26.0	217.0	1	1	0	1	1
1	2	76.0	1	3.0	0	33536.0	26.0	183.0	0	1	0	0	0
2	3	47.0	1	28.0	0	38294.0	26.0	27.0	1	1	0	1	1
3	4	21.0	1	11.0	1	28619.0	152.0	203.0	0	1	1	0	0
4	5	29.0	1	41.0	1	27496.0	152.0	39.0	0	0	1	0	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...
381104	381105	74.0	1	26.0	1	30170.0	26.0	88.0	0	1	0	0	0
381105	381106	30.0	1	37.0	1	40016.0	152.0	131.0	0	1	1	0	0
381106	381107	21.0	1	30.0	1	35118.0	160.0	161.0	0	1	1	0	0
381107	381108	68.0	1	14.0	0	44617.0	124.0	74.0	0	0	0	1	1
381108	381109	46.0	1	29.0	0	41777.0	26.0	237.0	0	1	0	0	0
381109 rows × 13 columns

Column rename and data type value change to integer
train=train.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
train['Vehicle_Age_lt_1_Year']=train['Vehicle_Age_lt_1_Year'].astype('int')
train['Vehicle_Age_gt_2_Years']=train['Vehicle_Age_gt_2_Years'].astype('int')
train['Vehicle_Damage_Yes']=train['Vehicle_Damage_Yes'].astype('int')
train["premium_age_ratio"] = train["Annual_Premium"]/train["Age"]
train["premium_vintage_ratio"] = train["Annual_Premium"]/train["Vintage"]
train
id	Age	Driving_License	Region_Code	Previously_Insured	Annual_Premium	Policy_Sales_Channel	Vintage	Response	Gender	Vehicle_Age_lt_1_Year	Vehicle_Age_gt_2_Years	Vehicle_Damage_Yes	premium_age_ratio	premium_vintage_ratio
0	1	44.0	1	28.0	0	40454.0	26.0	217.0	1	1	0	1	1	919.409091	186.423963
1	2	76.0	1	3.0	0	33536.0	26.0	183.0	0	1	0	0	0	441.263158	183.256831
2	3	47.0	1	28.0	0	38294.0	26.0	27.0	1	1	0	1	1	814.765957	1418.296296
3	4	21.0	1	11.0	1	28619.0	152.0	203.0	0	1	1	0	0	1362.809524	140.980296
4	5	29.0	1	41.0	1	27496.0	152.0	39.0	0	0	1	0	0	948.137931	705.025641
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
381104	381105	74.0	1	26.0	1	30170.0	26.0	88.0	0	1	0	0	0	407.702703	342.840909
381105	381106	30.0	1	37.0	1	40016.0	152.0	131.0	0	1	1	0	0	1333.866667	305.465649
381106	381107	21.0	1	30.0	1	35118.0	160.0	161.0	0	1	1	0	0	1672.285714	218.124224
381107	381108	68.0	1	14.0	0	44617.0	124.0	74.0	0	0	0	1	1	656.132353	602.932432
381108	381109	46.0	1	29.0	0	41777.0	26.0	237.0	0	1	0	0	0	908.195652	176.274262
381109 rows × 15 columns

Normalizing data between -1 and 1
num_feat = ['Age','Vintage','premium_age_ratio','premium_vintage_ratio']
ss = StandardScaler()
train[num_feat] = ss.fit_transform(train[num_feat])
Normalizing data between 0 and 1
mm = MinMaxScaler()
train[['Annual_Premium']] = mm.fit_transform(train[['Annual_Premium']])
train=train.drop('id',axis=1)
cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year','Vehicle_Age_gt_2_Years','Vehicle_Damage_Yes']
for column in cat_feat:
    train[column] = train[column].astype('str')
Applying the same processes to the Test Data
# test['Gender'] = test['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
test=pd.get_dummies(test,drop_first=True)
test = test.rename(columns={'Gender_Male':'Gender'})
test=test.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
test['Vehicle_Age_lt_1_Year']=test['Vehicle_Age_lt_1_Year'].astype('int')
test['Vehicle_Age_gt_2_Years']=test['Vehicle_Age_gt_2_Years'].astype('int')
test['Vehicle_Damage_Yes']=test['Vehicle_Damage_Yes'].astype('int')
test["premium_age_ratio"] = test["Annual_Premium"]/test["Age"]
test["premium_vintage_ratio"] = test["Annual_Premium"]/test["Vintage"]
test=test.drop('id',axis=1)
ss = StandardScaler()
test[num_feat] = ss.fit_transform(test[num_feat])


mm = MinMaxScaler()
test[['Annual_Premium']] = mm.fit_transform(test[['Annual_Premium']])
for column in cat_feat:
    test[column] = test[column].astype('str')
Data Modeling
XGBOOST Classifier
X_train, Y_train = train.drop("Response",axis = 1), train["Response"]
X_train.head()
Age	Driving_License	Region_Code	Previously_Insured	Annual_Premium	Policy_Sales_Channel	Vintage	Gender	Vehicle_Age_lt_1_Year	Vehicle_Age_gt_2_Years	Vehicle_Damage_Yes	premium_age_ratio	premium_vintage_ratio
0	0.333777	1	28.0	0	0.179957	26.0	0.748795	1	0	1	1	0.023841	-0.327306
1	2.396751	1	3.0	0	0.147043	26.0	0.342443	1	0	0	0	-0.783760	-0.333165
2	0.527181	1	28.0	0	0.169680	26.0	-1.521998	1	0	1	1	-0.152904	1.951662
3	-1.148985	1	11.0	1	0.123649	152.0	0.581474	1	1	0	0	0.772756	-0.411377
4	-0.633242	1	41.0	1	0.118306	152.0	-1.378580	0	1	0	0	0.072365	0.632109
Y_train
0         1
1         0
2         1
3         0
4         0
         ..
381104    0
381105    0
381106    0
381107    0
381108    0
Name: Response, Length: 381109, dtype: int64
Setting columns data type as integer to prevent for discrepency
for column in cat_feat:
    X_train[column] = X_train[column].astype('int')
    test[column] = test[column].astype('int')
    
for column in num_feat:
    X_train[column] = X_train[column].astype('int')
    test[column] = test[column].astype('int')
Training Xgboost model
from xgboost import XGBClassifier, plot_importance

# fit model no training data
model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, Y_train)

predictions = model.predict(test)
predictions = [round(value) for value in predictions]
Testing model on reference data
test = pd.DataFrame()
test['Response'] = sub['Response']
test["Pred_Response"] = predictions
test
Response	Pred_Response
0	0	0
1	0	0
2	0	0
3	0	0
4	0	0
...	...	...
127032	0	0
127033	0	0
127034	0	0
127035	0	0
127036	0	0
127037 rows × 2 columns

### Checking Model Accuracy
accuracy = accuracy_score(test['Response'].values, test['Pred_Response'].values)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
Accuracy: 99.68%
