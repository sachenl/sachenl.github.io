---
layout: post
title:      "Phase 2 Project"
date:       2021-11-09 00:25:59 -0500
permalink:  phase_2_project
---



## Project Overview

For this project, I used  regression modeling to analyze house sales in a northwestern county.

### The Data

This project uses the King County House Sales dataset, which can be found in  `kc_house_data.csv` in the data folder in this repo. The description of the column names can be found in `column_names.md` in the same folder. As with most real world data sets, the column names are not perfectly described, so you'll have to do some research or use your best judgment if you have questions about what the data means.


### Business Problem

We have had the house selling records for the last few years. With these data, I want to build a model in which I can use the features in the data about the house to predict the price. In this case, we can guide both the seller and buyer to their business. The seller can use the model to predict the selling price of their house and if they need to do any renovation before selling their home. The buyer can have some suggestions about which kind of house they can afford based on their budget. To the details goal：

1. polish the data which have no meaning or is null to the price.
2. remove the features which do not contribute to the house price.
3. check if there are some high correlated features in which some of them can be removed.
4. build the linear regression model.
5. check how the features can contribute to the house change.

## After reviewing the data, I load the house data in to the dataframe and did some necessary modification

I steply removed and polished most of the columns which is not contribute to the price of house
1. The id is not related to the price
2. Split the date file to month and year.
3. Since the lat and long data is high related to the zipcode, I need to remove them.
4. Remodle the zip column with only the first three number
5. Remove the sqft_living15 and sqft_lot15 from columns.
6. Change the yr_built to the age of house at sold time
7. Change the yr_renovated to if the house is renovated and is the renovated within 10 and 30 years at sold.


After the initial data polish, I checked the number of unique value for each of the columns. Some of the columns like price, sqft_living, sqft_lot had more than hundres of unique values which can be consider to be continues values. However, some of the features contain only few unique values. 

I then drawed the distribution of each of the columns  which had more than 10 unique value to check if there is any outlier values.

![fig1](https://raw.githubusercontent.com/sachenl/dsc-phase-2-project/main/pictures/fig1.png)

The above figures show that there are multipal columns contain some outlier data. I then collected all the columns and remove the outlier by 1.5 x   IQR
```
to_modify = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','sqft_above','sqft_basement']
for col in to_modify:
    Q1 = df_precessed[col].quantile(0.25)
    Q3 = df_precessed[col].quantile(0.75)
    IQR = Q3 - Q1
    df_precessed = df_precessed[(df_precessed[col] >= Q1 - 1.5*IQR) & (df_precessed[col] <= Q3 + 1.5*IQR)]
```

![](https://raw.githubusercontent.com/sachenl/dsc-phase-2-project/main/pictures/fig2.png)

The data looks much better now with very few of outlier numbers.


### In order to check the relationship between the price with most of the columns with few unique numbers,  I plot their relations in seperate figures.
![](https://raw.githubusercontent.com/sachenl/dsc-phase-2-project/main/pictures/fig3.png)

The figures show that the house price have clear relationship with all of the features. However, there is few figures are pretty close to each other. 

### The scatter plot of each two columns shows in general how the feature realated to each other and if there is any obvious correlation between them.

![](https://raw.githubusercontent.com/sachenl/dsc-phase-2-project/main/pictures/fig4.png)

Base on the scatter figure above, there are several features correlated with each other. However, visual approach to finding correlation cannot be automated, so a numeric approach is a good next step.


### I tested the pairs of feature with correlation more than 0.75.
```
df = df_precessed.corr().abs().stack().reset_index().sort_values(0, ascending = False)
df['pairs'] = list(zip(df.level_0, df.level_1))
df.set_index(['pairs'], inplace = True)
df.drop(columns = ['level_0', "level_1"], inplace  = True)
df.columns = ['cc']
df.drop_duplicates(inplace = True)
df[(df.cc>.7) & (df.cc<1)]
```

pairs                                                                      CC

(sqft_living, sqft_above)	                             0.814755
(renovated_30, is_renovated)	                 0.808098
(month, year)                                                 	0.786899



There are three pairs of features high related with each other. I need to remove at least one of the features in each pair. Comparing the last list, I decided to delete the columns sqft_above, renovated_30, year, month. 



# Regression
Until now, I finished the polish of the all the features and then I will split the data to trainning and testing parts to do the fitting.

### split the data to training and testing part
```
from sklearn.model_selection import train_test_split
y = df_precessed['price']
X = df_precessed.drop('price', axis  = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(len(X_train), len(X_test), len(y_train), len(y_test))
```

I then checked the  heatmap of the data to find out the most correlated feature and make the base line


![fig5](https://raw.githubusercontent.com/sachenl/dsc-phase-2-project/main/pictures/fig5.png)




Base on the heatmap above, the feature most strongly correlated with the price is sqft_living. So I selected the grade to make the baseline of linear regression fitting.
    
most_correlated_feature = "grade"
```
baseline_model = LinearRegression()


splitter = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)

baseline_scores = cross_validate(
    estimator=baseline_model,
    X=X_train[[most_correlated_feature]],
    y=y_train,
    return_train_score=True,
    cv=splitter
```

Based on the ''grade'' feature,  I make the baseline_model and find the score of trainning and validation score. 

Train score:      0.30102469477546406
Validation score: 0.3146285604628449

Because we are using the .score method of LinearRegression, these are r-squared scores. That means that each of them represents the amount of variance of the target ( price) that is explained by the model's features (currently just the number of grade) and parameters (intercept value and coefficient values for the features).

In general this seems like not a very strong model. However, it is getting nearly identical performance on training subsets compared to the validation subsets, explaining around 50% of the variance both times.

We will need to add more features to the model to check if there is any improvement.

### Build a Model with All Numeric Features 

```
second_model = LinearRegression()

second_model_scores = cross_validate(
    estimator=second_model,
    X=X_train,
    y=y_train,
    return_train_score=True,
    cv=splitter
)
```
Current Model
Train score:      0.5322810924227112
Validation score: 0.5360008901543358

Baseline Model
Train score:      0.30102469477546406
Validation score: 0.3146285604628449


Our second model got better scores on the training data, and better scores on the validation data. However, I still want to continue to check how each feature work in general. Then I choose to check the coef value of the regression

###  Select the Best Combination of Features
I checked the linear regression fitness for all the features first.

```
import statsmodels.api as sm

sm.OLS(y_train, sm.add_constant(X_train)).fit().summary()
```

The regresssion results showed as below

![](https://raw.githubusercontent.com/sachenl/dsc-phase-2-project/main/pictures/regression.png)

### Base on the p value, I temperaly select 10 columns in which p<0.05
```
select_cat = ['bedrooms', 'bathrooms','sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
              'sqft_basement', 'renovated_10', 'age_sold']
X_train_third = X_train[select_cat]

third_model = LinearRegression()

third_model_scores = cross_validate(
    estimator=third_model,
    X=X_train_third,
    y=y_train,
    return_train_score=True,
    cv=splitter
)
```

current Model
Train score:      0.53218577738026
Validation score: 0.536160764472546

second Model
Train score:      0.5322810924227112
Validation score: 0.5360008901543358

Baseline Model
Train score:      0.30102469477546406
Validation score: 0.3146285604628449


There is a little bit improve on the prediction, but very little.


 I also tried to selecting Features with sklearn.feature_selection

```
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler

#Importances are based on coefficient magnitude, so
#we need to scale the data to normalize the coefficients
X_train_for_RFECV = StandardScaler().fit_transform(X_train)

model_for_RFECV = LinearRegression()

#Instantiate and fit the selector
selector = RFECV(model_for_RFECV, cv=splitter)
selector.fit(X_train_for_RFECV, y_train)

#Print the results
print("Was the column selected?")
for index, col in enumerate(X_train.columns):
    print(f"{col}: {selector.support_[index]}")
```

Was the column selected?
bedrooms: True
bathrooms: True
sqft_living: True
sqft_lot: True
floors: True
waterfront: True
view: True
condition: True
grade: True
sqft_basement: True
zipcode: False
is_renovated: False
renovated_10: True
age_sold: True

The RFE methods give me the same selection of features above.

The results showed that the auto sedlected features did not give better score than the third model.

Now, I remade the third model features to best_features to validate the final model.


#Base on the train score and validation score, the best columns until now is the third model. 


```
X_train_final = X_train[select_cat]
X_test_final = X_test[select_cat]


final_model = LinearRegression()

#Fit the model on X_train_final and y_train
final_model.fit(X_train_final, y_train)

#Score the model on X_test_final and y_test
#use the built-in .score method
final_model.score(X_test_final, y_test)
```


# Validation
## import the mse to check the mse value
```
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, final_model.predict(X_test_final), squared=False)

#check the distribution of price in test data
y_test.hist(bins = 100)
y_test.mean()
```
![](https://raw.githubusercontent.com/sachenl/dsc-phase-2-project/main/pictures/download.png)

The mse value is 130331. The mean of the price is 445683.

 This means that for an average house price, this algorithm will be off by about $130331 thousands. Given that the mean value of house price is 445683, the algorithm can patially set the price. However, we still want to have a human double-check and adjust these prices rather than just allowing the algorithm to set them. 
 I then printed the final coefficients of each features. 

bedrooms         -17734.819592
bathrooms         22333.279123
sqft_living         104.197345
sqft_lot             -7.266691
floors            17895.940478
waterfront       140605.432066
view              30502.795377
condition         20867.253746
grade            104396.240619
sqft_basement        10.140670
renovated_10      46690.148414
age_sold           2655.946868
Name: Coefficients, dtype: float64

Intercept: -670009.6123770761

The coefficients of the selected features shows that the sqft_living is the most important features which affected the house price since the sqft_living is normally more than 1000. The affection of features sqft_lot and sqft_basement are very small even they are negtive numbers.  The affection of grade on the house is very big also. Also, if the house is renovated within 10 years, the price will increased 46690 dollars. If there is waterfront of the house, the price of the house will increase dramtically with 140605.


### For the validation, I first plot the scatter plot of Predicted Price vs the Actual Price
```
preds = final_model.predict(X_test_final)
fig, ax = plt.subplots(figsize =(5,5))

perfect_line = np.arange(y_test.min(), y_test.max())
#perfect_x = [0, 1]
#perfect_y = [0, 1]

#ax.plot(perfect_x, perfect_y, linestyle="--", color="black", label="Perfect Fit")
ax.plot(perfect_line,perfect_line, linestyle="--", color="black", label="Perfect Fit")
ax.scatter(y_test, preds, alpha=0.5)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.legend();
```
![](https://raw.githubusercontent.com/sachenl/dsc-phase-2-project/main/pictures/fig6.png)

### I then tested the residuals by qqplot

```
import scipy.stats as stats
residuals = (y_test - preds)
sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True);


```


![](https://raw.githubusercontent.com/sachenl/dsc-phase-2-project/main/pictures/fig7.png)



```
fig, ax = plt.subplots()
ax.scatter(preds, residuals, alpha=0.5)
ax.plot(preds, [0 for i in range(len(X_test))])
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Actual - Predicted Value");
```
![](https://raw.githubusercontent.com/sachenl/dsc-phase-2-project/main/pictures/fig8.png)

The validation of prediction and real data shows that the prediction price for most house whose price is low (20% of the max price) is close to the real price.  qqplot showes that the house price is well predicted when the house price is not very high. However, for the high value price house, the prediction is not very acturate. There is a lot of shift of prediction price when the house value increase especialy when house price is more than 2 million. 


## Summary

Our model predicted well the house price on many of the features. The Coefficients are like between bedrooms -17734, bathrooms 22333, sqft_living 104, sqft_lot -7, floors 17895, waterfront 140605 , view 30502, condition 20867 , grade 104396, sqft_basement 10 , renovated_10 46690, age_sold 2655,

To the buyer, they can estimate the price of the house base on the features of the house. 
To the seller, if they want to sell the house in a better value, they can try to renovate the house and make water front if possible. They can also doing something to improve the grade level of the house which can also increase the house value dramaticlly. 


