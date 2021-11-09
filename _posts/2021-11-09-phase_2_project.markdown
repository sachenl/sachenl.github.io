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

![](http://data:image/png;base64,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%0A)

#The above figures show that there are multipal columns contain some outlier data. I then collected all the columns and remove them 
to_modify = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','sqft_above','sqft_basement']
for col in to_modify:
    Q1 = df_precessed[col].quantile(0.25)
    Q3 = df_precessed[col].quantile(0.75)
    IQR = Q3 - Q1
    df_precessed = df_precessed[(df_precessed[col] >= Q1 - 1.5*IQR) & (df_precessed[col] <= Q3 + 1.5*IQR)]

### check the data after modification
fig, axs = plt.subplots(2,5, figsize = (15,6))
plt1 = sns.boxplot(df_precessed['price'], ax = axs[0,0])
plt2 = sns.boxplot(df_precessed['bedrooms'], ax = axs[0,1])
plt3 = sns.boxplot(df_precessed['bathrooms'], ax = axs[0,2])
plt4 = sns.boxplot(df_precessed['sqft_living'], ax = axs[0,3])
plt5 = sns.boxplot(df_precessed['sqft_lot'], ax = axs[0,4])
plt1 = sns.boxplot(df_precessed['floors'], ax = axs[1,0])
plt2 = sns.boxplot(df_precessed['sqft_above'], ax = axs[1,1])
plt3 = sns.boxplot(df_precessed['sqft_basement'], ax = axs[1,2])
plt4 = sns.boxplot(df_precessed['age_sold'], ax = axs[1,3])

The data looks much better now with very few of outlier numbers.


#  In order to check the relationship between the price with most of the columns with few unique numbers, 
# I plot their relations in seperate figures.
plt.figure(figsize=(20, 12))
plt.subplot(4,3,1)
sns.boxplot(x = 'bedrooms', y = 'price', data = df_precessed)
plt.subplot(4,3,2)
sns.boxplot(x = 'floors', y = 'price', data = df_precessed)
plt.subplot(4,3,3)
sns.boxplot(x = 'waterfront', y = 'price', data = df_precessed)
plt.subplot(4,3,4)
sns.boxplot(x = 'view', y = 'price', data = df_precessed)
plt.subplot(4,3,5)
sns.boxplot(x = 'condition', y = 'price', data = df_precessed)
plt.subplot(4,3,6)
sns.boxplot(x = 'grade', y = 'price', data = df_precessed)
plt.subplot(4,3,7)

sns.boxplot(x = 'is_renovated', y = 'price', data = df_precessed)
plt.subplot(4,3,8)
sns.boxplot(x = 'renovated_10', y = 'price', data = df_precessed)
plt.subplot(4,3,9)
sns.boxplot(x = 'renovated_30', y = 'price', data = df_precessed)
plt.subplot(4,3,10)
sns.boxplot(x = 'bathrooms', y = 'price', data = df_precessed)
plt.subplot(4,3,11)

sns.boxplot(x = 'month', y = 'price', data = df_precessed)
plt.show()

### The scatter plot of each two columns shows in general how the feature realated to each other and if there is any obvious correlation between them.
scatter_matrix = pd.plotting.scatter_matrix(
    df_precessed,
    figsize  = [20, 20],
    marker   = ".",
    s        = 0.2,
    diagonal = "kde"
)

for ax in scatter_matrix.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize = 10, rotation = 90)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 10, rotation = 0)

Base on the scatter figure above, there are several features correlated with each other. However, visual approach to finding correlation cannot be automated, so a numeric approach is a good next step.


### I tested the pairs of feature with correlation more than 0.75.
df = df_precessed.corr().abs().stack().reset_index().sort_values(0, ascending = False)
df['pairs'] = list(zip(df.level_0, df.level_1))
df.set_index(['pairs'], inplace = True)
df.drop(columns = ['level_0', "level_1"], inplace  = True)
df.columns = ['cc']
df.drop_duplicates(inplace = True)
df[(df.cc>.7) & (df.cc<1)]


There are three pairs of features high related with each other. I need to remove at least one of the features in each pair. Comparing the last list, I decided to delete the columns sqft_above, renovated_30, year, month. 

to_drop = ['sqft_above', 'renovated_30', 'year', 'month' ]
df_precessed = df_precessed.drop(to_drop,axis  = 1 )


# Regression
Until now, I finished the polish of the all the features and then I will split the data to trainning and testing parts to do the fitting.

### split the data to training and testing part
from sklearn.model_selection import train_test_split
y = df_precessed['price']
X = df_precessed.drop('price', axis  = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(len(X_train), len(X_test), len(y_train), len(y_test))

#check heatmap of the data to find out the most correlated feature and make the base line

heatmap_data = pd.concat([y_train, X_train], axis = 1)
corr = heatmap_data.corr()

#setup figure for heatmap
fig, ax = plt.subplots(figsize = (15, 15))

### Plot a heatmap of the correlation matrix, with both numbers and colors indicating the correlations
sns.heatmap(
    # Specifies the data to be plotted
    data=corr,
    # The mask means we only show half the values,
    # instead of showing duplicates. It's optional.
    mask=np.triu(np.ones_like(corr, dtype=bool)),
    # Specifies that we should use the existing axes
    ax=ax,
    # Specifies that we want labels, not just colors
    annot=True,
    # Customizes colorbar appearance
    cbar_kws={"label": "Correlation", "orientation": "horizontal", "pad": .2, "extend": "both"}
    )

    #Customize the plot appearance
    ax.set_title("Heatmap of Correlation Between Attributes (Including Price)");
    
most_correlated_feature = "grade"


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, ShuffleSplit

baseline_model = LinearRegression()


splitter = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)

baseline_scores = cross_validate(
    estimator=baseline_model,
    X=X_train[[most_correlated_feature]],
    y=y_train,
    return_train_score=True,
    cv=splitter
)

print("Train score:     ", baseline_scores["train_score"].mean())
print("Validation score:", baseline_scores["test_score"].mean())

Because we are using the .score method of LinearRegression, these are r-squared scores. That means that each of them represents the amount of variance of the target ( price) that is explained by the model's features (currently just the number of grade) and parameters (intercept value and coefficient values for the features).

In general this seems like not a very strong model. However, it is getting nearly identical performance on training subsets compared to the validation subsets, explaining around 50% of the variance both times.

We will need to add more features to the model to check if there is any improvement.

## Build a Model with All Numeric Features 

second_model = LinearRegression()

second_model_scores = cross_validate(
    estimator=second_model,
    X=X_train,
    y=y_train,
    return_train_score=True,
    cv=splitter
)

print("Current Model")
print("Train score:     ", second_model_scores["train_score"].mean())
print("Validation score:", second_model_scores["test_score"].mean())
print()
print("Baseline Model")
print("Train score:     ", baseline_scores["train_score"].mean())
print("Validation score:", baseline_scores["test_score"].mean())

Our second model got better scores on the training data, and better scores on the validation data. However, I still want to continue to check how each feature work in general. Then I choose to check the coef value of the regression

###  Select the Best Combination of Features
import statsmodels.api as sm

sm.OLS(y_train, sm.add_constant(X_train)).fit().summary()

### Base on the p value, I temperaly select 10 columns in which p<0.05
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

print("current Model")
print("Train score:     ", third_model_scores["train_score"].mean())
print("Validation score:", third_model_scores["test_score"].mean())

print("second Model")
print("Train score:     ", second_model_scores["train_score"].mean())
print("Validation score:", second_model_scores["test_score"].mean())
print()
print("Baseline Model")
print("Train score:     ", baseline_scores["train_score"].mean())
print("Validation score:", baseline_scores["test_score"].mean())

There is a little bit improve on the prediction, but very little.
 I tried to selecting Features with sklearn.feature_selection

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


X_train_final = X_train[select_cat]
X_test_final = X_test[select_cat]


final_model = LinearRegression()

#Fit the model on X_train_final and y_train
final_model.fit(X_train_final, y_train)

#Score the model on X_test_final and y_test
#use the built-in .score method
final_model.score(X_test_final, y_test)


# Validation
## import the mse to check the mse value
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, final_model.predict(X_test_final), squared=False)

#check the distribution of price in test data
y_test.hist(bins = 100)
y_test.mean()

""" This means that for an average house price, this algorithm will be off by about $130331 thousands. Given that the mean value of house price is 445683, the algorithm can patially set the price. However, we still want to have a human double-check and adjust these prices rather than just allowing the algorithm to set them. """

print(pd.Series(final_model.coef_, index=X_train_final.columns, name="Coefficients"))
print()
print("Intercept:", final_model.intercept_)



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

import scipy.stats as stats
residuals = (y_test - preds)
sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True);

fig, ax = plt.subplots()

ax.scatter(preds, residuals, alpha=0.5)
ax.plot(preds, [0 for i in range(len(X_test))])
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Actual - Predicted Value");

The validation of prediction and real data shows that the prediction price for most house whose price is low (20% of the max price) is close to the real price. qqplot showes that the house price is not well normal distributed but peaked in the middle. There is a lot of shift of prediction price when the house value increase especialy when house price is more than 2 million.


## Summary

Our model predicted well the house price on many of the features. The Coefficients are like between bedrooms -17734, bathrooms 22333, sqft_living 104, sqft_lot -7, floors 17895, waterfront 140605 , view 30502, condition 20867 , grade 104396, sqft_basement 10 , renovated_10 46690, age_sold 2655,

To the buyer, they can estimate the price of the house base on the features of the house. To the seller, if they want to sell the house in a better value, they can try to renovate the house and make water front if possible.


