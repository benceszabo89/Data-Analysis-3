# Analysis of the Airbnb prices in Sicily

## 1. Introduction & Data

In this assignment I analysed the Airbnb prices in Sicily 2025Q2. I built 5 different statistical models (OLS, LASSO, CART, Random Forest, GBM boosting) to predict the prices. For the evaluation of the models I used 2 holdout datasets (Airbnb prices in Sicily 2025Q3 and Airbnb prices in Puglia 2025Q3). In the analysis I reported the prediction error of all 3 datasets.

## 2. Data Wrangling

I created 2 cleaning functions for the datasets. The first one selects the important features of the datasets and does the basic data cleaning (cleans the target variable, creates the amenities dummies, creates dummies for room types, bathrooms, kid-friendliness and superhost status). Overall I analysed the following features:
1. basic listing features:
	- accommodates: number of accommodates
	- beds: number of beds
	- bathrooms: number of bathrooms
	- d_apartment: indicator for apartments (room type)
	- d_private_room: indicator for private rooms (room type)
	- d_shared_room: indicator for shared rooms (room type)
	- minimum_nights: minimum nights required for booking
2. other listing features:
	- n_amenities: number of amenities
	- d_kids_amenities: indicator for any kid amenities
	- d_bath_private: indicator for private bath (bathroom type)
	- d_bath_shared: indicator for shared bath (bathroom type)
3. amenities:
	- d_amenity_kitchen,
 	- d_amenity_wifi,
 	- d_amenity_heating,
 	- d_amenity_cooking_basics,
 	- d_amenity_air_conditioning,
 	- d_amenity_luggage_dropoff_allowed,
 	- d_amenity_tv,
 	- d_amenity_pets_allowed,
 	- d_amenity_free_street_parking,
 	- d_amenity_private_entrance,
 	- d_amenity_dedicated_workspace,
 	- d_amenity_patio_or_balcony,
 	- d_amenity_sea_view,
 	- d_amenity_waterfront,
	- d_amenity_elevator,
 	- d_amenity_self_check_in
4. host features:
	- d_superhost
5. listing reviews
	- number_of_reviews
	- review_scores_rating

I filtered out the listings with missing prices and also the listings where minimum nights required were more than 14 days as I wanted to analyse short-term rentals.

The second cleaning function drops rows where the target variable is missing, inputs the missing values with 1 in bathrooms and bedrooms and with the median in review scores. I ran the data wrangling functions in all 3 datasets.

## 3. Modelling

I built 3 different models, starting with the simplest one to the more complex ML boosting algorithm and reported the summary of the model performance for comparison.

### 3.1. OLS

I built 5 models, adding more and more features from the above categories. The simplest model has 6, and the most complex model has 28 features. I saw that both the train and holdout RMSEs dropped in case of the 3rd model with 25 features. I looked at the coefficients and the direction of the coefficients seems to be reasonable, for example the price increases with the number of accommodates, number of beds and bathrooms, if the room type is an apartment or a private room compared to a shared room, etc. The price decreases if the minimum nights requirement increases.

### 3.2 LASSO

I ran the LASSO selection with the most extensive list of features, including all the above features. The LASSO selected 27 variables for the analysis. I saw significant overlap with the 3rd OLS model in terms of the selected features. We can see only a slight decrease in the train and test RMSEs.

### 3.3 CART

Next I ran a regression tree with pruning where I set the maximum depth of 5 to avoid too complex models. For the model selection I used cross-validated RMSE. The model ended with 17 final nodes and it used 9 different features. From the feature importance chart we can see that all the selected features (minimum nights, luggage drop-off, balcony, private entrance, sea view, number of amenities, superhost status, number of reviews and number of bathrooms) have more than 1% importance. The model produces slightly lower RMSE than the previous models.

### 3.4 Random Forest

I saw that the decision tree didn't include other, seemingly important features so I tried the Random Forest algorithm. I did the grid tuning with 5, 10, 15, 20 and 25 maximum features and minimum sample sizes in the leaf nodes of 500, 1000 and 1500. Again, I used the cross-validated RMSE for model selection. Overall, the model used 25 different features. The most important features in this model correspond to the features in the decision tree model. The RMSE is significantly lower in the different city holdout set than the previous models, however, almost 3 times as many features were used in this model.

### 3.5 GBM

Finally I used the GBM boosting model. Did the grid tuning with 20 and 30 estimators with the maximum depth of 5 and 10. The train RMSE dropped but the test RMSE is only slightly lower compared to the Random Forest (unfortunately I had to run the model twice because first time I messed up, that's why it is appended twice to the model comparison table).

## 4. Summary

While simpler OLS and LASSO models provided a solid baseline with intuitive coefficients, the more complex tree-based methods (Random Forest and GBM) produced better predictive performance. Specifically, the Random Forest model achieved a significantly lower RMSE on the city holdout set, suggesting it generalized better to new locations than the single Decision Tree. Although the GBM model offered the lowest training error, compared to the Random Forest the improvement was marginal for the cost of increased complexity. The simplest algorithm-based model was the CART with 9 features.
