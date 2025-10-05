
# Churn Prediction and Recommendation System for a Streaming Service

## Executive Summary

This project report details two interconnected data science initiatives aimed at enhancing a streaming service's user engagement and retention: Customer Churn Prediction and a Movie Recommendation System.

The **Customer Churn Prediction** project successfully developed a Decision Tree classifier, achieving an F1-score of 0.9934, to identify users at risk of churning. This provides a powerful tool for proactive customer retention.

The **Movie Recommendation System** project implemented an Item-Based Collaborative Filtering approach to personalize the user experience. The system generates movie recommendations based on user ratings. A key challenge identified was data consistency between movie IDs in the ratings and metadata, which limited the full retrieval of movie titles.

Together, these projects offer a comprehensive strategy to combat churn and improve user satisfaction by identifying at-risk users and enhancing content discovery.

## Rationale

In the competitive streaming industry, customer retention and engagement are vital. High churn rates impact revenue and increase acquisition costs. Accurately predicting churn allows for targeted retention strategies, reducing costs and improving customer lifetime value.

Personalized recommendations are crucial for user satisfaction and engagement. A well-designed recommendation system helps users discover content they enjoy, increasing viewing time and reducing churn. Combining these initiatives creates a powerful synergy: identifying at-risk users and proactively engaging them with personalized content.

## Research Questions

1.  **Customer Churn Prediction:** Can we accurately predict customer churn for the streaming service based on available customer data, and what factors are most indicative of churn?
2.  **Movie Recommendation System:** How can we build an effective movie recommendation system using available user ratings to provide personalized suggestions to users on a streaming platform?

## Data Sources

### Customer Churn Prediction

*   A [Streaming Service Data](https://www.kaggle.com/datasets/akashanandt/streaming-service-data?resource=download) dataset from Kaggle
*   Contains customer information, subscription details, engagement metrics, and churn status.

### Movie Recommendation System

*   **A subset of The [Movie Dataset](https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Frounakbanik%2Fthe-movies-dataset) from Kaggle, specifically:**
  
    *   `movies_metadata.csv`: Contains metadata for movies.
    *   `ratings_small.csv`: Contains a subset of user ratings.
    *   `links_small.csv`: Contains links between movie identifiers.
    *   `keywords.csv`: Contains movie plot keywords (though not extensively used).

## Methodology

### Customer Churn Prediction Methodology

1.  **Data Loading and Exploration (EDA):** Loaded and explored the dataset, identified missing values, and visualized relationships between features and churn. Missing values in 'Age' and 'Satisfaction_Score' were imputed with the mean.
2.  **Data Preprocessing:** Split data into features (X) and target (y), then into training and testing sets. Categorical features were one-hot encoded, and numerical features were scaled using StandardScaler.
3.  **Model Building and Evaluation:** Trained and evaluated classification models: Dummy Classifier, Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Support Vector Machine (SVM), using accuracy, precision, recall, and F1-score.
4.  **Hyperparameter Tuning (GridSearchCV):** Used GridSearchCV with cross-validation to optimize hyperparameters for Logistic Regression, KNN, Decision Tree, and SVM.
5.  **Model Comparison:** Compared performance metrics of all models to identify the best performer.

### Movie Recommendation System Methodology

1.  **Data Loading and Preparation:** Loaded CSV files, cleaned identifier columns, handled missing values, and converted string representations of lists.
2.  **Data Merging:** Merged dataframes to create `recommendation_df` with user, movie, rating, and some movie metadata.
3.  **Item-Based Collaborative Filtering Implementation:**
    *   Created a user-item pivot table from `recommendation_df`.
    *   Calculated cosine similarity between movies to build an item similarity matrix.
    *   Developed a function to generate recommendations for a user based on highly-rated, unseen movies.

## Results

### Customer Churn Prediction Results

The Decision Tree model achieved exceptional performance with an F1-score of 0.9934 on the test set. Hyperparameter tuning confirmed its strong performance with a best cross-validation F1-score of 0.9896.

| Model                 | Accuracy | Precision | Recall | F1-score |
| :-------------------- | :------- | :-------- | :----- | :------- |
| Dummy                 | 0.545    | 0.000     | 0.000  | 0.000    |
| Logistic Regression   | 0.806    | 0.792     | 0.778  | 0.785    |
| KNN                   | 0.843    | 0.855     | 0.789  | 0.821    |
| Decision Tree         | 0.994    | 0.998     | 0.989  | 0.993    |
| SVM                   | 0.940    | 0.946     | 0.921  | 0.933    |

### Movie Recommendation System Results

*   Data was successfully loaded, cleaned, and merged.
*   An Item-Based Collaborative Filtering model was implemented to calculate movie similarities.
*   The system can generate recommended movie IDs.
*   Retrieving movie titles was impacted by inconsistencies between movie IDs in the ratings data and movie metadata.

## Next Steps

### Customer Churn Prediction Next Steps

1.  **Investigate Decision Tree Overfitting:** Explore pruning or ensemble methods.
2.  **Explore Ensemble Methods:** Train and evaluate Random Forests and Gradient Boosting models.
3.  **Feature Importance Analysis:** Identify most influential churn factors for business insights.
4.  **Deployment and Monitoring:** Prepare for production deployment and continuous monitoring.

### Movie Recommendation System Next Steps

1.  **Address Data Consistency:** Resolve discrepancies between movie IDs in ratings and metadata.
2.  **Formal Model Evaluation:** Implement a proper evaluation framework using metrics like Precision@K, Recall@K, or AUC.
3.  **Explore Parameter Tuning:** Experiment with similarity metrics and number of similar items.
4.  **Consider Other Collaborative Filtering Techniques:** Explore User-Based Collaborative Filtering or Matrix Factorization.
5.  **Explore Hybrid Approaches (with better data):** Investigate combining collaborative and content-based methods if more complete data is available.

### Integrated Next Steps

1.  **Combine Insights for Targeted Engagement:** Integrate churn prediction with recommendation system to proactively re-engage at-risk users with personalized suggestions.
2.  **A/B Testing:** Measure the real-world impact of interventions and the recommendation system on retention, engagement, and customer lifetime value.
