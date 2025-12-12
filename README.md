# What Makes a Recipe Well-Rated?
## An Analysis and Prediction of Food.com Recipe Ratings

**Authors:** Kavya Shah, Ishayu Ghosh  
**Course:** DSC 80 – Data Science in Practice  
---

## Introduction

Food.com is a popular platform where users can discover, share, and rate recipes. Understanding what makes a recipe successful—as measured by its average rating—is valuable for both recipe creators and the platform itself. This project investigates factors that influence recipe ratings and builds a predictive model to estimate how well a recipe will be received based on its characteristics.

**Research Question**: What factors influence the average rating of recipes on Food.com, and can we predict a recipe's average rating using only information available when the recipe is first posted?

Our dataset contains information about recipes and user interactions from Food.com. After merging the recipes and interactions datasets, we work with a cleaned dataset containing **231,637 recipes** with the following relevant columns:

- `avg_rating`: The average rating (1-5 stars) for each recipe, computed from non-zero user ratings
- `minutes`: The preparation time in minutes
- `n_steps`: The number of steps in the recipe
- `n_ingredients`: The number of ingredients required
- `calories`, `protein_pdv`, `carbs_pdv`: Nutritional information (calories and percent daily values)
- `submitted`: The submission date of the recipe
- `tags`, `ingredients`, `steps`, `nutrition`: Recipe details stored as lists

We focus on understanding how recipe characteristics like preparation time, complexity (measured by number of steps and ingredients), and nutritional content relate to user ratings.

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

Our data cleaning process involved several key steps:

1. **Merging datasets**: We merged the `RAW_recipes.csv` and `RAW_interactions.csv` datasets on recipe ID to combine recipe information with user ratings.

2. **Handling zero ratings**: We replaced all zero ratings with `NaN` values. Zero ratings likely indicate that a user did not provide a rating rather than a true zero-star rating, so treating them as missing data is more appropriate for our analysis.

3. **Computing average ratings**: We computed the mean rating for each recipe from all non-zero user ratings, creating the `avg_rating` column.

4. **Parsing list columns**: Several columns (`tags`, `ingredients`, `steps`, `nutrition`) were stored as string representations of Python lists. We created a `clean_list_string` function to parse these strings into actual Python lists, enabling us to extract meaningful features like the number of ingredients (`n_ingredients`).

5. **Extracting nutrition information**: We parsed the `nutrition` column to extract individual nutritional values (calories, protein, carbohydrates, etc.) as separate columns.

6. **Creating derived features**: We created `n_ingredients` by counting the length of the ingredients list, and `log_minutes` as a log-transformed version of preparation time to handle the long tail in the distribution.

Here is the head of our cleaned DataFrame:

| id | name | minutes | n_steps | n_ingredients | avg_rating | calories | protein_pdv | carbs_pdv |
|----|------|---------|---------|---------------|------------|----------|-------------|-----------|
| 333281 | 1 brownies in the world    best ever | 40 | 10 | 9 | 4.0 | 138.4 | 10.0 | 50.0 |
| 453467 | 1 in canada chocolate chip cookies | 45 | 12 | 11 | 4.5 | 595.1 | 4.0 | 16.0 |
| 306168 | 412 broccoli casserole | 40 | 6 | 9 | 5.0 | 194.8 | 20.0 | 6.0 |
| 286009 | 412 french onion dip | 40 | 3 | 5 | 5.0 | 403.2 | 16.0 | 4.0 |
| 475785 | 412 tuna casserole | 45 | 8 | 7 | 5.0 | 350.2 | 19.0 | 7.0 |

### Univariate Analysis

We examined the distributions of key variables to understand the characteristics of recipes in our dataset.

<iframe
  src="assets/ingredients_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The distribution of the number of ingredients shows that most recipes use between 5 and 15 ingredients, with a peak around 8-10 ingredients. There is a long tail extending to recipes with 20+ ingredients, though these are relatively rare.

<iframe
  src="assets/prep_time_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The distribution of preparation time (capped at 4 hours for readability) reveals that most recipes take 30-60 minutes to prepare. There is a concentration of quick recipes (under 30 minutes) and a long tail of recipes that take several hours, likely including slow-cooker dishes and other time-intensive preparations.

### Bivariate Analysis

We explored relationships between recipe characteristics and average ratings to identify potential predictors.

<iframe
  src="assets/ingredients_vs_rating.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The scatter plot of number of ingredients versus average rating shows a weak positive relationship. While there is substantial variation, recipes with more ingredients tend to have slightly higher ratings on average. The trendline suggests a modest positive correlation, though the relationship is not particularly strong.

<iframe
  src="assets/prep_time_vs_rating.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The relationship between preparation time and average rating (for recipes taking ≤ 4 hours) shows a slight negative trend. Quicker recipes tend to have slightly higher ratings, though the relationship is weak and there is significant scatter around the trendline. This suggests that preparation time alone is not a strong predictor of recipe success.

### Interesting Aggregates

We examined how average ratings vary by the number of ingredients:

| n_ingredients | avg_rating |
|---------------|------------|
| 1 | 4.12 |
| 2 | 4.18 |
| 3 | 4.25 |
| 4 | 4.30 |
| 5 | 4.35 |
| 6 | 4.38 |
| 7 | 4.40 |
| 8 | 4.42 |
| 9 | 4.43 |
| 10 | 4.44 |

This table reveals a clear pattern: recipes with more ingredients tend to have higher average ratings. The relationship appears to plateau around 8-10 ingredients, suggesting that while complexity (as measured by ingredient count) is associated with higher ratings, there may be diminishing returns beyond a certain point.

---

## Assessment of Missingness

### NMAR Analysis

We believe that the `avg_rating` column is likely **NMAR** (Not Missing At Random). The missingness of average ratings depends on factors that we cannot observe in our dataset. Specifically, recipes may have missing ratings because:

1. **Low engagement**: Recipes that are less appealing, poorly formatted, or difficult to find may receive fewer views and thus fewer ratings. The factors that make a recipe less engaging (e.g., poor presentation, unappealing photos, unclear instructions) are not directly captured in our dataset.

2. **User behavior**: Some users may be more likely to rate recipes they enjoyed, while others may rate recipes they disliked. The decision to rate a recipe depends on unobserved user characteristics and motivations.

3. **Recipe visibility**: Recipes that are featured, promoted, or appear in search results more frequently are more likely to receive ratings. This visibility information is not available in our dataset.

To make the missingness MAR (Missing At Random), we would need additional data such as:
- Number of views or impressions for each recipe
- Recipe visibility metrics (search ranking, featured status)
- User engagement metrics (time spent viewing recipe, number of saves)
- Recipe presentation quality scores (photo quality, formatting)

### Missingness Dependency

We performed permutation tests to analyze the dependency of `avg_rating` missingness on other columns.

**Missingness of `avg_rating` vs `year_submitted`:**

- **Null Hypothesis (H₀):** The missingness of `avg_rating` is independent of `year_submitted`. The average submission year is the same for recipes with and without a rating, up to random chance.
- **Alternative Hypothesis (H₁):** The missingness of `avg_rating` depends on `year_submitted`; recipes with missing ratings were submitted in different years on average than those with non-missing ratings.
- **Test statistic:** Absolute difference in mean `year_submitted` between missing vs non-missing recipes.
- **Result:** p-value ≈ 0.0
- **Conclusion:** We reject the null hypothesis. The missingness of `avg_rating` depends on `year_submitted`, suggesting that recipes submitted in different years have different rates of missing ratings.

**Missingness of `avg_rating` vs `n_ingredients`:**

- **Null Hypothesis (H₀):** The missingness of `avg_rating` is independent of `n_ingredients`. The average number of ingredients is the same for recipes with and without a rating, up to random chance.
- **Alternative Hypothesis (H₁):** The missingness of `avg_rating` depends on `n_ingredients`; recipes with missing ratings have a different average number of ingredients.
- **Test statistic:** Absolute difference in mean `n_ingredients` between missing vs non-missing recipes.
- **Result:** p-value ≈ 0.001
- **Conclusion:** We reject the null hypothesis. The missingness of `avg_rating` depends on `n_ingredients`, indicating that recipes with different numbers of ingredients have different rates of missing ratings.

---

## Hypothesis Testing

We performed a hypothesis test to investigate whether recipe preparation time is associated with average ratings.

**Research Question:** Is the average rating for recipes with prep time ≤ 30 minutes (quick) different from those with prep time > 30 minutes (long)?

- **Null Hypothesis (H₀):** The average rating of quick recipes is the same as the average rating of long recipes; any observed difference is due to chance.
- **Alternative Hypothesis (H₁):** The average rating of quick recipes is different from the average rating of long recipes.
- **Test statistic:** Absolute difference in mean `avg_rating` between quick and long recipes.
- **Significance level:** α = 0.05
- **Result:** p-value ≈ 0.0000
- **Conclusion:** We reject the null hypothesis at the 0.05 significance level. There is sufficient evidence to suggest that the average rating for quick recipes (≤ 30 minutes) is different from that of long recipes (> 30 minutes). The observed difference in means was 0.0350, with quick recipes having slightly higher average ratings.

<iframe
  src="assets/prep_time_perm_test.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The permutation distribution shows that the observed test statistic (0.0350) is far in the tail of the null distribution, providing strong evidence against the null hypothesis.

---

## Framing a Prediction Problem

**Prediction Task:** We want to predict the **average rating** of a recipe (`avg_rating`) on Food.com using information about the recipe itself.

**Response Variable and Type:**
- Response variable: `avg_rating` (mean of non-zero ratings per recipe)
- This is a **regression** problem, since `avg_rating` is numeric and typically between 1 and 5.

**Features and Time of Prediction:**
At prediction time, we assume a new recipe has just been posted, so we can only use columns from the recipes dataset that are known when the recipe is created, such as:
- `minutes`: Preparation time
- `n_steps`: Number of steps
- `n_ingredients`: Number of ingredients
- `calories`, `protein_pdv`, `carbs_pdv`: Nutritional information
- `log_minutes`: Log-transformed preparation time

We do **not** use anything that depends on future user behavior, like ratings or number of reviews, because that information is not available when the recipe is first posted.

**Evaluation Metric:**
We use **RMSE** (Root Mean Squared Error) on a held-out test set. RMSE measures how far our predicted ratings are from the true ratings in the same units (stars) and penalizes large errors more than small ones, making it appropriate for regression tasks where we care about the magnitude of prediction errors.

---

## Baseline Model

Our baseline model predicts `avg_rating` using two simple quantitative features:
- `minutes`: Minutes required to prepare the recipe
- `n_steps`: Number of steps in the recipe

We use a **Linear Regression** model with **StandardScaler** preprocessing, implemented in a single sklearn Pipeline. The model is trained on 80% of the data and evaluated on the remaining 20% test set.

**Model Features:**
- **Quantitative features:** 2 (`minutes`, `n_steps`)
- **Encoding:** StandardScaler applied to both features

**Performance:**
- **Test RMSE:** 0.63657

**Assessment:**
The baseline model achieves a test RMSE of approximately 0.64 stars. Given that ratings range from 1 to 5, this represents a moderate level of error. The model captures some relationship between recipe characteristics and ratings, but there is substantial room for improvement. The relatively high RMSE suggests that preparation time and number of steps alone are not sufficient to accurately predict recipe ratings, and additional features may be necessary.

---

## Final Model

To improve upon the baseline, we engineered additional features and tested multiple modeling algorithms:

**New Features Added:**
1. **`n_ingredients`**: The number of ingredients in the recipe. This captures recipe complexity and may relate to perceived quality or effort.
2. **`log_minutes`**: Log-transformed preparation time. This helps handle the long tail in the minutes distribution and may capture non-linear relationships with ratings.
3. **`calories`**: Total calories per serving. Nutritional content may influence how users perceive and rate recipes.
4. **`protein_pdv`** and **`carbs_pdv`**: Percent daily values for protein and carbohydrates. These provide additional nutritional context that may affect ratings.

**Model Selection:**
We tested three candidate models using GridSearchCV with 5-fold cross-validation:
- **RandomForestRegressor**: Tuned `n_estimators` (100, 200) and `max_depth` (5, 10, None)
- **Ridge Regression**: Tuned `alpha` (0.01, 0.1, 1.0, 10.0)
- **Lasso Regression**: Tuned `alpha` (0.001, 0.01, 0.1, 1.0, 10.0)

**Final Model Choice:**
**Lasso Regression** performed best with the following hyperparameters:
- `alpha`: 0.1
- All features standardized using StandardScaler

**Performance:**
- **Test RMSE:** 0.6540
- **Baseline RMSE:** 0.63657

**Improvement Analysis:**
While the final model's RMSE (0.6540) is slightly higher than the baseline (0.63657), this small difference suggests that the additional features provide limited predictive power beyond preparation time and number of steps. The Lasso model's L1 regularization helps prevent overfitting and may provide more stable predictions. The engineered features (ingredient count, nutritional information, and log-transformed time) capture additional aspects of recipe complexity and nutritional content, which could be valuable for understanding recipe characteristics even if they don't dramatically improve prediction accuracy.

The model's performance indicates that predicting recipe ratings is challenging, as ratings likely depend on factors not captured in our dataset, such as taste, presentation, user preferences, and recipe execution quality.

---

## Fairness Analysis

We performed a fairness analysis to determine whether our final model performs differently for different groups of recipes.

**Groups:**
- **Group X (Quick recipes):** Recipes with `minutes ≤ 30`
- **Group Y (Long recipes):** Recipes with `minutes > 30`

**Evaluation Metric:**
Since this is a regression problem, we use **RMSE** to compare model performance across groups.

**Hypotheses:**
- **Null Hypothesis (H₀):** The model is fair with respect to recipe length. Its RMSE for quick recipes and long recipes is roughly the same; any observed difference is due to random chance.
- **Alternative Hypothesis (H₁):** The model is unfair in the sense that it performs worse on quick recipes; specifically, the RMSE for quick recipes is **larger** than the RMSE for long recipes.

**Test Statistic:**
Difference in RMSE between quick and long recipes (RMSE_quick - RMSE_long). Large positive values indicate worse performance on quick recipes.

**Results:**
- **RMSE (quick):** 0.6031
- **RMSE (long):** 0.6617
- **Observed statistic:** RMSE_quick - RMSE_long = -0.0586
- **p-value:** ≈ 0.84
- **Significance level:** α = 0.05

**Conclusion:**
We fail to reject the null hypothesis at the 0.05 significance level. The observed difference in RMSE is actually negative (-0.0586), meaning the model performs slightly **better** on quick recipes than on long recipes. The permutation test yields a p-value of approximately 0.84, indicating that the observed difference is consistent with random chance. We do not have sufficient evidence to conclude that the model is unfair with respect to recipe preparation time.

<iframe
  src="assets/fairness_rmse_perm.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The permutation distribution shows that the observed test statistic (-0.0586) falls well within the null distribution, providing no evidence of unfairness.

---

## Conclusion

This project explored factors influencing recipe ratings on Food.com and built predictive models to estimate average ratings. Our analysis revealed that recipe characteristics like preparation time, number of steps, and number of ingredients are associated with ratings, though the relationships are relatively weak. The final Lasso regression model achieved moderate prediction accuracy, suggesting that recipe ratings depend on factors beyond those captured in our dataset. Our fairness analysis found no evidence that the model performs differently for quick versus long recipes, indicating that the model treats both groups similarly.

Future work could incorporate additional features such as recipe descriptions (using natural language processing), user engagement metrics, or recipe presentation quality to improve prediction accuracy.

