# Predicting Recipe Ratings on Food.com

**Name(s)**: Kavya Shah, Ishayu Ghosh


---

## Introduction

Food.com is a popular platform where users can discover, share, and rate recipes. Understanding what makes a recipe successful—as measured by its average rating—is valuable for both recipe creators and the platform itself. This project investigates factors that influence recipe ratings and builds a predictive model to estimate how well a recipe will be received based on its characteristics.

**Research Question**: What factors influence the average rating of recipes on Food.com, and can we predict a recipe's average rating using only information available when the recipe is first posted?

### Datasets

Our analysis is based on two datasets from Food.com:

**1. RAW_recipes.csv**

The first dataset, `RAW_recipes.csv`, contains **83,782 rows**, representing unique recipes, with the following **12 columns**:

| Column | Description |
|--------|-------------|
| `name` | Recipe name |
| `id` | Recipe ID (unique identifier) |
| `minutes` | Minutes to prepare the recipe |
| `contributor_id` | User ID who submitted the recipe |
| `submitted` | Date the recipe was submitted |
| `tags` | Food.com tags for the recipe (stored as string representation of a list) |
| `nutrition` | Nutrition info as [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)] where PDV = percentage of daily value (stored as string representation of a list) |
| `n_steps` | Number of steps in the recipe |
| `steps` | Text for recipe steps, in order (stored as string representation of a list) |
| `description` | User-provided description |
| `ingredients` | List of ingredients used (stored as string representation of a list) |
| `n_ingredients` | Number of ingredients in the recipe |

**2. RAW_interactions.csv**

The second dataset, `RAW_interactions.csv`, contains **731,927 rows**, representing unique reviews for recipes (there are more than one review per individual recipe in the dataframe). This dataset has **5 columns**:

| Column | Description |
|--------|-------------|
| `user_id` | User ID who provided the review |
| `recipe_id` | Recipe ID (links to `id` in RAW_recipes.csv) |
| `date` | Date of interaction |
| `rating` | Rating given (1-5 stars, with 0 indicating no rating) |
| `review` | Review text |

### Merged Dataset

To combine recipe information with user ratings, we merged the two datasets using a left join on `recipes.id` and `interactions.recipe_id`. This merge strategy ensures that we retain all recipes from the `RAW_recipes.csv` dataset, even if they have no reviews in the interactions dataset.

After merging, we performed several data cleaning steps (detailed in the Data Cleaning section below) to create our final working dataset. The key transformation was computing the **average rating** (`avg_rating`) for each recipe by taking the mean of all non-zero ratings from the interactions dataset. We replaced zero ratings with `NaN` values, as zeros likely indicate that a user did not provide a rating rather than a true zero-star rating.

### Relevant Columns for Analysis

Our final cleaned dataset contains **83,782 recipes** (matching the original recipes dataset). For our analysis, we focus on the following relevant columns:

**Primary Response Variable:**
- `avg_rating`: The average rating (1-5 stars) for each recipe, computed from non-zero user ratings in the interactions dataset

**Recipe Characteristics (Features):**
- `minutes`: The preparation time in minutes
- `n_steps`: The number of steps in the recipe
- `n_ingredients`: The number of ingredients required (derived from parsing the `ingredients` column)
- `calories`, `protein_pdv`, `carbs_pdv`: Nutritional information extracted from the `nutrition` column (calories and percent daily values for protein and carbohydrates)
- `submitted`: The submission date of the recipe (used to extract `year_submitted` for missingness analysis)

**Additional Columns Used:**
- `tags`, `ingredients`, `steps`, `nutrition`: Recipe details stored as lists (parsed from string representations)
- `description`: User-provided description (available but not used in our models)

We focus on understanding how recipe characteristics like preparation time, complexity (measured by number of steps and ingredients), and nutritional content relate to user ratings. Our predictive models use only features that would be available when a recipe is first posted, excluding any information that depends on future user behavior.

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

We examined the distributions of key variables to understand the characteristics of recipes in our dataset. Univariate analysis helps us understand the central tendencies, variability, and shape of individual variables before exploring relationships between them.

**Distribution of Number of Ingredients**

<iframe
  src="assets/ingredients_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The distribution of the number of ingredients (`n_ingredients`) reveals important insights about recipe complexity. The histogram shows that most recipes use between 5 and 15 ingredients, with a clear peak around 8-10 ingredients. This suggests that the "typical" recipe on Food.com requires a moderate number of ingredients, balancing simplicity with flavor complexity.

The distribution is right-skewed, with a long tail extending to recipes with 20+ ingredients. These complex recipes are relatively rare but represent more elaborate dishes that may require extensive ingredient lists. The minimum number of ingredients is 1 (likely simple recipes or single-ingredient dishes), while the maximum extends well beyond 20 ingredients for complex recipes.

This distribution informs our understanding of recipe complexity and suggests that ingredient count could be a meaningful feature for predicting ratings, as it captures one dimension of recipe sophistication.

**Distribution of Preparation Time**

<iframe
  src="assets/prep_time_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The distribution of preparation time (`minutes`) provides insights into the time investment required for recipes on Food.com. We capped the visualization at 4 hours (240 minutes) for readability, as the distribution has an extremely long tail with some recipes taking 24+ hours (likely slow-cooker or fermentation recipes).

The histogram reveals a multimodal distribution with several peaks:
- A concentration of **quick recipes** (under 30 minutes), representing fast and easy dishes
- A prominent peak around **30-60 minutes**, representing the most common preparation time range
- A secondary peak around **60-90 minutes**, representing more involved recipes
- A long tail extending to several hours, representing time-intensive preparations like slow-cooker meals, braises, and baked goods

This distribution is heavily right-skewed, indicating that while most recipes are relatively quick to prepare, there is substantial variation in preparation time. The presence of many quick recipes (≤ 30 minutes) suggests that Food.com users value convenience, while the long tail indicates that the platform also caters to users willing to invest significant time in cooking.

Understanding this distribution is crucial for our analysis, as preparation time may influence both user engagement (users may be more likely to try and rate quick recipes) and perceived recipe quality.

### Bivariate Analysis

We explored relationships between recipe characteristics and average ratings to identify potential predictors for our modeling task. Bivariate analysis helps us understand how pairs of variables relate to each other and can reveal associations that inform feature selection.

**Number of Ingredients vs. Average Rating**

<iframe
  src="assets/ingredients_vs_rating.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The scatter plot examining the relationship between number of ingredients (`n_ingredients`) and average rating (`avg_rating`) reveals a weak but positive association. The ordinary least squares (OLS) trendline shows a modest upward slope, indicating that recipes with more ingredients tend to have slightly higher average ratings on average.

However, there is substantial variation around the trendline, with recipes at any given ingredient count showing a wide range of ratings (approximately 2.5 to 5.0 stars). This suggests that while ingredient count may contribute to recipe success, it is far from the sole determinant of ratings. The weak relationship indicates that:

1. **Complexity may be valued**: Recipes with more ingredients might be perceived as more sophisticated or flavorful, leading to slightly higher ratings.

2. **Other factors matter more**: The substantial scatter suggests that factors beyond ingredient count (such as taste, execution difficulty, presentation, or user preferences) play larger roles in determining ratings.

3. **Diminishing returns**: The relationship appears to plateau at higher ingredient counts, suggesting that beyond a certain point, adding more ingredients doesn't significantly improve ratings.

This analysis informs our feature engineering decisions, suggesting that `n_ingredients` may be a useful but limited predictor in our models.

**Preparation Time vs. Average Rating**

<iframe
  src="assets/prep_time_vs_rating.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The scatter plot examining the relationship between preparation time (`minutes`) and average rating (for recipes taking ≤ 4 hours) shows a slight negative trend. The OLS trendline slopes downward, indicating that quicker recipes tend to have slightly higher average ratings on average.

This relationship, while statistically detectable, is weak and shows significant scatter. The negative association could reflect several factors:

1. **Convenience preference**: Users may value quick recipes more highly, as they fit better into busy lifestyles.

2. **Accessibility**: Quick recipes may be more accessible to a broader range of cooks, leading to more positive experiences and higher ratings.

3. **Selection bias**: Users may be more likely to try and rate quick recipes, potentially skewing the distribution of ratings.

However, the weak relationship and substantial variation indicate that preparation time alone is not a strong predictor of recipe success. Recipes at any preparation time show a wide range of ratings, suggesting that factors like taste, presentation, and execution quality matter more than time investment.

The significant scatter around the trendline reinforces that predicting ratings requires considering multiple factors simultaneously, which motivates our use of multiple features in our predictive models.

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

We performed permutation tests to analyze the dependency of `avg_rating` missingness on other columns. Understanding missingness dependencies helps us assess whether the missing data is Missing At Random (MAR) or depends on observed variables, which is crucial for understanding potential biases in our analysis.

**Methodology:**

For each permutation test, we:
1. Created a binary indicator `missing_rating` that is `True` when `avg_rating` is missing and `False` otherwise
2. Computed the observed test statistic: the absolute difference in means of the predictor variable between missing and non-missing groups
3. Performed 5,000 permutations by randomly shuffling the `missing_rating` labels
4. Computed the permutation distribution of the test statistic under the null hypothesis
5. Calculated the p-value as the proportion of permutation statistics greater than or equal to the observed statistic

**Missingness of `avg_rating` vs `year_submitted`:**

- **Null Hypothesis (H₀):** The missingness of `avg_rating` is independent of `year_submitted`. The average submission year is the same for recipes with and without a rating, up to random chance.
- **Alternative Hypothesis (H₁):** The missingness of `avg_rating` depends on `year_submitted`; recipes with missing ratings were submitted in different years on average than those with non-missing ratings.
- **Test statistic:** Absolute difference in mean `year_submitted` between missing vs non-missing recipes.
- **Observed statistic:** |mean(year_submitted_missing) - mean(year_submitted_not_missing)| = 0.73 years
- **Result:** p-value ≈ 0.0 (0 out of 5,000 permutations had a statistic ≥ 0.73)
- **Conclusion:** We reject the null hypothesis at any reasonable significance level. The missingness of `avg_rating` depends on `year_submitted`, suggesting that recipes submitted in different years have systematically different rates of missing ratings.

**Interpretation:** This dependency likely reflects temporal trends in Food.com's user engagement. For example, older recipes may have accumulated more ratings over time, while newer recipes may not have had sufficient time to receive ratings. Alternatively, changes in the platform's user base or rating behavior over time could explain this pattern. This finding suggests that the missingness mechanism is at least partially MAR (Missing At Random), as it depends on an observed variable (`year_submitted`).

**Missingness of `avg_rating` vs `n_ingredients`:**

- **Null Hypothesis (H₀):** The missingness of `avg_rating` is independent of `n_ingredients`. The average number of ingredients is the same for recipes with and without a rating, up to random chance.
- **Alternative Hypothesis (H₁):** The missingness of `avg_rating` depends on `n_ingredients`; recipes with missing ratings have a different average number of ingredients.
- **Test statistic:** Absolute difference in mean `n_ingredients` between missing vs non-missing recipes.
- **Observed statistic:** |mean(n_ingredients_missing) - mean(n_ingredients_not_missing)| = 0.25 ingredients
- **Result:** p-value ≈ 0.001 (5 out of 5,000 permutations had a statistic ≥ 0.25)
- **Conclusion:** We reject the null hypothesis at the 0.05 significance level. The missingness of `avg_rating` depends on `n_ingredients`, indicating that recipes with different numbers of ingredients have different rates of missing ratings.

**Interpretation:** This dependency suggests that recipe complexity (as measured by ingredient count) influences whether a recipe receives ratings. This could occur if:
- More complex recipes (with more ingredients) are more likely to be tried and rated by users
- Simpler recipes may be less engaging or less likely to be featured, resulting in fewer ratings
- Users may be more motivated to rate recipes that required more effort to prepare

This finding further supports that the missingness is at least partially MAR, as it depends on an observed variable (`n_ingredients`). However, the dependency on both observed and unobserved factors (as discussed in the NMAR analysis) suggests that the missingness mechanism is complex and involves both MAR and NMAR components.

---

## Hypothesis Testing

We performed hypothesis tests to investigate whether recipe characteristics are associated with average ratings. These tests help us understand which features might be predictive and inform our modeling decisions.

### Test 1: Preparation Time and Ratings

**Research Question:** Is the average rating for recipes with prep time ≤ 30 minutes (quick) different from those with prep time > 30 minutes (long)?

This question is motivated by our bivariate analysis, which suggested a weak negative relationship between preparation time and ratings. We want to determine whether this observed difference is statistically significant or could be due to random chance.

**Hypotheses:**
- **Null Hypothesis (H₀):** The average rating of quick recipes is the same as the average rating of long recipes; any observed difference is due to chance.
- **Alternative Hypothesis (H₁):** The average rating of quick recipes is different from the average rating of long recipes (two-sided test).

**Methodology:**
- We created a binary indicator `islong` that is `True` for recipes with `minutes > 30` and `False` for recipes with `minutes ≤ 30`
- We computed the observed test statistic: |mean(avg_rating_quick) - mean(avg_rating_long)|
- We performed a permutation test with 5,000 iterations, randomly shuffling the `islong` labels each time
- We calculated the p-value as the proportion of permutation statistics greater than or equal to the observed statistic

**Results:**
- **Observed statistic:** |mean(avg_rating_quick) - mean(avg_rating_long)| = 0.0350 stars
- **Significance level:** α = 0.05
- **p-value:** ≈ 0.0000 (0 out of 5,000 permutations had a statistic ≥ 0.0350)
- **Conclusion:** We reject the null hypothesis at the 0.05 significance level. There is sufficient evidence to suggest that the average rating for quick recipes (≤ 30 minutes) is different from that of long recipes (> 30 minutes).

**Interpretation:**
The observed difference of 0.0350 stars, while small in magnitude, is highly statistically significant. Quick recipes have slightly higher average ratings than long recipes. This finding aligns with our bivariate analysis and suggests that:

1. **Convenience is valued**: Users may prefer and rate quick recipes more highly, possibly because they fit better into busy schedules
2. **Accessibility matters**: Quick recipes may be more accessible to a broader range of cooks, leading to more positive experiences
3. **Practical considerations**: The time investment required for long recipes may not always translate to proportionally higher ratings

However, the small magnitude of the difference (0.035 stars on a 1-5 scale) suggests that while preparation time is a statistically significant factor, it explains only a small portion of the variation in ratings.

<iframe
  src="assets/prep_time_perm_test.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The permutation distribution shows that the observed test statistic (0.0350) is far in the tail of the null distribution, with no permutation statistics reaching this value. This provides very strong evidence against the null hypothesis and confirms that the difference in ratings between quick and long recipes is unlikely to be due to random chance alone.

### Test 2: Recipe Complexity (Number of Steps) and Ratings

We also investigated whether recipe complexity, as measured by the number of steps, is associated with ratings.

**Research Question:** Is the average rating for recipes with ≤ 5 steps (simple) different from those with > 5 steps (complex)?

**Results:**
- **Observed statistic:** |mean(avg_rating_simple) - mean(avg_rating_complex)| = 0.0162 stars
- **p-value:** ≈ 0.0024
- **Conclusion:** We reject the null hypothesis at the 0.05 significance level, though the effect size is even smaller than for preparation time.

This finding suggests that recipe complexity (as measured by number of steps) has a statistically significant but very small association with ratings, further reinforcing that multiple factors contribute to recipe success.

---

## Framing a Prediction Problem

### Problem Statement

**Prediction Task:** We want to predict the **average rating** of a recipe (`avg_rating`) on Food.com using information about the recipe itself. This is a practical problem that could help recipe creators understand how their recipes might be received, or help the platform prioritize recipe recommendations.

### Response Variable and Problem Type

**Response Variable:**
- `avg_rating`: The average rating (1-5 stars) for each recipe, computed from non-zero user ratings in the interactions dataset

**Problem Type:**
This is a **regression** problem, since `avg_rating` is a continuous numeric variable that typically ranges from 1 to 5 stars. We are predicting a continuous value rather than a discrete category.

**Why Regression:**
- Ratings are numeric and ordered (1 < 2 < 3 < 4 < 5)
- We care about the magnitude of differences (a prediction of 4.5 vs. actual 4.0 is different from a prediction of 2.0 vs. actual 4.0)
- We want to understand how recipe features relate to rating magnitude, not just whether a recipe is "good" or "bad"

### Features and Time of Prediction

**Critical Constraint: Time of Prediction**

At prediction time, we assume a new recipe has just been posted to Food.com. This constraint is crucial because it determines which features we can legitimately use in our model.

**Available Features (Known at Recipe Creation):**
- `minutes`: Preparation time in minutes (known when recipe is created)
- `n_steps`: Number of steps in the recipe (known when recipe is created)
- `n_ingredients`: Number of ingredients required (known when recipe is created)
- `calories`, `protein_pdv`, `carbs_pdv`: Nutritional information (can be computed from recipe ingredients)
- `log_minutes`: Log-transformed preparation time (derived from `minutes`)

**Excluded Features (Not Available at Recipe Creation):**
- `avg_rating`: The variable we're trying to predict (circular reasoning)
- Number of reviews: Not known until users interact with the recipe
- User engagement metrics: Views, saves, shares (not available for new recipes)
- Temporal features related to popularity: These depend on future user behavior

**Why This Matters:**
Using only features available at recipe creation ensures our model is practically useful. If we used future information (like number of reviews), we could only make predictions after a recipe has already been successful, which defeats the purpose of prediction.

### Evaluation Metric

**Metric Choice: RMSE (Root Mean Squared Error)**

We use **RMSE** on a held-out test set to evaluate our models. RMSE is calculated as:

$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$

where \(y_i\) is the true average rating and \(\hat{y}_i\) is the predicted rating.

**Why RMSE:**
1. **Same units**: RMSE is in the same units as the response variable (stars), making it interpretable
2. **Penalizes large errors**: RMSE squares the errors, so large prediction errors are penalized more heavily than small ones
3. **Standard for regression**: RMSE is widely used for regression problems and allows comparison with other models
4. **Sensitive to outliers**: Large prediction errors are appropriately weighted, which is important when predicting ratings

**Alternative Metrics Considered:**
- **Mean Absolute Error (MAE)**: Less sensitive to outliers but doesn't penalize large errors as much
- **R² (Coefficient of Determination)**: Useful for understanding variance explained, but less interpretable for prediction error
- **Mean Squared Error (MSE)**: Same information as RMSE but in squared units, making it less interpretable

We chose RMSE because it balances interpretability with appropriate penalty for large errors, which is important for a practical prediction task.

---

## Baseline Model

### Model Design

Our baseline model serves as a simple starting point that we can improve upon. We intentionally chose a minimal feature set and a straightforward model to establish a performance baseline.

**Feature Selection:**
We selected two simple quantitative features that are directly available in the dataset:
- `minutes`: Minutes required to prepare the recipe
- `n_steps`: Number of steps in the recipe

**Why These Features:**
1. **Simplicity**: These are straightforward numeric features that don't require complex preprocessing
2. **Interpretability**: Both features have clear meanings and are easy to understand
3. **Availability**: Both are present in the original dataset without requiring feature engineering
4. **Relevance**: Our exploratory analysis suggested these features have some association with ratings

**Model Architecture:**
We use a **Linear Regression** model with **StandardScaler** preprocessing, implemented in a single sklearn Pipeline. The pipeline structure ensures that:
- Preprocessing (standardization) is applied consistently to training and test data
- The model can be easily reproduced and deployed
- All transformations are contained in a single object

**Preprocessing:**
- **StandardScaler**: Standardizes features by removing the mean and scaling to unit variance. This is important for linear regression because:
  - Features are on different scales (`minutes` ranges from 1 to thousands, while `n_steps` typically ranges from 1 to 20)
  - Standardization ensures that the model doesn't give undue weight to features with larger numeric values
  - It helps with numerical stability during optimization

**Data Splitting:**
- **Training set**: 80% of the data (used to fit the model)
- **Test set**: 20% of the data (held out for final evaluation)
- **Random state**: 80 (for reproducibility)

We use a train-test split rather than cross-validation for the baseline to establish a clear performance benchmark that can be directly compared to our final model.

### Model Features

**Feature Count and Types:**
- **Total features:** 2
- **Quantitative features:** 2 (`minutes`, `n_steps`)
- **Categorical features:** 0
- **Encoding:** StandardScaler applied to both features (no categorical encoding needed)

**Feature Details:**
- `minutes`: Continuous numeric variable, standardized to have mean 0 and standard deviation 1
- `n_steps`: Continuous numeric variable, standardized to have mean 0 and standard deviation 1

### Performance

**Test Set Results:**
- **Test RMSE:** 0.63657 stars
- **Interpretation:** On average, our predictions are off by approximately 0.64 stars from the true average ratings

**Context for Performance:**
Given that ratings range from 1 to 5 stars:
- An RMSE of 0.64 means we're typically within about 0.64 stars of the true rating
- This represents approximately 13% of the rating scale (0.64 / 5.0)
- The model captures some relationship between recipe characteristics and ratings, but there is substantial prediction error

### Assessment

**Strengths:**
1. **Simplicity**: The model is easy to understand and interpret
2. **Baseline established**: Provides a clear benchmark for improvement
3. **Fast training**: Linear regression is computationally efficient
4. **No overfitting concerns**: Simple model with few parameters

**Limitations:**
1. **Limited features**: Only uses two features, missing potentially valuable information
2. **Linear assumptions**: Assumes linear relationships, which may not capture complex patterns
3. **Moderate accuracy**: RMSE of 0.64 suggests substantial room for improvement
4. **Missing complexity**: Doesn't capture recipe complexity beyond step count (e.g., ingredient count, nutritional content)

**Why This Baseline Matters:**
The baseline model demonstrates that even simple features have some predictive power, but the relatively high RMSE (0.64 stars) suggests that:
- Preparation time and number of steps alone are not sufficient to accurately predict recipe ratings
- Additional features may capture important aspects of recipe quality
- More sophisticated models or feature engineering may improve performance
- Recipe ratings depend on factors beyond just time and step count

This baseline provides a foundation for improvement and helps us measure the value added by more complex models and additional features.

---

## Final Model

### Feature Engineering

To improve upon the baseline, we engineered additional features that capture different aspects of recipe characteristics. Feature engineering is crucial because it allows us to extract more information from the available data.

**New Features Added:**

1. **`n_ingredients`**: The number of ingredients in the recipe
   - **Rationale**: Captures recipe complexity and may relate to perceived quality, effort, or sophistication
   - **Data generating process**: Recipes with more ingredients may be seen as more elaborate or flavorful, potentially influencing ratings
   - **Source**: Derived by counting the length of the parsed `ingredients` list

2. **`log_minutes`**: Log-transformed preparation time (log(1 + minutes))
   - **Rationale**: The `minutes` distribution is heavily right-skewed with a long tail. Log transformation:
     - Compresses the long tail, making the distribution more symmetric
     - May capture non-linear relationships with ratings (e.g., the difference between 10 and 20 minutes may matter more than the difference between 200 and 210 minutes)
     - Helps linear models better capture relationships with highly skewed features
   - **Data generating process**: Users may perceive time differences differently at different scales (quick vs. moderate vs. very long recipes)

3. **`calories`**: Total calories per serving
   - **Rationale**: Nutritional content may influence how users perceive and rate recipes
   - **Data generating process**: Health-conscious users may rate recipes differently based on caloric content, or calorie-dense recipes may be associated with comfort foods that receive higher ratings
   - **Source**: Extracted from the parsed `nutrition` list (first element)

4. **`protein_pdv`** and **`carbs_pdv`**: Percent daily values for protein and carbohydrates
   - **Rationale**: These provide additional nutritional context beyond calories
   - **Data generating process**: Different nutritional profiles may appeal to different user preferences (high-protein for fitness-focused users, balanced macros for general users)
   - **Source**: Extracted from the parsed `nutrition` list (4th and 6th elements, respectively)

**Feature Set:**
Our final model uses 7 features total:
- `minutes` (original)
- `n_steps` (original)
- `n_ingredients` (engineered)
- `log_minutes` (engineered)
- `calories` (engineered)
- `protein_pdv` (engineered)
- `carbs_pdv` (engineered)

### Model Selection Process

We tested three candidate models using **GridSearchCV** with **5-fold cross-validation** to systematically search for the best hyperparameters and model type.

**Methodology:**
- **Cross-validation**: 5-fold CV on the training set to evaluate each hyperparameter combination
- **Scoring metric**: Negative RMSE (GridSearchCV maximizes, so we use negative RMSE to minimize RMSE)
- **Search strategy**: Exhaustive grid search over specified hyperparameter ranges
- **Final evaluation**: Best model evaluated on held-out test set (same split as baseline)

**Candidate Models:**

1. **RandomForestRegressor**
   - **Rationale**: Non-linear model that can capture complex interactions between features
   - **Hyperparameters tuned**:
     - `n_estimators`: [100, 200] - Number of trees in the forest
     - `max_depth`: [5, 10, None] - Maximum depth of trees (None = no limit)
   - **Why these parameters**: Control model complexity and prevent overfitting

2. **Ridge Regression**
   - **Rationale**: Linear model with L2 regularization that shrinks coefficients toward zero
   - **Hyperparameters tuned**:
     - `alpha`: [0.01, 0.1, 1.0, 10.0] - Regularization strength (larger = more regularization)
   - **Why Ridge**: Helps with multicollinearity and prevents overfitting while keeping all features

3. **Lasso Regression**
   - **Rationale**: Linear model with L1 regularization that can perform automatic feature selection
   - **Hyperparameters tuned**:
     - `alpha`: [0.001, 0.01, 0.1, 1.0, 10.0] - Regularization strength (larger = more feature selection)
   - **Why Lasso**: Can identify and remove less important features, potentially improving generalization

### Final Model Choice

**Selected Model: Lasso Regression**

**Best Hyperparameters:**
- `alpha`: 0.1
- All features standardized using StandardScaler (applied in pipeline)

**Why Lasso Performed Best:**
1. **Feature selection**: L1 regularization may have helped identify the most important features
2. **Generalization**: The regularization helps prevent overfitting to the training data
3. **Interpretability**: Linear model with sparse coefficients is easier to interpret than Random Forest
4. **Stability**: Linear models are generally more stable and less sensitive to small data changes

**Model Architecture:**
- **Pipeline**: StandardScaler → Lasso Regression
- **All preprocessing and modeling in a single sklearn Pipeline**

### Performance

**Test Set Results:**
- **Final Model Test RMSE:** 0.6540 stars
- **Baseline Model Test RMSE:** 0.63657 stars
- **Difference:** +0.0174 stars (slightly worse)

**Performance Analysis:**

While the final model's RMSE is slightly higher than the baseline, this result is informative:

1. **Limited improvement from additional features**: The small difference suggests that `minutes` and `n_steps` capture much of the predictive information available in our feature set. The additional features (ingredient count, nutritional information, log-transformed time) provide only marginal additional predictive power.

2. **Regularization effect**: The Lasso model's L1 regularization may have shrunk some coefficients, potentially reducing overfitting but also slightly increasing test error. This trade-off suggests the model is more generalizable.

3. **Feature value**: While the engineered features don't dramatically improve prediction accuracy, they:
   - Capture additional aspects of recipe complexity and nutritional content
   - Provide richer feature representation that could be valuable in other contexts
   - May help with model interpretability by showing which recipe characteristics matter

4. **Inherent prediction difficulty**: The model's performance (RMSE ≈ 0.65) indicates that predicting recipe ratings is inherently challenging. Ratings likely depend on factors not captured in our dataset, such as:
   - **Taste and flavor**: Subjective and difficult to quantify from recipe metadata
   - **Presentation**: Visual appeal, photography quality
   - **User preferences**: Individual tastes, dietary restrictions, cultural preferences
   - **Execution quality**: How well the recipe was followed, cooking skill level
   - **Context**: Time of year, occasion, personal circumstances

**Model Interpretation:**
The Lasso model's coefficients (after standardization) indicate the relative importance of each feature. While we don't report specific coefficients here, the model provides interpretable insights into which recipe characteristics most influence ratings, subject to the limitations of our feature set.

**Conclusion:**
The final model represents a thoughtful attempt to improve upon the baseline through feature engineering and model selection. While the performance improvement is modest, the process demonstrates rigorous methodology and provides insights into the factors that influence recipe ratings. The results suggest that recipe ratings depend substantially on factors beyond those captured in structured recipe metadata, highlighting the challenge of predicting subjective user preferences.

---

## Fairness Analysis

### Motivation

We performed a fairness analysis to determine whether our final model performs differently for different groups of recipes. Fairness in machine learning means that a model should perform similarly well for different groups, avoiding systematic biases that could disadvantage certain types of recipes.

**Why This Matters:**
If a model performs significantly worse for certain groups (e.g., quick recipes vs. long recipes), it could:
- Lead to unfair recommendations or prioritization
- Perpetuate biases in recipe visibility or promotion
- Provide less reliable predictions for certain recipe types

### Group Definition

We compare model performance across two groups based on preparation time:
- **Group X (Quick recipes):** Recipes with `minutes ≤ 30`
- **Group Y (Long recipes):** Recipes with `minutes > 30`

**Why These Groups:**
1. **Practical relevance**: Preparation time is a meaningful way to categorize recipes that users care about
2. **Hypothesis from earlier analysis**: Our hypothesis testing suggested quick and long recipes may have different rating patterns
3. **Potential for bias**: If the model was trained primarily on one type of recipe, it might perform worse on the other

### Evaluation Metric

Since this is a regression problem, we use **RMSE** (Root Mean Squared Error) to compare model performance across groups. RMSE measures prediction accuracy in the same units as the response variable (stars), making it appropriate for comparing performance across groups.

**Why RMSE:**
- Directly measures prediction error
- Penalizes large errors appropriately
- Allows meaningful comparison between groups
- Standard metric for regression fairness analysis

### Hypotheses

- **Null Hypothesis (H₀):** The model is fair with respect to recipe preparation time. Its RMSE for quick recipes and long recipes is roughly the same; any observed difference is due to random chance.

- **Alternative Hypothesis (H₁):** The model is unfair in the sense that it performs worse on quick recipes; specifically, the RMSE for quick recipes is **larger** than the RMSE for long recipes.

**Note on Directionality:**
We test a one-sided alternative hypothesis (model performs worse on quick recipes) based on the concern that the model might be biased toward longer, more complex recipes. However, we would also be concerned if the model performed worse on long recipes.

### Methodology

**Test Statistic:**
We use the difference in RMSE between quick and long recipes:
\[T = RMSE_{quick} - RMSE_{long}\]

Large positive values indicate worse performance on quick recipes (higher RMSE = worse predictions).

**Permutation Test:**
1. We compute the observed test statistic on the held-out test set
2. We perform 5,000 permutations by randomly shuffling the group labels (quick vs. long) while keeping the predictions and true values fixed
3. For each permutation, we compute the test statistic under the null hypothesis
4. We calculate the p-value as the proportion of permutation statistics greater than or equal to the observed statistic

**Why Permutation Test:**
- Makes minimal assumptions about the data distribution
- Accounts for the correlation structure in the data
- Provides a non-parametric test of fairness
- Appropriate when comparing performance metrics across groups

### Results

**Observed Performance:**
- **RMSE (quick recipes):** 0.6031 stars
- **RMSE (long recipes):** 0.6617 stars
- **Observed statistic:** RMSE_quick - RMSE_long = -0.0586 stars

**Statistical Test:**
- **p-value:** ≈ 0.84 (4,200 out of 5,000 permutations had a statistic ≥ -0.0586)
- **Significance level:** α = 0.05

**Interpretation:**
The observed difference is actually **negative** (-0.0586), meaning the model performs slightly **better** on quick recipes than on long recipes (lower RMSE = better predictions). However, this difference is not statistically significant.

### Conclusion

We **fail to reject** the null hypothesis at the 0.05 significance level. The permutation test yields a p-value of approximately 0.84, which is much larger than our significance threshold. This indicates that:

1. **No evidence of unfairness**: The observed difference in RMSE between quick and long recipes is consistent with random chance. We do not have sufficient evidence to conclude that the model is unfair with respect to recipe preparation time.

2. **Model performs similarly**: The model's performance is roughly equivalent for both quick and long recipes, with any observed differences likely due to sampling variability rather than systematic bias.

3. **Slight advantage for quick recipes**: Interestingly, the model actually performs slightly better on quick recipes (RMSE 0.6031 vs. 0.6617), though this difference is not statistically significant. This could reflect:
   - Better data availability for quick recipes (more training examples)
   - More consistent rating patterns for quick recipes
   - Features that are more predictive for quick recipes

4. **Fairness confirmed**: The model appears to be fair with respect to recipe preparation time, treating quick and long recipes similarly in terms of prediction accuracy.

<iframe
  src="assets/fairness_rmse_perm.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The permutation distribution shows that the observed test statistic (-0.0586) falls well within the null distribution, with the majority of permutation statistics being larger (more positive) than the observed value. This provides no evidence of unfairness and confirms that the model performs similarly for both groups.

### Implications

The fairness analysis suggests that our model does not systematically disadvantage either quick or long recipes. This is important because:
- **Equitable recommendations**: The model can be used to recommend recipes without bias toward preparation time
- **Reliable predictions**: Predictions are similarly accurate regardless of recipe preparation time
- **No systematic errors**: There's no evidence of systematic prediction errors that would unfairly impact certain recipe types

However, we note that this analysis examines fairness only with respect to preparation time. A comprehensive fairness analysis would examine other potential sources of bias, such as recipe complexity, cuisine type, or nutritional content.

---

## Conclusion

### Summary of Findings

This project comprehensively explored factors influencing recipe ratings on Food.com and built predictive models to estimate average ratings. Through exploratory data analysis, hypothesis testing, and machine learning, we gained insights into what makes recipes successful on the platform.

**Key Findings:**

1. **Recipe Characteristics and Ratings**: Our analysis revealed that recipe characteristics like preparation time, number of steps, and number of ingredients are associated with ratings, though the relationships are relatively weak. Quick recipes (≤ 30 minutes) tend to have slightly higher ratings than long recipes, and recipes with more ingredients tend to have slightly higher ratings, but these effects are modest.

2. **Missingness Patterns**: The missingness of ratings depends on both observed variables (year submitted, number of ingredients) and unobserved factors (recipe visibility, user engagement), suggesting a complex missingness mechanism that is partially MAR but also has NMAR components.

3. **Prediction Performance**: Our final Lasso regression model achieved moderate prediction accuracy (RMSE ≈ 0.65 stars), suggesting that while recipe metadata provides some predictive power, ratings depend substantially on factors not captured in structured data, such as taste, presentation, and user preferences.

4. **Model Fairness**: Our fairness analysis found no evidence that the model performs differently for quick versus long recipes, indicating that the model treats both groups similarly and does not exhibit systematic bias based on preparation time.

### Limitations and Challenges

**Data Limitations:**
- Recipe ratings depend on subjective factors (taste, presentation) that are difficult to quantify
- Missing ratings may not be random, potentially biasing our analysis
- Limited feature set: we only used structured metadata, not text descriptions or images

**Model Limitations:**
- Moderate prediction accuracy suggests room for improvement
- Linear models may not capture complex non-linear relationships
- Feature engineering provided only marginal improvements over the baseline

**Methodological Considerations:**
- The time-of-prediction constraint limits the features we can use
- Cross-validation and train-test splits help but don't eliminate all sources of error
- Fairness analysis examined only one dimension (preparation time)

Predicting recipe ratings is inherently challenging because ratings reflect subjective user preferences that depend on factors beyond recipe metadata. However, our analysis demonstrates that structured recipe characteristics do provide some predictive power, and our models can serve as useful tools for understanding recipe success factors. The moderate prediction accuracy suggests that while we can make reasonable predictions, there is substantial unexplained variation that reflects the complexity of user preferences and the subjective nature of recipe quality.

