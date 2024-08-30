# Early Detection and Mitigation of Financial Risks in Home Health Agencies: A CMS Intervention Approach

## Executive Summary
On April 27, 2023, CMS (Centre for Medicare and Medicaid Services.) announced a new proposed rule (CMS-2439-P) that requires at least 80% of Medicaid payments for home care worker compensation to be spent directly on worker wages, limiting the amount of profit HHAs can retain. Agencies that fail to comply with the new proposed rule may incur civil money penalties and sanctions, which may impact their financial stability.

Through a prelimiary analysis it was hypothesised that early intervention through grants and incentives for struggling HHAs could prevent financial instability, creating business value and reducing negative sentiments about CMS-2439-P.

## Objective
Employ machine learning algorithms to develop a business decision strategy that helps CMS evaluate the financial impact of their early intervention strategies, thereby identifying profitability, enhancing patient care quality, and exploring the potential to increase patient visit volume.

## Tech Stack
Programming Language: Python

IDE: Jupyter Notebook

Data Ingestion, Transformation and Cleaning: Scikit-Learn (SimpleImputer, OneHotEncoder, LabelEncoder)

Data Visualization: Matplotlib, Seaborn, Tableau

Machine Learning Model Development: Train_Test_Split, LogisticRegression, LinearRegression, Decision Trees, Lasso, Ridge, LassoCV, RidgeCV, ElasticNet, ElasticNetCV, SVM

Model Evaluation: Scikit-Learn - ModelSelection, ConfusionMatrix, Accuracy, Precision, Recall, F1-score, and ROC-AUC, and Cross-Validation.

## Source and collection

CMS website: https://data.cms.gov

Dataset and Dictionary URLs:

Cost Report
https://data.cms.gov/provider-compliance/cost-report/home-health-agency-cost-report

Specification/Demographics
https://data.cms.gov/provider-characteristics/hospitals-and-other-facilities/provider-of-services-file-internet-quality-improvement-and-evaluation-system-home-health-agency-ambulatory-surgical-center-and-hospice-providers

Quality of Service
https://data.cms.gov/provider-data/search?theme=Home%20health%20services

https://data.cms.gov/provider-data/sites/default/files/data_dictionaries/home_health/HHS_Data_Dictionary.pdf

Initially, the focus was on the HHA Cost Report dataset, which includes identifying information and financial features such as Total Liabilities, Accounts Payable, and Total Visits. The target variable for this analysis, labeled as Net Income or Loss, also comes from this dataset.

It was recognized, however, that other qualitative and quantitative features could influence the variability in net income or net loss. For this reason, the cost report was merged with two additional datasets. One dataset was the Quality of Patient Care dataset, which includes features such as the Quality of Patient Care star rating and the frequency of patients’ breathing improvements. The other dataset was the Provider of Service dataset, which includes features such as Employee Count and Operating Room Count.

## Feature Selection and Data Transformation

The final set of variables was selected based on extensive financial and healthcare-related research. The financial features chosen are important for calculating specific financial ratios, which will be discussed below. Additionally, the quality of patient care provides insight into the satisfaction of those who utilize the HHA’s services. This, in turn, influences the likelihood of a person returning to that specific HHA, ultimately impacting the HHA’s financial performance.

![Features for model training](https://github.com/arnab-raychaudhari/ml-driven-risk-mitagation-for-underperforming-HHAs/blob/dc079d353d44ca77df3c8a6a603842c83d47d25a/Features-for-Model-Development.png)

The selected variables ensure the use of metrics aimed at optimizing patient outcomes, enhancing cost-efficiency, improving patient satisfaction, and driving continuous improvement in healthcare delivery. The three datasets were merged using the common identifier, CCN. A number of null values were identified, particularly in the features of interest. In such cases, categorical variables were imputed based on the most frequent value, and continuous variables were imputed based on the mean value. Symbols in numeric values, such as commas and dollar signs, were also removed to ensure accurate data types. Subsequently, a deeper analysis of financial features was conducted, leading to the creation of three common financial ratios based on selected features.

![Feature transformation to optimize sample size](https://github.com/arnab-raychaudhari/ml-driven-risk-mitagation-for-underperforming-HHAs/blob/bda1e118425a8a7b638aca9311f5e0057de80d58/Financial-Calculation.png)

The decision to create ratios was influenced by common financial industry practices and the need for standardization. For instance, differences in asset size can make peer analysis challenging, such as comparing an HHA with $100 in current assets to another with $1,000,000 in current assets. Ratio analysis, such as the ratio of current assets to current liabilities, facilitates peer comparison between HHAs with varying asset sizes. Additionally, ratio analysis provides a more comprehensive and insightful understanding of financial performance and enhances forecasting and prediction capabilities. Finally, a current ratio greater than 20 was deemed inconsistent with industry averages, leading to the exclusion of HHAs with a current ratio less than 0 and greater than 20 from the dataset.

## Managerial Decisions

To maximize the outcome of incentivizing Home Health Agencies (HHAs) at risk of going out of business, CMS must design a business strategy. This strategy should integrate machine learning model predictions to make informed decisions and target entities that could generate the maximum profit. Integrating the business value of incentivizing an HHA with the probability of incurring a loss or profit in the near future is essential.

Rather than training a limited number of models, three distinct sets of models were trained, each dedicated to a specific purpose: predicting the probability of loss, the value of incentivizing, and the value of not incentivizing a Home Health Agency (HHA). These predictions will be used to calculate the expected benefit of incentivizing an HHA. Among the features considered for measuring the value of an incentive, Quality of Service Rating was identified, as it reflects the value for a federal institution when the incentive is utilized effectively by the HHA. Conversely, Total Visits is seen as a key metric for the value of not incentivizing, as it could decrease if the HHA is not profitable.

![The relationship between Expected Benefit and Value of Incentive](https://github.com/arnab-raychaudhari/ml-driven-risk-mitagation-for-underperforming-HHAs/blob/d8cca4dbe2079c4c86f3db5975b5ff65774a361a/Expected-Benefit-and-Incentive-Values.png)

To provide a range of options for CMS, three distinct classifiers will be trained on the sample data to predict each of Pr(X) and Vni(X). Similarly, for Vi(X), three regressors will be trained, ultimately producing a minimum of nine model sets.

![Set of Machine Learning models](https://github.com/arnab-raychaudhari/ml-driven-risk-mitagation-for-underperforming-HHAs/blob/75753f21207da3ddf7e361cfb583b7d7faafd162/ML%20Model%20Set.png)

Upon training and applying the models to the X_Test sample, predictions for each HHA in X_Test were computed using the Expected Benefit formula and ranked according to the scores in descending order. These ranked lists were then used to draw Profit Curves.

The expected benefit scores were compared against a threshold derived from the value of incentivizing (and not incentivizing) to make a decision regarding whether the ML models recommend CMS to incentivize an HHA. If the score is greater than the threshold, the model set proposes a decision to incentivize the HHA; otherwise, it does not.

Next, the predictions were compared with the actual outcomes (Y_Test) to build a confusion matrix, referred to as the Expected Probability Matrix (EPM). The threshold is given by:

![Threshold Formula](https://github.com/arnab-raychaudhari/ml-driven-risk-mitagation-for-underperforming-HHAs/blob/5a2b2a283d8ddb87888edb7914230953a8abd368/Threshold%20Fromula.png)

To construct the confusion matrix, a deliberate decision was made to consider HHAs that have actually incurred losses as those targeted with an incentive. Since real-world data on which HHAs are incentivized by CMS is not available, this assumption was deemed reasonably safe. Each HHA in the test sample was evaluated by each of the 27 model sets, and the outcome was added to the confusion matrix of the respective model set. In the end, 27 confusion matrices were generated, one for each of the 27 model sets under consideration.

![Confusion Matrix EPM Flow](https://github.com/arnab-raychaudhari/ml-driven-risk-mitagation-for-underperforming-HHAs/blob/f833299106a32b63f4f4c9f5c7cf36be98f55a00/Confusion-Matrix-EPM-Flow.png)

After comparing the predictions from the model sets with the actual outcomes, a more robust cost-benefit framework was designed to guide CMS management toward more informed decisions, incorporating a proxy for prevailing business practices.

The framework explores the potential dollar value that CMS could derive by following the model sets implemented. Due to the lack of real-world business information, certain assumptions were made to develop this proxy.

While analyzing the business value, it was observed that CMS has the authority to review Medicare and Medicaid reimbursement claims before making approval decisions. One reason for HHAs incurring financial losses is CMS declining some of their reimbursement claims. These declined bills can be considered savings for CMS. Therefore, in the cost-benefit framework, these reimbursement savings were generalized to be $10,000 per instance of an HHA predicted to be struggling financially and eventually rescued by CMS.

The administrative cost for targeting a deficit-ridden HHA was approximated to be $1,000. In summary, the dollar benefit for targeting a true positive case is $99,000 ($100,000 - $1,000). If CMS incentivizes an HHA per model prediction and the entity is not incurring financial loss (i.e., a false positive case), then only the administrative cost would be lost, with no surplus savings of $100,000 from reimbursement denials. Conversely, if CMS follows the model prediction not to incentivize an HHA that is financially struggling (i.e., a false negative case), then an opportunity to save $100,000 is lost. Lastly, if an HHA is profitable and this is verified by the model set, then such an entity will not create an income opportunity for CMS.

Thus, the cost-benefit framework used in this implementation is designed to optimize the financial impact of decisions based on model predictions.

![Cost Benefit Framework](https://github.com/arnab-raychaudhari/ml-driven-risk-mitagation-for-underperforming-HHAs/blob/bc797724cd629452c3dbde1f84dec3b19339c063/Cost-Benefit-Framework.png)

Having the capability to rely on the models for informed predictions and the availability of a cost-benefit framework does not, by itself, equip the management team at CMS to make their best decision. The next challenge is to apply the prediction in conjunction with the cost-benefit framework to compute the profits for each HHA in X_Test. This computation of profits will ultimately enable CMS to decide how to conduct the targeting and how much revenue they can generate from the intervention campaign.

By considering the class priors, drawing performance measures from the Expected Probability Matrix (EPM), and using the numbers in the cost-benefit matrix, the following Expected Profit formula was devised.

![Expected Profit Formulation](https://github.com/arnab-raychaudhari/ml-driven-risk-mitagation-for-underperforming-HHAs/blob/90eb54787b998566ec92a31c7b590a8e217052f4/ExpectedProfitFormulation.png)

Merely computing the Expected Profit for all objects in X_Test will not enable CMS to assess the value of the intervention campaign. The Profit Curve, plotted between expected profit and the percentage of test instances, allows management to determine which set of models will provide the best results based on specific goals. One curve for each model set will be plotted against the ranked list generated by that set.

For demonstration purposes, an assumption was made that CMS will run a pilot instance of the intervention campaign, starting by targeting the top 10% and 90% of the loss-incurring HHAs.

## Model Development

The focus was on six different models: three regressors and three classifiers. The three regressors included LASSO, Ridge, and Elastic Net (EN). The three classifiers included Decision Trees (DT), Logistic Regression (LR), and Support Vector Machines (SVM). Specific decisions were required for these models. For instance, it was important to encode an HHA with a loss as 0 and an HHA with a profit as 1. Additionally, it was crucial to subset the data to ensure that the predictions were built off of the loss-related data, as discussed in the course. This approach ensures that the model output can effectively target the HHAs that require incentivization.

## Regressors

Regressors were chosen to predict the value of incentivizing an HHA (V_i) because this value is derived from a continuous variable, the quality of patient care star rating. Despite their pros and cons, the selected regressor models are deemed most appropriate for this modeling purpose.

**LASSO**: The Lasso model, short for Least Absolute Shrinkage and Selection Operator, is a linear regression model used for feature selection and predictions. It penalizes specific coefficients in the model by adding a constraint to the optimization problem. Some coefficients may be pushed to zero, effectively eliminating irrelevant features. The benefit of LASSO is its ability to regularize the model and ensure interpretability. This model was chosen for predicting the value of incentivizing an HHA because it uses feature selection in the prediction process, although no features were ultimately eliminated overall. LASSO ensures that the outputs yield a reliable predictive model for decision-makers in the healthcare industry.

**RIDGE**: Ridge is another linear regression model that incorporates regularization and prevents overfitting. It excels at finding an appropriate balance between bias and variance, which is crucial for generalization and assists in accurate value prediction. Ridge was selected for predicting the value of incentivizing an HHA because of its regularization capabilities. Ridge ensures that the outputs yield a reliable predictive model for decision-makers in the healthcare industry.

**ELASTIC NET**: Elastic Net is a regularization technique that combines both Lasso and Ridge processes in machine learning models, yielding a balance between feature selection and coefficient penalization. Elastic Net was chosen for predicting the value of incentivizing an HHA because it uses a combination of the processes from LASSO and Ridge. Elastic Net ensures that the outputs yield a combined balance of the benefits offered by both Ridge and LASSO.

## Classifiers

Classifiers were chosen to predict the value of not incentivizing an HHA (V_ni) and to predict the probability that an HHA incurs a loss (P_r). The value of not incentivizing is derived from a categorical variable, specifically whether or not the total visits at an HHA fell below the population median. Additionally, the probability of incurring a loss is derived from a categorical variable, a flag indicating whether or not the HHA incurred a loss. The selected classifier models, despite their pros and cons, are considered most appropriate for these specific modeling purposes.

**DECISION TREE**: Decision Trees are a machine learning algorithm used for regression or classification; in this case, they are used for classification purposes. They partition the data into regions to make predictions based on decision rules. By striking the right balance between bias and variance, Decision Trees can capture patterns in the data while achieving predictive accuracy. Decision Trees were chosen for predicting the value of not incentivizing an HHA and predicting the probability that an HHA incurs a loss. This choice is logical because Decision Trees can ensure that the feature regions are split according to an appropriate balance of bias and variance.

**LOGISTIC REGRESSION**: Logistic Regression is a widely used method for binary classification. It utilizes the logistic function to capture non-linear relationships and employs regularization techniques like L1 and L2 while performing feature selection. Logistic Regression was selected for predicting the value of not incentivizing an HHA and predicting the probability that an HHA incurs a loss. This approach is appropriate as the relationships may not all be linear, and Logistic Regression can ensure that the relevant features are effectively analyzed during the prediction process.

**SUPPORT VECTOR MACHINES (SVMs)**: SVMs are a supervised learning technique that involves hyperparameter selection and the choice of a kernel function. SVMs were chosen for predicting the value of not incentivizing an HHA and predicting the probability that an HHA incurs a loss. This choice makes sense as SVMs can handle both linear and non-linear relationships, providing greater flexibility and robustness to the prediction outputs, especially in cases where multiple types of relationships may exist.

## RESULTS AND CONCLUSION

