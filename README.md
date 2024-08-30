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
