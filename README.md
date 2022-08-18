# Final Project Data Scientist Virtual Internship Experience ID/X Partners
## Credit Risk Scoring Prediction
By : Trianto Haryo Nugroho
### Description :

This is the final project of my contract period as an Intern Data Scientist at ID/X Partners. I was involved in the project of a lending company. I've collaborate with various other departments on this project to provide technology solutions for the company. I built a model that can predict credit risk using a company-provided data set consisting of accepted and rejected request data. In addition, I also provide visual media to present solutions to clients. The visual media you create is clear, easy to read, and communicative. The work on this end-to-end solution is carried out in the Python programming language while still referring to the Data Science framework/methodology.
### Use Case

Credit Risk Scoring Prediction
#### Objective Statement:
* Get business insight about the distribution of the 4 credit scoring classifications
* Reducing the risk in deciding to apply for a credit loan that has a bad credit score
* Increase income by accepting credit loan applications with a good credit score
#### Challenges:
* Large size of data that is still not clean and various data types
* Need several coordination with various other department
* Original dataset from clients who need immediate business problem solutions
#### Methodology / Analytic Technique:
* Descriptive Analysis
* Diagnostic Analysis
* Predictive Analysis
* Multiclass Classification Algorithm
#### Business Benefit:
* Helping Risk Management Team to create credit score prediction for credit loan application
* Knowing the factors that affect the credit score
### Business Understanding
* Credit scoring is a statistical analysis performed by lenders and financial institutions to determine the creditworthiness of a person or a small, owner-operated business. Credit scoring is used by lenders to help decide whether to extend or deny credit.
* What are the factors that cause a good credit score?
* What are the factors that cause a bad credit score?
* What kind of prediction model is the best for credit score prediction?
* What recommendations are given to lending companies to accept or reject a credit loan?
### Data Understanding
* Data is loan credit historical dataset from 2007 - 2014
* The dataset consists of 466,285 rows and 75 columns
#### Data Dictionary:
1. **unnamed** : Index. 
2. **id** : A unique loan credit assigned ID for the loan listing.
3. **member_id** : A unique LC assigned ID for the borrower.
4. **loan_amnt** : Last month payment was received.
5. **funded_amnt** : The total amount committed to that loan at that point in time.
6. **funded_amnt_inv** : The total amount committed to that loan at that point in time for portion of total amount funded by investors.
7. **term** : The number of payments on the loan. Values are in months and can be either 36 or 60.
8. **int_rate** : Interest Rate on the loan.
9. **installment** : The monthly payment owed by the borrower if the loan originates.
10. **grade** : LC assigned loan grade.
11. **sub_grade** : LC assigned loan grade.
12. **emp_title** : The job title supplied by the Borrower when applying for the loan.
13. **emp_length** : Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.
14. **home_ownership** : The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.
15. **annual_inc** : The self-reported annual income provided by the borrower during registration.
16. **verification_status** : Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified.
17. **issue_d** : The month which the loan was funded.
18. **loan_status** : Current status of the loan.
19. **pymnt_plan** : Indicates if a payment plan has been put in place for the loan.
20. **url** : URL for the LC page with listing data.
21. **desc** : Loan description provided by the borrower.
22. **purpose** : A category provided by the borrower for the loan request.
23. **title** : The loan title provided by the borrower.
24. **zip_code** : The first 3 numbers of the zip code provided by the borrower in the loan application.
25. **addr_state** : The state provided by the borrower in the loan application.
26. **dti** : A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
27. **delinq_2yrs** : The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years.
28. **earliest_cr_line** : The month the borrower's earliest reported credit line was opened.
29. **inq_last_6mths** : The number of inquiries in past 6 months (excluding auto and mortgage inquiries).
30. **mths_since_last_delinq** : The number of months since the borrower's last delinquency.
31. **mths_since_last_record** : The number of months since the last public record.
32. **open_acc** : The number of open credit lines in the borrower's credit file.
33. **pub_rec** : Number of derogatory public records.
34. **revol_bal** : Total credit revolving balance.
35. **revol_util** : Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.        
36. **total_acc** : The total number of credit lines currently in the borrower's credit file.
37. **initial_list_status** : The initial listing status of the loan. Possible values are – W, F.
38. **out_prncp** : Remaining outstanding principal for total amount funded.
39. **out_prncp_inv** : Remaining outstanding principal for portion of total amount funded by investors.
40. **total_pymnt** : Payments received to date for total amount funded.
41. **total_pymnt_inv** : Payments received to date for portion of total amount funded by investors.
42. **total_rec_prncp** : Principal received to date.
43. **total_rec_int** : Interest received to date.
44. **total_rec_late_fee** : Late fees received to date.
45. **recoveries** : Post charge off gross recovery.
46. **collection_recovery_fee** : Post charge off collection fee.
47. **last_pymnt_d** : Last month payment was received.
48. **last_pymnt_amnt** : Last total payment amount received.
49. **next_pymnt_d** : Last month payment was received.
50. **last_credit_pull_d** : The most recent month LC pulled credit for this loan.
51. **collections_12_mths_ex_med** : Number of collections in 12 months excluding medical collections.
52. **mths_since_last_major_derog** : Months since most recent 90-day or worse rating.
53. **policy_code** : publicly available policy_code=1, new products not publicly available policy_code=2.
54. **application_type** : Indicates whether the loan is an individual application or a joint application with two co-borrowers.
55. **annual_inc_joint** : The combined self-reported annual income provided by the co-borrowers during registration.
56. **dti_joint** : A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations, excluding mortgages and the requested LC loan, divided by the co-borrowers' combined self-reported monthly income.
57. **verification_status_joint** : Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified.
58. **acc_now_delinq** : The number of accounts on which the borrower is now delinquent.
59. **tot_coll_amt** : Total collection amounts ever owed.
60. **tot_cur_bal** : Total current balance of all accounts.
61. **open_acc_6m** : Number of open trades in last 6 months.
62. **open_il_6m** : Number of currently active installmant trades.
63. **open_il_12m** : Number of installment accounts opened in past 12 months.
64. **open_il_24m** : Number of installment accounts opened in past 24 months.
65. **mths_since_rcnt_il** : Months since most recent installment accounts opened.
66. **total_bal_il** : Total current balance of all installment accounts.
67. **il_util** : Ratio of total current balance to high credit/credit limit on all install acct.
68. **open_rv_12m** : Number of revolving trades opened in past 12 months.
69. **open_rv_24m** : Number of revolving trades opened in past 24 months.
70. **max_bal_bc** : Maximum current balance owed on all revolving accounts.
71. **all_util** : Balance to credit limit on all trades.
72. **total_rev_hi_lim** : Total revolving high credit/credit limit.
73. **inq_fi** : Number of personal finance inquiries.
74. **total_cu_tl** : Number of finance trades.
75. **inq_last_12m** : Number of credit inquiries in past 12 months. 
### Data Preparation
* Python Version : 3.8.8
* Packages : Pandas, Numpy, Matplotlib, Seaborn, Sklearn, statsmodels.api, etc
### I. Load Library & Data
### II. Exploratory Data Analysis
### III. Data Splitting
### IV. Outlier Handling
### V. Missing Value Handling
### VI. Feature Selection
### VII. Modeling
### VIII. Evaluation
### IX. Data Inference
### X. Data Inference
