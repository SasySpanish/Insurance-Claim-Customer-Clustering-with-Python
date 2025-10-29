# Insurance Customers Synthetic Dataset

> **Project**: Synthetic data for insurance customer analysis  
> **Description**: This folder contains the raw synthetic dataset used for customer segmentation, risk profiling, and clustering analysis.  
> **Dataset File**: `data_synthetic.csv`  
> **Generation**: Monte Carlo simulation based on Poisson (frequency) and log-normal (severity) distributions.  
> **Purpose**: Used as input for EDA, clustering (PCA + KMeans), and business insights in the `/results/` folder.

---

## Dataset Overview

- **Rows**: ~50,000 (estimated; full file may vary – truncated in preview)  
- **Columns**: 30  
- **Format**: CSV (comma-separated values)  
- **Source**: [Insurance Claims and Policy Data](https://www.kaggle.com/datasets/ravalsmit/insurance-claims-and-policy-data)  
- **Key Features**: Includes demographic, behavioral, policy, and risk-related attributes.  
- **Simulation Notes** [(see here)](src/):  
  - Claim frequency: Poisson with lambda adjusted by `Risk Profile`, `Claim History`, etc.  
  - Claim severity: Log-normal, influenced by `Premium Amount`, `Policy Type`, etc.  
  - Derived fields: `Sim_Frequency`, `Sim_Total_Loss`, `Expected_Loss` (added in preprocessing scripts).

This dataset is **raw** and requires preprocessing (e.g., scaling numerics, encoding categoricals) before modeling.

---

## Column Descriptions

| Column Name | Type | Description | Example Values |
|-------------|------|-------------|---------------|
| **Customer ID** | Integer | Unique identifier for each customer. | 84966, 95568 |
| **Age** | Integer | Age of the customer (e.g., 20–70). | 23, 26 |
| **Gender** | String | Gender of the customer. | Female, Male |
| **Marital Status** | String | Marital status. | Married, Widowed, Single, Divorced, Separated |
| **Occupation** | String | Job or profession. | Entrepreneur, Manager, Nurse, Artist, Salesperson, Lawyer, Teacher, Engineer |
| **Income Level** | Integer | Annual income (20,000–150,000). | 70541, 54168 |
| **Education Level** | String | Highest education attained. | Associate Degree, Doctorate, Bachelor's Degree, Master's Degree, High School Diploma |
| **Geographic Information** | String | State or region in India. | Mizoram, Goa, Rajasthan, Sikkim, West Bengal, Uttar Pradesh, Himachal Pradesh, Manipur, Gujarat, Andaman and Nicobar Islands, Tripura, Telangana, Puducherry, Nagaland, Meghalaya, Andhra Pradesh, Daman and Diu, Delhi, Assam |
| **Location** | Integer | Numeric code for location (possibly ZIP-like). | 37534, 63304 |
| **Behavioral Data** | String | Behavioral indicators (e.g., policy-related). | policy5, policy1 (seems like placeholders) |
| **Purchase History** | String | Date or policy purchase info. | 04-10-2018, 11-06-2018 |
| **Policy Start Date** | String | Start date of the policy (various formats). | 04-10-2018, 11-06-2018 |
| **Policy Renewal Date** | String | Renewal date of the policy. | 08-01-2023, 09-06-2020 |
| **Claim History** | String/Integer | Number or date of claims. | 12-03-2023, 06-09-2023 (mixed; some dates, some counts like 5, 0) |
| **Interactions with Customer Service** | Integer/String | Number of interactions or type. | 5, 0 (counts in sample) |
| **Insurance Products Owned** | String | Policies owned. | policy2, policy1 |
| **Coverage Amount** | Integer | Amount covered by the policy. | 366603, 780236 |
| **Premium Amount** | Integer | Annual premium (500–5,000). | 2749, 1966 |
| **Deductible** | Integer | Deductible amount. | 1604, 1445 |
| **Policy Type** | String | Type of policy. | Group, Family, Individual, Business |
| **Customer Preferences** | String | Preferences (e.g., communication). | Email, Mail |
| **Preferred Communication Channel** | String | Preferred channel. | In-Person Meeting, Text, Phone, Email, Mail |
| **Preferred Contact Time** | String | Preferred time for contact. | Afternoon, Morning, Evening, Anytime, Weekends |
| **Preferred Language** | String | Language preference. | English, French, German, Mandarin, Spanish |
| **Risk Profile** | Integer | Risk level (0–3). | 1, 2, 3, 0 |
| **Previous Claims History** | Integer | Number of previous claims (0–3). | 3, 2, 1, 0 |
| **Credit Score** | Integer | Credit score. | 728, 792, 719 |
| **Driving Record** | String | Driving history. | DUI, Clean, Accident, Major Violations, Minor Violations |
| **Life Events** | String | Recent life events. | Job Change, Retirement, Childbirth, Divorce, Marriage |
| **Segmentation Group** | String | Pre-defined segment. | Segment5, Segment3, Segment2, Segment4, Segment1 |



