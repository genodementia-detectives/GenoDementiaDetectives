# GenoDementiaDetectives

## Project Description
Our team, GeoDementia Detectives, set out to create a machine learning model that can accurate predict the onset of dementia in patient from gene expression data generated through RNA sequencing. The code in this notebook provides information on model training, evaluation as well as feature selection. The model used was Logistic Regression and in this notebook we explored model performance.

## Data Source
We downloaded RNA expression data from the Aging, Dementia, and Traumatic Brain Injury Project, which links to brain tissue data on this website: https://aging.brain-map.org/download/index. The data was organized using the notebook "data_import_transformationl.ipynb," available at (insert link). Subsequently, the compiled data was saved in a parquet-formatted file, accessible at (insert link).

The RNAseq expression data was then retrieved from the parquet-formatted file and subjected to analysis using the pandas library. Additionally, a series of scikit-learn tools were employed for the evaluation of machine learning models.

Specifically, the following files were used in our modeling:
* rows-genes.csv: provides gene name and other gene information
* DonorInformation.csv - provides characterists of donors including age, gender, and a data column called "act_demented" that labels each donor as "Dementia" or "No Dementia"
* columns-samples.csv - provides information about the tissue sample and connects the "rnaseq_profile_id" to a "donor_id," which helps with merging datasets
* fpkm_table_normalized.csv - provides normalized gene expression data by "gene_id" and "rnaseq_profile_id"

## Imported Libraries and Versions
* pandas: 2.1.3
* numpy: 1.22.0
* altair: 4.2.2
* pyarrow: 14.0.1
* scikit-learn: 0.24.2
* seaborn: 0.11.2
* matplotlib: 3.8.2

## Sample import statement
```
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, precision_score

from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
```

## Model Inputs and Features
PLACEHOLDER USED - JONATHAN REPLACE WITH DIAGRAM
![image](https://github.com/hermonpe/GenoDementiaDetectives/assets/116966914/874a464d-ceb6-4f83-81e5-e46e7441d57a)

## Model Outputs
