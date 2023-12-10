# 🧬 GenoDementiaDetectives 🧬

## Project Description
Our team, GeoDementia Detectives, set out to create a machine learning model that can accurate predict the onset of dementia in patient from gene expression data generated through RNA sequencing. The code in this notebook provides information on model training, evaluation as well as feature selection. The model used was Logistic Regression and in this notebook we explored model performance.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Get a copy

Get a copy of this project by simply running the git clone command.

``` git
git clone https://github.com/hermonpe/GenoDementiaDetectives.git
```

### Prerequisites

Before running the project, we have to install all the dependencies from requirements.txt

``` pip
pip install -r requirements.txt
```

Please note that you need a GPU with proper setup to run the analysis.  Our team used Greak Lakes. We used 2 cores and 180 GB of memory.

## Data Source
We downloaded RNA expression data from the Aging, Dementia, and Traumatic Brain Injury Project, which links to brain tissue data on this website: https://aging.brain-map.org/download/index. 

![image](https://github.com/genodementia-detectives/GenoDementiaDetectives/blob/main/images/data_source_image.png)

The data was organized using the notebook "1. data_import_transformation (12092023A).ipynb," available at ([code/notebooks/1. data_import_transformation (12092023A).ipynb](https://github.com/genodementia-detectives/GenoDementiaDetectives/blob/70e74de5fc3448c190cc608699ed841487c14584/code/notebooks/1.%20data_import_transformation%20(12092023A).ipynb)). Subsequently, the compiled data will be saved in a parquet-formatted file. The RNAseq expression data was then retrieved from the parquet-formatted file and subjected to analysis using the pandas library. Additionally, a series of scikit-learn tools were employed for the evaluation of machine learning models.

Specifically, the following files were used in our modeling: 
* rows-genes.csv: provides gene name and other gene information
* DonorInformation.csv - provides characterists of donors including age, gender, and a data column called "act_demented" that labels each donor as "Dementia" or "No Dementia"
* columns-samples.csv - provides information about the tissue sample and connects the "rnaseq_profile_id" to a "donor_id," which helps with merging datasets
* fpkm_table_normalized.csv - provides normalized gene expression data by "gene_id" and "rnaseq_profile_id"

  ## opening the fpkm_table_normalized.csv zip file
```
   import zipfile
  # Specify the path to your zip file
  zip_file_path = '../data/external/fpkm_table_normalized.csv.zip'
  
  # Specify the extraction directory
  extracted_folder = '../data/external'
  
  # Open the zip file
  with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
      # Extract all the contents into the specified directory
      zip_ref.extractall(extracted_folder)
```
## Data Access

The Aging, Dementia and Traumatic Brain Injury Study is a detailed neuropathologic, molecular and transcriptomic characterization of brains of control and TBI exposure cases from a unique aged population-based cohort from the Adult Changes in Thought (ACT) study.  
This study was developed by a consortium consisting of the University of Washington, Kaiser Permanente Washington Health Research Institute, 
and the Allen Institute for Brain Science, and was supported by the Paul G. Allen Family Foundation. 

This freely available resource (http://aging.brain-map.org/) presents a systematic and extensive data set of study participant metadata, 
quantitative histology and protein measurements of neuropathology, and RNA sequencing (RNA-seq) analysis of hippocampus and neocortex. 
Specific methodological details are available on the “Documentation” tab at http://aging.brain-map.org/.

https://knowledge.brain-map.org/data/38UR89M15WYUTDSN425/summary

The Terms of Use and license information can be found here: https://alleninstitute.org/terms-of-use/. We have also pasted the Terms of Use as of May 2022 in our Github Repository DataAcess.txt file.

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

## Project Approach
![image](https://github.com/hermonpe/GenoDementiaDetectives/blob/main/SIADS699_visual_for%20git%20readme.gif)
