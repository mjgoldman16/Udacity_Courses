# README

A README documentation file which briefly describes the software and libraries used in your project, including any necessary references to supporting material. If your project requires setup/startup, ensure that your README includes the necessary instructions.

These libaries are basic data manipulations libraries for Python. They are used to create dataframes, get statistics, and time how long each cell is taking. I don't believe I need to include much information here


```python
# The basics
import pandas as pd
import numpy as np
from scipy import stats
import time
```

These libraries are used for visulaization. I use these to visualize the AUPRC curves, the distribution curves, and the correlation plot. 


```python
# The visulaizations tools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
```

There are a number of non-neural network modeling libraries. They initialize the models that we will use in our capstone project


```python
# The models, optimizers, and other related libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet, SGDClassifier 
from xgboost import XGBClassifier
```

These are tensorflow libraries to create neural networks. These are used to create the components of our neural network.


```python
# The neural network imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras import layers
```

These are are preprocessing libraries. They are used to normalize columns (such as amount and time) as well as split up the dataset


```python
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
```

We used this to explore the possibility of running our data through a PCA. It takes the data and puts it into linear non-correlated principal components. We used this in our data exploration


```python
from sklearn.decomposition import PCA
```

We use these libraries to score our create the pipeline, cross validate our models, and score our them. We have a number of different scoring metrics, which we use together in a custome scoring function


```python
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, precision_recall_curve, confusion_matrix, f1_score, average_precision_score, recall_score, classification_report
```

We use these libraries to rebalance the dataset. The NearMiss takes the samples the majority class that is most similar to the minority class, until the dataset is 50/50. The SMOTE upsamples the minority class and creates synthetic results that are similar to them.


```python
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
```

We use these libraries to optimize our models. The bayesian optimization is used to optimize the non-neural network models, while Adam is used to optimize our two neural networks


```python
from bayes_opt import BayesianOptimization
from tensorflow.keras.optimizers import Adam
```

We use these libraries to save our models so we don't have to retrain our models everytime we restart the notebook


```python
import joblib 
import pickle
from tensorflow.keras.models import load_model
```
