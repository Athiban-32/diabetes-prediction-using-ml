import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.datasets import load_diabetes
load_diabetes=load_diabetes()
x=load_diabetes.data
y=load_diabetes.target
data=pd.DataFrame(x,columns=load_diabetes.feature_names)
data.head()
print(load_diabetes.DESCR)
print(data.shape)
data.info()
data.describe()

