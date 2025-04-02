from sklearn.datasets import load_iris
import pandas as pd
from multi_classifier_tester import test_all_classifiers
import streamlit as st

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

results = test_all_classifiers(X,y)

st.dataframe(results, use_container_width=True, hide_index=True)