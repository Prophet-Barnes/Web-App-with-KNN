from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st, numpy as np

# Defining title for the Web Page
st.title('Iris Flower Species Classifier') 

# Defining Input and Output
var = load_iris()
x = var.data    # Input
y = var.target  # Output

# Creating and Fitting a KNN Model
model = KNeighborsClassifier(n_neighbors = 13, metric = 'euclidean')
model.fit(x, y)

# Creating Sliders for the Web Page
x_min = np.min(x, axis = 0)   # Loading min values of the input features in a numpy array
x_max = np.max(x, axis = 0)   # Loading max values of the input features in a numpy array

sepal_length = st.slider('Sepal Length', float(x_min[0]), float(x_max[0]))
sepal_width = st.slider('Sepal Width', float(x_min[1]), float(x_max[1]))
petal_length = st.slider('Petal Length', float(x_min[2]), float(x_max[2]))
petal_width = st.slider('Petal Width', float(x_min[3]), float(x_max[3]))

y_pred = model.predict([[sepal_length , sepal_width, petal_length, petal_width]])
op = ['Iris-Setosa','Iris-Versicolor','Iris-Virginica']
st.title('Prediction: ' + str(op[y_pred[0]]))       # To obtain output value in 1D, we provided y_pred[0]
