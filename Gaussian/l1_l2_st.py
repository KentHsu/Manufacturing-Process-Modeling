import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
from .gaussian_basis import GaussianFeatures


def basis_plot(model):
    rng = np.random.RandomState(1)
    x = 10 * rng.rand(50)
    y = 0.5 * (rng.rand(50) - 0.5)
    model.fit(x[:, np.newaxis], y)

    xfit = np.linspace(0, 10, 1000)
    yfit = model.predict(xfit[:, np.newaxis])

    fig, ax = plt.subplots(2, sharex=True)
    ax[0].scatter(x, y)
    ax[0].plot(xfit, yfit)
    ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
    
    ax[1].plot(model.steps[0][1].centers_,
	       model.steps[1][1].coef_)
    ax[1].set(xlabel='basis location',
	      ylabel='coefficient',
	      xlim=(0, 10))
    return fig


def regularization():
    # setup sidebar for Gaussian regression
    st.sidebar.title('Dimension and Width')
    dimension = st.sidebar.slider('Adjust to change the dimension',\
                                   10, 30, value=20)
    width = st.sidebar.slider('Adjust to change the width',\
                               0.5, 5.5, value=3.0, step=0.25)

    # regularization regresion model
    l2_model = make_pipeline(GaussianFeatures(dimension, width),\
                             Ridge(alpha=0.1))
    l1_model = make_pipeline(GaussianFeatures(dimension, width),\
                             Lasso(alpha=0.001))


    # Streamlit application
    st.title("Gaussian Function Basis Regression")

    st.header("L2 Regularization")
    st.markdown("L2 regularization gives penalties on large regression coefficients")
    st.markdown("Small coefficient to prevent overfitting")
    l2_fig = basis_plot(l2_model)
    st.pyplot(l2_fig)

    st.header("L1 Regularization")
    st.markdown("Zero coefficients when applying L1 regulariztion")
    l1_fig = basis_plot(l1_model)
    st.pyplot(l1_fig)


