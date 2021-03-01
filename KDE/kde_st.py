import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def kernel_density_estimation():
    # setup sidebar for KDE
    st.sidebar.title('Bandwidth')
    bandwidth = st.sidebar.slider('bandwidth', 0.05, 1.0, step=0.05)
    def make_data(N, f=0.3, rseed=1):
        rand = np.random.RandomState(rseed)
        x = rand.randn(N)
        x[int(f * N):] += 5
        return x

    x_ker = make_data(1000)
    x_d = np.linspace(-4, 8, 200)

    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(x_ker[:, None])
    logprob = kde.score_samples(x_d[:, None])


    st.title('Kernel Density Estimation')
    st.header("KDE provides similar concept")
    st.markdown("Jagged shape shows up when bandwidth is short")
    fig, ax = plt.subplots()
    plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
    st.pyplot(fig)
