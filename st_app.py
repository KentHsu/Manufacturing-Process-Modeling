import streamlit as st
from Gaussian.gaussian_st import gaussian_regression
from Gaussian.l1_l2_st import regularization
from KDE.kde_st import kernel_density_estimation


def main():
    page = st.sidebar.selectbox("Choose a Theme",\
           ["README", "Gaussian Regression", 
	    "L1/L2 Regularization", "Kernel Density Estimation"])
    if page == "Gaussian Regression":
        gaussian_regression()
    elif page == "L1/L2 Regularization":
        regularization()
    elif page == "Kernel Density Estimation":
        kernel_density_estimation()
    else:
        st.title("Manufacturing Process Modeling")
        st.header("Themes:")
        st.markdown("* Gaussian Regression")
        st.markdown("* L1/L2 Regularization")
        st.markdown("* Kernel Density Estimation")


if __name__ == "__main__":
    main()

