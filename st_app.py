import streamlit as st
from gaussian_st import gaussian_regression
from kde_st import kernel_density_estimation


def main():
    page = st.sidebar.selectbox("Choose a Theme", ["README", "Gaussian Regression", "Kernel Density Estimation"])
    if page == "Gaussian Regression":
        gaussian_regression()
    elif page == "Kernel Density Estimation":
        kernel_density_estimation()
    else:
        st.title("Manufacturing Process Modeling")
        st.header("Themes:")
        st.markdown("* Gaussian Regression")
        st.markdown("* Kernel Density Estimation")


if __name__ == "__main__":
    main()

