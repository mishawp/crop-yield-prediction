import pandas as pd
import pydeck as pdk
import streamlit as st
import numpy as np
import os
import requests
from matplotlib import pyplot as plt

def main():
    page = st.sidebar.selectbox("Выбрать страницу", ["Тяжелые хвосты распределений", "Iris Dataset"])

    if page == "Тяжелые хвосты распределений":
        st.header("""Сгенерировать N случайных событий из распределения Фреше с функцией распределения:""")
        st.latex(r'''    
            F(x) = exp(-(\gamma x)^{-1/\gamma}1\{x>0\})
            ''')
        st.text("Для получения результата:")
        st.markdown("* Сгенерируем N нормально распределенных случайных величин $U_i$ [0,1] (нулевое среднее и единичная диспресия).")
        st.markdown("* Вычислим N cлучайных величин с распределением Фреше по формуле:")
        st.latex(r'''    
                    X_i=\dfrac{1}{\gamma}\left(-lnU_i)^{-\gamma}\right)
                ''')
        mu, sigma = 0, 1  # mean and standard deviation
        gamma = st.slider('Желаемая гамма', 0.25, 2.25, 0.5, 0.25)
        N = st.number_input("Желаемое N", 100, 10000, 10000)
        U = np.abs(np.random.normal(mu, sigma, N))
        X = 1 / gamma * (-np.log(U)) ** (-gamma)
        X2 = X[X < 20]
        fig, ax = plt.subplots()
        count, bins, ignored = plt.hist(X2, 100, density=True)
        plt.plot(bins,
                 np.exp(- (gamma * bins) ** (-1 / gamma)) * (1 / gamma) * (gamma * bins) ** (-1 / gamma - 1) * gamma,
                 linewidth=2, color='r')
        st.pyplot(fig)
        
    elif page == "Iris Dataset":
        st.header("""Демонстрация Fisher's Iris датасета""")
        df = load_data()

        # Plotting the dataset
        fig, ax = plt.subplots()
        plt.scatter(df['sepal_length'], df['sepal_width'], c='blue', label='Iris-setosa')
        plt.scatter(df['sepal_length'], df['petal_width'], c='red', label='Iris-versicolor')
        plt.scatter(df['sepal_length'], df['petal_length'], c='green', label='Iris-virginica')

        plt.xlabel('sepal_length')
        plt.ylabel('sepal_width')

        plt.title('Iris Fisher Dataset')

        plt.legend()
        st.pyplot(fig)
        
@st.cache_data
def load_data():
    if not os.path.isfile("data/iris.csv"):
        # Download the Iris Fisher dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        response = requests.get(url)
        data = response.text
        
    # Save the dataset to a local file
    with open("data/iris.csv", "w") as file:
        file.write(data)
    # Load the dataset into a pandas DataFrame
    df = pd.read_csv("data/iris.csv", header=None,
                     names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    return df
      
      
if __name__ == '__main__':  
    main()