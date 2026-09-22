# Automobile Dataset Prediction Using LSTM

## Project Description

This project demonstrates the application of **Long Short-Term Memory (LSTM) neural networks** to an automobile dataset. The dataset is obtained from Kaggle and contains **205 automobile records with 26 different attributes**, including symboling, wheel-base, length, width, height, curb-weight, engine-size, horsepower, city-mpg, highway-mpg, and price.

The project begins by loading and exploring the automobile dataset using Python, Pandas, NumPy, and KaggleHub. Numerical features such as **symboling, wheel-base, and length** are selected for the LSTM experiment. Missing or invalid values represented by `"?"` are converted to missing values and replaced using the median. The selected features are then normalized using **MinMaxScaler** so that they can be effectively processed by the neural network.

To prepare the data for an LSTM model, a **time-step of 5** is used. The dataset is transformed into sequences where each input contains five consecutive records. This produces an input shape of **(200, 5, 3)** and a target shape of **(200,)**.

Different types of LSTM architectures are implemented and demonstrated in the project, including:

* **Basic/Vanilla LSTM**
* **Stacked LSTM**
* **Bidirectional LSTM**
* **Multivariate LSTM**
* **Many-to-One LSTM**
* **Many-to-Many LSTM**
* **LSTM Autoencoder**
* **Text/Next-Word LSTM**
* **Dropout LSTM**

The Basic LSTM uses 50 LSTM units followed by a Dense output layer and is trained for 20 epochs. The Stacked LSTM uses two LSTM layers with 64 and 32 units respectively. The Bidirectional LSTM processes the input sequence in both forward and backward directions.
The project also demonstrates sequence-to-sequence learning through the **Many-to-Many LSTM**, where the input and output sequences both contain five time steps. An **LSTM Autoencoder** is also implemented to reconstruct the original input sequence.
Finally, a **Dropout LSTM model** is trained to help reduce overfitting. The trained model is used to generate a prediction and is saved in Keras format as `automobile_lstm_model.keras`.

## Objective

The main objective of this project is to understand how different **LSTM architectures** can process sequential automobile data and generate predictions. It provides practical experience with data preprocessing, normalization, sequence creation, LSTM model construction, training, prediction, and model saving.

## Technologies Used

* Python
* Google Colab
* TensorFlow / Keras
* Pandas
* NumPy
* Scikit-learn
* KaggleHub
* Matplotlib
* Seaborn

## Outcome

The project successfully demonstrates multiple LSTM architectures on the automobile dataset and shows how sequential data can be transformed into suitable input sequences for recurrent neural networks. The final trained Dropout LSTM model is saved as `automobile_lstm_model.keras` for future use.
