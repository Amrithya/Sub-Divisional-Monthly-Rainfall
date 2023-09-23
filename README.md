# Sub-Divisional-Monthly-Rainfall
Data Analysis and prediction of rainfall in sub divisional parts of India with the rainfall data from 1901 to 2017

#Introduction
Did you know that India is a country with diverse weather conditions? The climate of India consists of a wide range of weather conditions across a vast geographic scale and varied topography. With the diversity in weather conditions, the most anticipated season in India is the monsoon. Rain is of immense importance in India due to its significant impact on various aspects of the country's environment, agriculture, economy, and culture. Here are some key reasons why rain is crucial in India. India heavily depends on agriculture, and hence, rainfall plays a major role in the Indian economy. Rainfall doesn't always bring joy; sometimes, it's miserable. India is highly vulnerable to tropical cyclones in the basin, from the east or the west. On average, 2 or 3 tropical cyclones make landfall in India each year, with about one being a severe tropical cyclone or greater. So, it's very important to study rainfall in India. Here we do a data analysis on the rainfall amount all over the subdivisional parts of India and a predition of rainfall in Tamil Nadu using neural networks to predict the avg rainfall, the neural net is used to create multiple features that helps in predicting the data points with more seasonal variations.

#Packages
As we are trying to implement neural networks, we need Dense, Activation, Dropout, LSTM, we need libraries like TensorFlow/Keras. Installing TensorFlow will give us access to all the imports.

```bash
pip install requirement.txt
```

#Dataset
```bash
https://data.gov.in/resource/sub-divisional-monthly-rainfall-1901-2017
```

#Implementation

We apply the MinMax scaler from sklearn to normalize data in the (0, 1) interval.The purpose of fit_transform is to both "fit" the transformer to the data and then "transform" the data using the learned parameters.

```bash
scaler = MinMaxScaler(feature_range = (0, 1))
dataset = scaler.fit_transform(data_raw)
```

Using 60% of data for training, 40% for validation.This allows us to train our model on one subset, tune hyperparameters on another, and finally evaluate the model's performance on a third, unseen test subset.

```bash
TRAIN_SIZE = 0.80
```

Create test and training sets for one-step-ahead regression.

```bash
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)
```

'fit_model' that builds and trains a neural network model using the Keras library, which is integrated into TensorFlow. This function is designed to create a specific neural network architecture for a regression task.Inside the function, a Sequential model is created. Sequential models are linear stacks of layers in which you add one layer at a time.

An LSTM layer with 2000 units is added to the model. It uses the hyperbolic tangent (tanh) activation function and the hard sigmoid inner activation function. The input_shape is set to (1, window_size), which implies that the input data is expected to have a shape of (batch_size, 1, window_size).

A Dropout layer with a dropout rate of 0.2 is added after the LSTM layer. Dropout is a regularization technique that helps prevent overfitting.

Several Dense layers are added to the model with different numbers of units and dropout rates between them. These Dense layers introduce non-linearity into the model.

The final Dense layer has a single unit and uses a linear activation function, indicating that this model is designed for regression tasks where it predicts continuous numeric values.

The model is compiled with the mean squared error (MSE) loss function and the Adam optimizer, which is a popular choice for optimization.

RMSE is used evaluating the performance of regression models, especially when dealing with continuous numeric predictions
```bash
rmse_train, train_predict = predict_and_score(model1, train_X, train_Y)
rmse_test, test_predict = predict_and_score(model1, test_X, test_Y)
```
