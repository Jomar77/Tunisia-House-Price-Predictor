# Tunisian House Price Prediction
This is a Github repository containing a machine learning model for predicting house prices in Tunisia. The model is built using Python and utilizes various libraries for data analysis and machine learning such as pandas, scikit-learn, and XGBoost.

## Dataset
The dataset used for training and testing the model is a collection of Tunisian house prices, including features such as number of bedrooms, area, location, and other relevant information. The dataset is available in the data folder of this repository.

## Setup and Installation
To set up the environment and run the code in this repository, please follow the instructions below:

Clone the repository onto your local machine using the following command:

```
git clone https://github.com/Jomar77/tunisian-house-price-prediction.git
```

Install the necessary libraries by running the following command in your terminal:

```
pip install -r requirements.txt
```
Once the dependencies are installed, open the tunisian-house-price-prediction.ipynb notebook in Jupyter and run the cells to clean the data, analyze it using linear regression, lasso regression, and decision tree regressor, and visualize it using heatmaps and histograms.

Once the analysis is complete, train the model using the train.py script in the src directory:

```
cd src
python main.ipynb
```
Once the model is run, it will give you a pickle file that can be used for predicting the house market.

## Results
The analysis using linear regression, lasso regression, and decision tree regressor revealed that certain features such as area, location, and number of bedrooms have a significant impact on house prices in Tunisia. 

## Website Deployment Attempt
This repository was originally meant to be used for a website that predicts Tunisian house prices by inputting the number of rooms, bathrooms, size, and location of the house. However, deployment issues forced me to give up on this project. If you are interested in the web application aspect of this project, please refer to this [repo](https://github.com/Jomar77/The-Jomar-Project) for the website deployment attempt.

## Future Improvements
In the future, there are several ways that this project could be improved:

Add more features to the model to improve its accuracy. For example, the age of the house or the quality of the house materials could be included as predictors.
Explore other machine learning algorithms to compare their performance with the current model. For example, a random forest or a neural network might be able to achieve higher accuracy than the current model.
Fix the deployment issues and create a web application to make the model accessible to a wider audience. This could involve using a cloud computing platform such as Google Cloud or Amazon Web Services to host the application.

## Contributing
If you would like to contribute to this repository, please feel free to submit a pull request or create an issue. All contributions are welcome!

## License
This repository is licensed under the MIT License. Please see the LICENSE file for more information.
