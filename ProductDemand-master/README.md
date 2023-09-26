# Product Demand Forecast

It is Timeseries Forecasting project.

# What it Does?
  - It forecasts the demand for a particular product
  
# Datset Used
- Product Demand : https://www.kaggle.com/felixzhao/productdemandforecasting
- ###### It contains a Data File.


        Historical Product Demand - Demand for all products 


# Time Series Forecasting
- We will be applying two types of forecasting on the data
-- **Simple Smoothing**
-- **Exponential Smoothing**
-- **Decomposition**
# Data Preparation
- **Select Product**
  - Dataset contains around ~2000 products. I have chose product **1359**, which had maximum number of records.
- **Aggregate Data**
  - Dataset contains demand for a particular day. Data has been aggregated such that the demand is grouped for a single **month**
- **Remove Outliers**
  - For a particular month(last month), data is is incomplete. So, it has been removed.
- **Train and Test Data**
  - Data has been seperated into test and train. Test data contains demand for last **9 months**.


# Analysis

-- Below Graph shows the demand for the product 1359 for all the years.

![Product Demand](https://raw.githubusercontent.com/nareshkumar66675/ProductDemand/master/reports/Product1359Demand.png "Product Demand")

#### Simple Smoothing:
- Simple Smoothing has been applied to forecast the demand for the product 1359.
- Below graphs shows the Forecast for the product.

![Simple Smoothing](https://raw.githubusercontent.com/nareshkumar66675/ProductDemand/master/reports/SimpleSmooth.png "Simple Smoothing")  

#### Exponential Smoothing:
- Below graphs shows the Forecast for the product.

![Exponential Smoothing](https://raw.githubusercontent.com/nareshkumar66675/ProductDemand/master/reports/ExpSmooth.png "Exponential Smoothing") 

#### Exponential Smoothing:
- Below graphs shows the Forecast for the product using Exponential Smoothing.

![Exponential Smoothing](https://raw.githubusercontent.com/nareshkumar66675/ProductDemand/master/reports/ExpSmooth.png "Exponential Smoothing") 

#### Decomposition:
- Below graph shows the decomposition trend

![Trend](https://raw.githubusercontent.com/nareshkumar66675/ProductDemand/master/reports/DecomTrend.png "Trend") 

- Below graph shows the Forecast for the product using decomposition.

![Exponential Smoothing](https://raw.githubusercontent.com/nareshkumar66675/ProductDemand/master/reports/Decomposition.png "Exponential Smoothing") 

#### Results:
- Below graph shows the **RMSE** for each method.

![Mean Square Error](https://raw.githubusercontent.com/nareshkumar66675/ProductDemand/master/reports/RMSE.png "MEan Square Error")


# Project Struture

##### Src
- ProductDemand.py - python file exported from Jupyter
##### Notebooks
- ProductDemand.ipynb - Jupyter notebook
##### Data
- External - ProductDemand Data
##### Reports
- Graphs - Forecast & Mean Square Error

***


  
