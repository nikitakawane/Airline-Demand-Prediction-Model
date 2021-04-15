#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os    
def processData(csvDataFile):       

    '''Reading the csv file name and converting it into
      date Columns as dates and returns the data frame'''
    data = pd.read_csv(csvDataFile, sep=',', header=0)
    
    '''Converted the departure date and the booking date to date time objects and
      Created a new column of Days prior which is calculated as
      the departure date subtracted by the booking date and is converted into string of day of the week.
      Returns New dataframe'''
    data['departure_date'] = pd.to_datetime(data['departure_date'])
    data['booking_date'] = pd.to_datetime(data['booking_date'])
    data['days_prior'] = (data['departure_date'] - data['booking_date']).dt.days
    
    ''' New column of final Demand is created which is calculated as
     the Maximum Cumulative Booking according to the Departure Date to get Final Demand.
    Returns the New Dataframe'''  
    data['day'] = data['departure_date'].dt.day_name()
    data['demand'] = data['cum_bookings'].groupby(data['departure_date']).transform(max)
    return data

def getAvgForecastRemainingDemand(trainingData):
    '''Creating two other columns to Forecast the remaining demand and to get the average of the Forecasted Remaining Demand.
    The remaining demand can be forecasted by the current demand and the cumulative bookings and the Average of the the
    Remaining demand can be calculated by taking the average of the forecasted demand according the number days prior to
     booking date and the day of the departure date (Weekday/weekend).Returns the new dataframe'''

    trainingData['forecast_remaining_demand'] = trainingData['demand'] - trainingData['cum_bookings']
    trainingData['avg_forecast_remaining_demand'] = trainingData.groupby(['days_prior','day'])['forecast_remaining_demand'].transform(np.mean)
    return trainingData

def getAvgHistoricalBookingRate(trainingData):
    '''Creating two new columns of historical booking rate and the average of the historical booking rate.
      The historical booking rate is calculated as the number of cumulative bookings to the total demand.
      The Average of the historical booking rate is calculated by taking the average of the historical booking rate
      according to the number of days prior to the departure date. Returns the new dataframe'''

    trainingData['historical_booking_rate'] = trainingData['cum_bookings']/trainingData['demand']
    trainingData['avg_historical_booking_rate'] = trainingData["historical_booking_rate"].groupby(trainingData['days_prior']).transform(np.mean)
    return trainingData

def additiveModel(trainingData, validationData):
    '''The Average Forecast Demand is calculated according to the number of days prior and the day of the departure date.
    Hence, we are subsetting and removing the duplicates of days_prior and day.'''

    df = trainingData.drop_duplicates(['days_prior', 'day'])

    #Creating another Dataframe with the unrepeated days prior and day columns
    #with the average of forecasted remaining demand column.

    df1 = df[['days_prior','day','avg_forecast_remaining_demand']]

    #Merged the validation data to create a new dataframe with the newly created dataframe of training data
    #merging on the days_prior and day columns
    df2 = validationData.merge(df1, left_on = ['days_prior', 'day'],right_on=['days_prior', 'day'])

    #creating a new column Additive model forecast to the dataframe which is calculated by
    #the addition of the cumulative booking and the average of the forecasted remaining demand.
    df2['forecast_add'] = df2['cum_bookings'] + df2['avg_forecast_remaining_demand']

    #Creating a new column of the Error estimation which is calculated by the total demand subtracted by the
    #Forecasted demand using the additive model.
    df2['error_add'] = abs(df2['demand'] - df2['forecast_add'])

    #The total sum of the Error of the additive model is calculated.
    errorAdd = df2['error_add'].sum()

    #New dataframe is created with the columns of departure date,booking date and the forecast(Additive model).
    #Returns the new Dataframe and the total error of the additive model.
    dfAdditive = pd.DataFrame(df2[['departure_date','booking_date','forecast_add']])
    return errorAdd, dfAdditive

def multiplicativeModel(trainingData, validationData):
    '''The Average Historical Booking rate is calculated according to the number of days prior to the booking date.
    Hence, subsetting the days prior by removing the duplicates.'''
    df = trainingData.drop_duplicates(['days_prior'])

    #New dataframe is created having the days_prior and the average historical booking rate columns.
    df1 = df[['days_prior','avg_historical_booking_rate']]

    #Merged the validation data to create a new dataframe on the newly created dataframe of the training dataset
    #merging on the days_prior columns.
    df2 = validationData.merge(df1, left_on=['days_prior'],right_on=['days_prior'])

    #Creating a new column for the multiplicative model forecast.
    #The Multiplicative model forecasts the demand as the cumulative bookings to the
    #Average Historical booking rates.
    df2['forecast_multi'] = df2['cum_bookings'] / df2['avg_historical_booking_rate']

    #Creating a new column to obtain the error with the use of multiplicative model
    #which is calculated by subtracting the forecasted demand by multiplicative model
    #to the total demand.
    df2['error_multi'] = abs(df2['demand'] - df2['forecast_multi'])

    #The total Error when used the multiplicatice model is calculated.
    errorMulti = df2['error_multi'].sum()

    #New Dataframe is created with the columns of departure date,Booking date and the forecasts of the multiplicative model.
    #Returns the new dataframe and the sum of the error when using the multiplicative model.
    dfMulti = pd.DataFrame(df2[['departure_date','booking_date','forecast_multi']])
    return errorMulti, dfMulti
    
def getMASE(error, totalError):
    '''Mean Absolute Scaled Error is estimated keeping the Naive forecast as the reference and
     it is calculated as the sum of the Errors from the additive forecasting model
    and the multiplicative forecasting model to the total error obtained from the validation data compared to the Naive forecast.
    Returns MASE'''
    MASE = error/totalError
    return MASE

def airlineForecast(trainingFile, validationFile):
    """ Reads the trainingfile and the validationfile inputs.
     Airline_data_training.csv is the input for the trainingFile
     and Airline_data_validation.csv is the input for the validationFile."""
    
    training = processData(trainingFile)
    validation = processData(validationFile)

    #Function getAvgForecastRemainingDemand is called with the input
    #from the training data set.
    training = getAvgForecastRemainingDemand(training)

     #Function  getAvgHistoricalBookingRate is called with the input
    #from the training data set.
    training = getAvgHistoricalBookingRate(training)

    #Creating a new column of the demand for the validation dataset which is calculated by the maximum of the cumulative bookings
    #according to the departure date.
    validation['demand'] = validation['cum_bookings'].groupby(validation['departure_date']).transform(max)

    #Removing the data where the days prior to the departure date is zero.
    # As the total demand will already be estimated before days prior =0
    validation = validation.loc[validation['days_prior'] > 0]

    #Creating a new column to calculate the error obtained by the model when estimated the demand
    #compared to the naive forecast model.
    validation['error'] = abs(validation['demand'] - validation['naive_fcst'])

    #Total error is calculated to compute the MASE.
    totalError = validation['error'].sum()

    #Reading the training and validation csv filed to estimate the additive model error and the Dataframe of the Addtive model
    additiveModelError, dfAdditive = additiveModel(training, validation)

    #Reading the training and validation csv filed to estimate the multiplicative model error and the Dataframe of the Multiplicative model.
    multiplicativeModelError, dfMulti = multiplicativeModel(training, validation)

    #getMase function is called with the additiveModelError,multiplicativeModelError and the total error as the parameters to obtain
    #MASE for additive model and Multiplicative model.Returns Additive model dataframe,Additive model MASE,Multiplicative model dataframe
    # and the Multiplicative MASE.
    additiveMASE = getMASE(additiveModelError, totalError)
    multiplicativeMASE = getMASE(multiplicativeModelError, totalError)
    
    print( "Dataframe for multiplicative model:","\n",dfMulti,"\n","Dataframe for Additive model:", "\n",
           dfAdditive,"\n",["MASE for Additive Model is: " ,additiveMASE,"MASE for Multiplicative Model is: ",multiplicativeMASE])


    
    
def main():
    # main function for the script to the read data for training file and validation file.
    airlineForecast("airline_data_training.csv", "airline_data_validation.csv")
    

main()

