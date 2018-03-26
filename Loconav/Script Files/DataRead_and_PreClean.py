import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import glob
import os

def read_SingleCSV(filepath):
    df = pd.read_csv(filepath)
    return df

def read__MultipleCSVs(folder_path):
    allfiles = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = []
    for file in allfiles:
        print (file)
        df = pd.read_csv(file, names=["lat", "long", "created_at", "updated_at", "device_id", "speed",
                                             "orientation", "distance", "received_at", "io_state", "availability",
                                             "blnk",
                                             'id'])
        df_list.append(df)

    return df_list

def perform_PreFormating(df):

    ##########################################################
    ## Capturing IO_State Data
    df['dev_state'] = df.io_state.apply(lambda x: x[1])

    ##########################################################
    ## Capturing Fuel Voltage
    df['FuelVoltage'] = df.io_state.apply(lambda x: int(x[-3:], 16))

    ##########################################################
    ## Extracting relevant Columns
    newDf = pd.DataFrame()
    newDf[['datetime', 'speed', 'distance', 'fuelVoltage', 'dev_state']] = df[['received_at', \
                                                                             'speed', 'distance', 'FuelVoltage', 'dev_state']]
    return newDf

def perform_postFormating(df):

    ###########################################################
    ## Extracting and Sorting Datetime
    df.datetime = df.datetime.apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
    df = df.sort_values(['datetime'], ascending=True)

    ###########################################################
    ## Calling Outliar function

    #print("Enter Fuel Upper Limit Cutoff : ");
    fuel_cutoff = 500#input()
    newDf = removeOutliar(df, fuel_cutoff)

    newDf = resetIndex(newDf)

    ###########################################################
    ### Calling Normalisation Function.
    newDf = norm(newDf, ['distance', 'fuelVoltage', 'speed'])

    ###########################################################
    ## Removing Device_State OFF Data
    newDf['dev_state'] = newDf['dev_state'].apply(lambda x: int(x))
    newDf = newDf[newDf['dev_state'] == 1]

    newDf = resetIndex(newDf)

    return newDf



def removeOutliar(df, y_outliar = 500):

    df = df.sort_values(['datetime'], ascending=True)
    df = df[df['distance'] >= 0]
    df = df[df.fuelVoltage <= int(y_outliar)]
    df = resetIndex(df)
    return df

def resetIndex(df):

    df = df.reset_index(drop=True)
    return df


#################################################
### Function to normalise
def norm(df, columns = []):

    for column in columns:
        df[column] = df[column] / df[column].max()

    return df


def rem_ErrData_OLD(dff, margin = 0.005):
    x = np.array(dff.index)
    y = np.array(dff.fuelVoltage)

    from sympy.geometry import Point
    i = 0
    dd1 = [0]
    dd2 = [y[1] - y[0]]
    for i in range(1, len(x)):
        try:
            d1 = abs(y[i] - y[i - 1])
            d2 = abs(y[i + 1] - y[i])
        except:
            continue
        dd1.append(d1)
        dd2.append(d2)

    dff['dd1'] = pd.Series(dd1)
    dff['dd2'] = pd.Series(dd2)

    ## Removing Error Data
    dff = dff[(dff.dd1 <= margin) & (dff.dd2 <= margin)]
    dff = dff.reset_index(drop=True)  ## Reseting index

    return dff


