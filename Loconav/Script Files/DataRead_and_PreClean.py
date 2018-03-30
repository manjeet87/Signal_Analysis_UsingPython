import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import glob
import os

def read_SingleCSV(filepath):
    df = pd.read_csv(filepath)
    return df

def read__MultipleCSVs(folder_path, nfiles):
    allfiles = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = []
    n = 0
    for file in allfiles:
        if (n< nfiles) :
            print (file)
            df = pd.read_csv(file, names=["lat", "long", "created_at", "updated_at", "device_id", "speed",
                                                 "orientation", "distance", "received_at", "io_state", "availability",
                                                 "blnk",
                                                 'id'])
            df_list.append(df)
            n+=1

    return df_list, allfiles

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

    df.speed = df.speed.apply(lambda x: float(x))
    df.distance = df.distance.apply(lambda x: float(x))
    df.fuelVoltage = df.fuelVoltage.apply(lambda x: float(x))

    ###########################################################
    ## Calling Outliar function

    #print("Enter Fuel Upper Limit Cutoff : ");

    newDf = removeOutliar(df)

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



def removeOutliar(df):

    df = df.sort_values(['datetime'], ascending=True)  ## Sorting Datetime
    df = df[df['distance'] >= 0]

    ## Removing Y-axis outliar using 'mean -3SD'
    df = df[abs(df.fuelVoltage - df.fuelVoltage.mean()) < 3 * df.fuelVoltage.std()]
    df = resetIndex(df)

    ## Removing Datetime Outliar
    timeDiff = df.datetime.shift(-1) - df.datetime  ## Calculating consecutive datetime differences between indexs
    try:
        ## Building list of Datetimes which are in diff greator than '2 Days' with immediate next datapoint.
        ## Then extracting index of last datetime of these jargon (sorted) dates
        lastIndex = timeDiff[timeDiff > pd.Timedelta('2 day')].index[-1]
    except:
        lastIndex = -1
    df = df[(lastIndex +1):]

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

