import numpy as np
import pandas as pd
import datetime
import glob
import os

def read__MultipleCSVs(folder_path, nfiles):
    allfiles = glob.glob(os.path.join(folder_path, "*.csv"))
    return allfiles

def read_SingleCSV(file):
    print(file)
    df = pd.read_csv(file, names=["lat", "long", "created_at", "updated_at", "device_id", "speed",
                                  "orientation", "distance", "received_at", "io_state", "availability",
                                  "blnk",
                                  'id'])
    df = df.reset_index(drop=True)
    if df.lat[0] == 'lat':
        df = df[1:]
    return df


def perform_PreFormating(df):

    ###########################################################
    ### Initial conversion of all data types to 'str' format to avoid future errors
    df['io_state'] = df.io_state.astype(str)
    df['distance'] = df.distance.astype(str)
    df['speed'] = df.speed.astype(str)
    df['lat'] = df.lat.astype(str)
    df['long'] = df.long.astype(str)

    ##########################################################
    ## Capturing IO_State Data
    df['io_state'] = df.io_state.apply(lambda x: x.zfill(8))
    df['dev_state'] = df.io_state.apply(lambda x: x[1])

    def fuelConvert(x):
        try:
            return int(x[-3:], 16)
        except:
            return 0


    ##########################################################
    ## Capturing Fuel Voltage
    df['FuelVoltage'] = df.io_state.apply(lambda x: fuelConvert(x))
    
    ##########################################################
    ## Extracting relevant Columns and create new Dataframe
    df2 = pd.DataFrame()
    df2[['datetime', 'lat','long','speed', 'distance', 'fuelVoltage', 'dev_state']] = df[['received_at','lat','long',
                                                                             'speed', 'distance', 'FuelVoltage', 'dev_state']]

    ###########################################################
    ## Extracting and Sorting Datetime
    df2 = df2.dropna(subset= ['datetime'])
    df2.datetime = df2.datetime.apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
    df2 = df2.sort_values(['datetime'], ascending=True)

    ##########################################################
    ### Reformatting Speed, Distance, Fuelvoltage Data to 'float' type
    df2.speed = df2.speed.apply(lambda x: typecast(x))
    df2.distance = df2.distance.apply(lambda x: typecast(x))
    df2.fuelVoltage = df2.fuelVoltage.apply(lambda x: typecast(x))

    df2 = resetIndex(df2)
    return df2

def typecast(x):
    try: 
        return float(x)
    except:
        return 0

def perform_postFormating(df):

    ###########################################################
    ## Removing Device_State OFF Data
    print (len(df))
   # print (df.datetime[0])
    df['dev_state'] = df['dev_state'].apply(lambda x: int(x))
    newDff = df[df['dev_state'] == 1].reset_index(drop = True)

    if len(newDff)==0:
        raise ValueError("ERROR!!! No Device-ONSTATE data available")

   # print (len(newDff))
    #print (df.datetime[0])
    
    ###########################################################
    ## Calling Outliar function

    #print("Enter Fuel Upper Limit Cutoff : ");
    newDf = removeOutliar(newDff)
    if len(newDf) <20:
        raise Exception("Empty Dataset passed. No theft analysis possible over it.")

   # print(newDf.fuelVoltage.max())
    newDf = resetIndex(newDf.copy())
    print (len(newDf))
    #print (newDf.datetime[0])
    
    ###########################################################
    ### Calling Normalisation Function.
    #newDf2 = norm(newDf.copy(), fuelMax)
   # print(newDf2.fuelVoltage.max())

    ###########################################################
    ## Removing bottom 1% of FuelMax Values
    newDf2 = newDf[newDf.fuelVoltage > 0.01*(newDf.fuelVoltage.max() - newDf.fuelVoltage.min())]
    
    newDf2 = resetIndex(newDf2.copy())
    print ("LenPostFormating: ", len(newDf2))
    
    return newDf2, newDff



def removeOutliar(df):

    df = df.sort_values(['datetime'], ascending=True)  ## Sorting Datetime
    df = df[df['distance'] >= 0]
    #print (len(df))
    ## Removing Y-axis outliar using 'mean -3SD'
    df = df[abs(df.fuelVoltage - df.fuelVoltage.median()) <= 2 * df.fuelVoltage.std()]
    df = resetIndex(df)
    #print (len(df))
    #print (df.datetime[0])

    ## Removing Datetime Outliar
    timeDiff = df.datetime.shift(-1) - df.datetime  ## Calculating consecutive datetime differences between indexs
    try:
        ## Building list of Datetimes which are in diff greator than '2 Days' with immediate next datapoint.
        ## Then extracting index of last datetime of these jargon (sorted) dates
        timeJumpIndex = timeDiff[timeDiff > pd.Timedelta('2 day')].index
        #print(timeJumpIndex)
        refIndex = lastIndex = 0

        for timeIndex in timeJumpIndex:
            if (timeIndex - refIndex) <= 4000:
                lastIndex = timeIndex
            else:
                break
    except:
        lastIndex = -1
    df = df[(lastIndex +1):]
    #print (len(df))
    #print (df.datetime[lastIndex +1])
    return df

def resetIndex(df):

    df = df.reset_index(drop=True)
    return df


#################################################
### Function to normalise
def norm(df, fuelMax = 100, fuelMin = 0):

    df['distance'] = df['distance'] / df['distance'].max()
    df['speed'] = df['speed'] / df['speed'].max()
    if fuelMax ==100 :
        df['fuelVoltage'] = (df['fuelVoltage'] - df['fuelVoltage'].min())/ (df['fuelVoltage'].max()- df['fuelVoltage'].min())
    else:
        df['fuelVoltage'] = (df['fuelVoltage'] - fuelMin)/(fuelMax - fuelMin)
        
    return df

