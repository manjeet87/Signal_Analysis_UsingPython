import numpy as np
import pandas as pd
import datetime
import glob
import os

def read_SingleCSV(filepath):
    df = pd.read_csv(filepath)
    return df

def read__MultipleCSVs(folder_path, nfiles):
    allfiles = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = []
    #dtype={"lat": str,"long":str, "speed":str, "distance": str, "io_state":str, "recieved_at":str}
    n = 0
    for file in allfiles:
        if (n< nfiles) :
            print (file)
            df = pd.read_csv(file, names=["lat", "long", "created_at", "updated_at", "device_id", "speed",
                                                 "orientation", "distance", "received_at", "io_state", "availability",
                                                 "blnk",
                                                 'id'])
            df = df.reset_index(drop=True)
            if df.lat[0] == 'lat' :
                df = df[1:]
            df_list.append(df)
            n+=1


    return df_list, allfiles

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
    newDf = pd.DataFrame()
    df[['datetime', 'lat','long','speed', 'distance', 'fuelVoltage', 'dev_state']] = df[['received_at','lat','long',
                                                                             'speed', 'distance', 'FuelVoltage', 'dev_state']]

    ###########################################################
    ## Extracting and Sorting Datetime
    df = df.dropna(subset= ['datetime'])
    df.datetime = df.datetime.apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
    df = df.sort_values(['datetime'], ascending=True)

    ##########################################################
    ### Reformatting Speed, Distance, Fuelvoltage Data to 'float' type
    df.speed = df.speed.apply(lambda x: typecast(x))
    df.distance = df.distance.apply(lambda x: typecast(x))
    df.fuelVoltage = df.fuelVoltage.apply(lambda x: typecast(x))
    
    return df

def typecast(x):
    try: 
        return float(x)
    except:
        return 0

def perform_postFormating(df):

    ###########################################################
    ## Removing Device_State OFF Data
    print (len(df))
    df['dev_state'] = df['dev_state'].apply(lambda x: int(x))
    newDf = df[df['dev_state'] == 1]
    print (len(newDf))
    
    ###########################################################
    ## Calling Outliar function

    #print("Enter Fuel Upper Limit Cutoff : ");
    newDf = removeOutliar(newDf)
   # print(newDf.fuelVoltage.max())
    print (len(newDf))
    
    ###########################################################
    ### Calling Normalisation Function.
    #newDf2 = norm(newDf.copy(), fuelMax)
   # print(newDf2.fuelVoltage.max())

    ###########################################################
    ## Removing bottom 1% of FuelMax Values
    newDf2 = newDf[newDf.fuelVoltage > 0.01*(newDf.fuelVoltage.max() - newDf.fuelVoltage.min())]
    
    newDf2 = resetIndex(newDf2.copy())
    print (len(newDf2))
    
    return newDf2, df



def removeOutliar(df):

    df = df.sort_values(['datetime'], ascending=True)  ## Sorting Datetime
    df = df[df['distance'] >= 0]
    print (len(df))
    ## Removing Y-axis outliar using 'mean -3SD'
    df = df[abs(df.fuelVoltage - df.fuelVoltage.mean()) < 2 * df.fuelVoltage.std()]
    df = resetIndex(df)
    print (len(df))

    ## Removing Datetime Outliar
    timeDiff = df.datetime.shift(-1) - df.datetime  ## Calculating consecutive datetime differences between indexs
    try:
        ## Building list of Datetimes which are in diff greator than '2 Days' with immediate next datapoint.
        ## Then extracting index of last datetime of these jargon (sorted) dates
        timeJumpIndex = timeDiff[timeDiff > pd.Timedelta('2 day')].index
        refIndex = lastIndex = timeJumpIndex[0]

        for timeIndex in timeJumpIndex:
            if (timeIndex - refIndex) <= 5000:
                lastIndex = timeIndex
            else:
                break
    except:
        lastIndex = -1
    df = df[(lastIndex +1):]
    print (len(df))
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

