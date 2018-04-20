import numpy as np
import pandas as pd

desired_width = 360
pd.set_option('display.width', desired_width)


def Clean_NoiseData(dff, level, fuelMax, fuelMin):
    x = np.array(dff.index)
    y = np.array(dff.fuelVoltage)
    
    i = 0
    
    ## Neighbourhood Distance
    Nds = 0.01*(fuelMax - fuelMin)
                
    dd00000 = [0, 0, 0, 0, 0, 0]
    dd0000 = [0, 0, 0, 0, 0]
    dd000 = [0, 0, 0, 0]
    dd00 = [0, 0, 0]
    dd0 = [0, 0]
    dd1 = [0]

    dd2 = [y[1] - y[0]]
    dd3 = [y[2] - y[0]]
    dd4 = [y[3] - y[0]]
    dd5 = [y[4] - y[0]]
    dd6 = [y[5] - y[0]]
    dd7 = [y[6] - y[0]]

    for i in range(1, len(x)):
        try:
            d00000 = abs(y[i] - y[i - 6])
            d0000 = abs(y[i] - y[i - 5])
            d000 = abs(y[i] - y[i - 4])
            d00 = abs(y[i] - y[i - 3])
            d0 = abs(y[i] - y[i - 2])
            d1 = abs(y[i] - y[i - 1])
            d2 = abs(y[i + 1] - y[i])
            d3 = abs(y[i + 2] - y[i])
            d4 = abs(y[i + 3] - y[i])
            d5 = abs(y[i + 4] - y[i])
            d6 = abs(y[i + 5] - y[i])
            d7 = abs(y[i + 6] - y[i])
        except:
            continue

        dd00000.append(d00000)
        dd0000.append(d0000)
        dd000.append(d000)
        dd00.append(d00)
        dd0.append(d0)
        dd1.append(d1)
        dd2.append(d2)
        dd3.append(d3)
        dd4.append(d4)
        dd5.append(d5)
        dd6.append(d6)
        dd7.append(d7)

        # print (i)

        # dd1.append(0)
        # dd2.append(0)
    dff['dd00000'] = pd.Series(dd00000)
    dff['dd0000'] = pd.Series(dd0000)
    dff['dd000'] = pd.Series(dd000)
    dff['dd00'] = pd.Series(dd00)
    dff['dd0'] = pd.Series(dd0)
    dff['dd1'] = pd.Series(dd1)
    dff['dd2'] = pd.Series(dd2)
    dff['dd3'] = pd.Series(dd3)
    dff['dd4'] = pd.Series(dd4)
    dff['dd5'] = pd.Series(dd5)
    dff['dd6'] = pd.Series(dd6)
    dff['dd7'] = pd.Series(dd7)

    p = dff['dd1']
    ## Removing Error Data
    if level == 1:
        dff1 = dff[(dff.dd1 <= 0.01) & (dff.dd2 <= 0.01)]
        dff1 = dff1.reset_index(drop=True)  ## Reseting index

    if level == 2:
        dff1 = dff[(dff.dd1 <= 0.005) & (dff.dd2 <= 0.005) & (dff.dd0 <= 0.01) & (dff.dd3 <= 0.01)]
        dff1 = dff1.reset_index(drop=True)  ## Reseting index

    if level == 3:
        dff1 = dff[(dff.dd1 <= 0.005) & (dff.dd2 <= 0.005) & (dff.dd0 <= 0.01) & (dff.dd3 <= 0.01) &
                   (dff.dd00 <= 0.015) & (dff.dd4 <= 0.015)]
        dff1 = dff1.reset_index(drop=True)  ## Reseting index

    if level == 4:
        dff1 = dff[(dff.dd1 <= 0.005) & (dff.dd2 <= 0.005) & (dff.dd0 <= 0.01) & (dff.dd3 <= 0.01) &
                   (dff.dd00 <= 0.015) & (dff.dd4 <= 0.015) & (dff.dd000 <= 0.02) & (dff.dd5 <= 0.02)]
        dff1 = dff1.reset_index(drop=True)  ## Reseting index

    if level == 5:
        dff1 = dff[(dff.dd1 <= 0.02) & (dff.dd2 <= 0.02) & (dff.dd0 <= 0.0364) & (dff.dd3 <= 0.04) &
                   (dff.dd00 <= 0.06) & (dff.dd4 <= 0.06) & (dff.dd000 <= 0.08) & (dff.dd5 <= 0.08) &
                   (dff.dd0000 <= 0.1) & (dff.dd6 <= 0.1)]
        dff1 = dff1.reset_index(drop=True)  ## Reseting index

    if level == 6:
        dff1 = dff[(dff.dd1 <= Nds) & (dff.dd2 <= Nds) & (dff.dd0 <= 2*Nds) & (dff.dd3 <= 2*Nds) &
                   (dff.dd00 <= 3*Nds) & (dff.dd4 <= 3*Nds) & (dff.dd000 <= 4*Nds) & (dff.dd5 <= 4*Nds) &
                   (dff.dd0000 <= 5*Nds) & (dff.dd6 <= 5*Nds) & (dff.dd00000 <= 6*Nds) & (dff.dd7 <= 6*Nds)]
        dff1 = dff1.reset_index(drop=True)  ## Reseting index

    # plt.rcParams['figure.figsize'] = [16, 4]
    # plt.plot(p, 'b.')
    # plt.title('Histogram - Consecutive Fuel Difference ', fontsize=15)
    # plt.ylim(0, 0.05)
    #
    # #     plt.ylim(0.02,1)
    # #     plt.savefig("test.png")
    # plt.rcParams['figure.figsize'] = [16, 4]
    # plt.plot(dff.index[:], dff.fuelVoltage[:], 'g.', markersize=2, linewidth=1);
    # plt.ylim(0, 1.1)
    dfClean = dff1[['datetime','lat', 'long','speed','distance','fuelVoltage']]
    dfClean['fuelVoltage_Percent'] = dfClean.fuelVoltage.apply(lambda x: round((100*x/(fuelMax - fuelMin)),2))

    return dfClean


def jump_point(dff, level = 0.05, fuelMax=100, fuelMin=0):
    x = np.array(dff.index)
    y = np.array(dff.fuelVoltage)
    d = np.array(dff.distance)
    
    #######################################################################
    ## No. of Neighbourhood pts, 'n', dependent on sampling Rate of IoT device
    ## Criteria Set = Avg No. of points inside 10Km window will be considered
    if len(dff)==0:
        print ("ERROR!! EMPTY DATASET PaSSED. Pass a Relevant DataSet")
        return list([])

    n = int(10000/(dff.distance.max()/len(dff)))
    
    level = level*(fuelMax - fuelMin)
    theft_pts = []
    refpts = []
    rctr = 0
    ctr = 0
    i = 0

    dd2 = [y[1] - y[0]]
    for i in range(1, len(x)):
        try:
            #             d1 =abs(y[i+1] - y[i])
            #             d2 =abs(y[i+2] - y[i])
            d_forward = (y[i + 1:i + n] - y[i])
            d_backward = (y[i + 1] - y[i - n:i])
            # print (d_backward)
            
        except:
            pass
            #print("**")
        # dd1.append(d1)
        # dd2.append(d2)
        # if (d1 >= 0.05) & (d2 >= 0.05)&(d3 >= 0.05)&(d4 >= 0.05)&(d5 >= 0.05)&(d5 >= 0.05):

        ###########################################################################
        #### Finding probable refueling Points

        if ((sum(d_forward > 1 * 3 * level) in list(range(n-1, n+1)))):  # & (sum(d_forward<0.1) == 19)):
            if (sum(d_backward > 1 * 3 * level) in list(range(n-1, n+1))):
                refpts.append(dff.index[i])
                rctr += 1

        ############################################################################
        #### Finding probable theft points

        if ((sum(d_forward < -1 * level) in list(range(n-1, n+1)))):  # & (sum(d_forward<0.1) == 19)):
            if (sum(d_backward < -1 * level) in list(range(n-1, n+1))):
                theft_pts.append(dff.index[i])
                ctr += 1
                # print(theft_pts, ctr)

        if (ctr >= 2):
            if ((theft_pts[ctr - 1] - theft_pts[ctr - 2]) in list(range(1, 6))):
                theft_pts.pop(ctr - 2)
                ctr -= 1
                # print (i)

                # dd1.append(0)
                # dd2.append(0)
    # dff['dd1'] = pd.Series(dd1)
    # dff['dd2'] = pd.Series(dd2)
    print(len(theft_pts))
    return theft_pts, refpts


def predit_MissingData(df_old, df_cleaned):
    ### Combining all data on common axiz
    ### After removing noise and OFF_State data, replacing them with last known predicted value.

    j = 0  ## Counter for cleaned Data
    i = 0  ## Counter for Old_df

    predict_Data = []
    lastdata_value = df_old.loc[0, 'fuelVoltage']

    while i < (len(df_old)):

        if j < len(df_cleaned):

            if (df_cleaned.loc[j, 'datetime'] > df_old.loc[i, 'datetime']):
                predict_Data.append(lastdata_value)
                print ('i = ',i)

            elif (df_cleaned.loc[j, 'datetime'] <= df_old.loc[i, 'datetime']):
                lastdata_value = df_old.loc[i, 'fuelVoltage']
                j += 1
                predict_Data.append(lastdata_value)
                print (j)
                # print ('*j = ',j)

        else:
            predict_Data.append(df_old.loc[i, 'fuelVoltage'])
            # print (i)

        i += 1
    df_old['predictFuelVolt'] = pd.Series(predict_Data)

    return df_old

#########################################################################
#### Function to find average to maximum fuel Consumption Rate,
#### To be used to validate theft points
#########################################################################
def findMax_decayRate(cleanDf, fuelMax, fuelMin):
    i = 0
    df_window = (fuelMax - fuelMin) / 500
    avgDT = []
    indexlst = []

    while i < len(cleanDf):
        df = 0
        dfRef = cleanDf.fuelVoltage[i]      ### Reference Fuel Voltage
        dsRef = cleanDf.distance[i]         ### Reference Distance position

        ### The loop windows work till the fuelVoltage drop reaches the desired drop level i.e. 'df_window'
        ### After that, it calculates the distance travelled with that consumption of fuel
        ### And calculate average consumtion rate/Voltage decay rate, dF/dS
        ### Incase, distance travelled is 'zero', dF/dS is randomly assigned value '1000'
        while (df <= df_window and i < len(cleanDf)):
            df = dfRef - cleanDf.fuelVoltage[i]
            if df < 0:
                dfRef = cleanDf.fuelVoltage[i]
            i += 1
            # print(i)

        if i < len(cleanDf):
            ds = cleanDf.distance[i] - dsRef
            if df > 0:
                # print (i)
                if ds == 0:
                    ds = df / 1000
                    # print("df = ",df,"****",i)
                avg = df / ds
                avgDT.append(avg)
                indexlst.append(cleanDf.index[i])

    med = pd.Series(avgDT).median()
    mean = pd.Series(avgDT).mean()
    avg = pd.Series(avgDT)
    ##################################################################
    ### Max Allowable decayRate = Median  +  3* MedianDeviation
    ##################################################################
    max_decayRate = 1000*(avg.median() + 3 * abs(avg - avg.median()).median())

    return max_decayRate

#########################################################################
#### Function to find longterm average fuel Consumption Rate,
#### To be used to validate theft points
#########################################################################
def findAvg_RefuelDistance(cleanDf, curr_fuelVoltage, fuelMax, fuelMin):
    i = 0
    df_window = (fuelMax - fuelMin) / 10
    avgDT = []
    indexlst = []

    while i < len(cleanDf):
        df = 0
        dfRef = cleanDf.fuelVoltage[i]  ### Reference Fuel Voltage
        dsRef = cleanDf.distance[i]  ### Reference Distance position

        ### The loop windows work till the fuelVoltage drop reaches the desired drop level i.e. 'df_window'
        ### After that, it calculates the distance travelled with that consumption of fuel
        ### And calculate average consumtion rate/Voltage decay rate, dF/dS
        ### Incase, distance travelled is 'zero', dF/dS is randomly assigned value '1000'
        while (df <= df_window and i < len(cleanDf)):
            df = dfRef - cleanDf.fuelVoltage[i]
            if df < 0:
                dfRef = cleanDf.fuelVoltage[i]
            i += 1
            # print(i)

        if i < len(cleanDf):
            ds = cleanDf.distance[i] - dsRef
            if df > 0:
                # print (i)
                if ds == 0:
                    ds = df / 1000
                    # print("df = ",df,"****",i)
                avg = df / ds
                avgDT.append(avg)
                indexlst.append(cleanDf.index[i])

    ##################################################################
    ### Calculating LongTerm Avg Decay Rate
    avg_decayRate_kM = 1000 * pd.Series(avgDT).median()

    #### Avg Refueling Distance, or avg distance that can be travelled in current fuellevel.
    refuel_Dist = (curr_fuelVoltage - fuelMin)/avg_decayRate_kM
    return refuel_Dist

#########################################################################
#### Function to find average to maximum fuel Consumption Rate,
#### To be used to validate theft points
#########################################################################
def findMax_decayRate(cleanDf, fuelMax, fuelMin):
    i = 0
    df_window = (fuelMax - fuelMin) / 500
    avgDT = []
    indexlst = []

    while i < len(cleanDf):
        df = 0
        dfRef = cleanDf.fuelVoltage[i]      ### Reference Fuel Voltage
        dsRef = cleanDf.distance[i]         ### Reference Distance position

        ### The loop windows work till the fuelVoltage drop reaches the desired drop level i.e. 'df_window'
        ### After that, it calculates the distance travelled with that consumption of fuel
        ### And calculate average consumtion rate/Voltage decay rate, dF/dS
        ### Incase, distance travelled is 'zero', dF/dS is randomly assigned value '1000'
        while (df <= df_window and i < len(cleanDf)):
            df = dfRef - cleanDf.fuelVoltage[i]
            if df < 0:
                dfRef = cleanDf.fuelVoltage[i]
            i += 1
            # print(i)

        if i < len(cleanDf):
            ds = cleanDf.distance[i] - dsRef
            if df > 0:
                # print (i)
                if ds == 0:
                    ds = df / 1000
                    # print("df = ",df,"****",i)
                avg = df / ds
                avgDT.append(avg)
                indexlst.append(cleanDf.index[i])

    med = pd.Series(avgDT).median()
    mean = pd.Series(avgDT).mean()
    avg = pd.Series(avgDT)
    ##################################################################
    ### Max Allowable decayRate = Median  +  3* MedianDeviation
    ##################################################################
    max_decayRate = 1000*(avg.median() + 3 * abs(avg - avg.median()).median())

    return max_decayRate


def generate_PredictTable(df_cleaned, theft_pts, max_DecayRate, fuelMax, fuelMin):
    result_df = pd.DataFrame()
    result_df['theft_index'] = [df_cleaned.index[i] for i in theft_pts]
    result_df['lat'] = [df_cleaned.lat[i] for i in theft_pts]
    result_df['long'] = [df_cleaned.long[i] for i in theft_pts]
    result_df['theft_time'] = [df_cleaned.datetime[i] for i in theft_pts]
    
    result_df['fuel_VoltageJump'] = [(df_cleaned.fuelVoltage[i] - df_cleaned.fuelVoltage[i + 1]) for i in theft_pts]
    result_df['fuel_VoltageJump(%)'] = [100*(df_cleaned.fuelVoltage[i] - df_cleaned.fuelVoltage[i + 1])/(fuelMax-fuelMin) for i in theft_pts]
    result_df['fuel_VoltageJump(%)'] = result_df['fuel_VoltageJump(%)'].apply(lambda x: round(x, 2))

    result_df['dist_jump(KM)'] = [(df_cleaned.distance[i + 1] - df_cleaned.distance[i]) * (.001) for i in theft_pts]
    result_df['time_jump'] = [(df_cleaned.datetime[i + 1] - df_cleaned.datetime[i]) for i in theft_pts]

    result_df['fuelVoltagePerKM'] =  result_df['fuel_VoltageJump'] /result_df['dist_jump(KM)']

    #result_df.to_csv(r"G:\Analytics\FuelAnalysis\results\reults.csv")

    # plt.plot(result_df.theft_time, result_df.FuelPerKM)
    # plt.semilogy()
    # plt.show()
    result_df = result_df[result_df['fuelVoltagePerKM'] >max_DecayRate]
    result_df = result_df.reset_index(drop=True)
    print (len(result_df))
    return result_df


def generate_ReFuelTable(df_cleaned, ref_pts, fuelMax, fuelMin):
    refuel_df = pd.DataFrame()
    refuel_df['ReFuel_index'] = [df_cleaned.index[i] for i in ref_pts]
    refuel_df['lat'] = [df_cleaned.lat[i] for i in ref_pts]
    refuel_df['long'] = [df_cleaned.long[i] for i in ref_pts]
    refuel_df['ReFuel_time'] = [df_cleaned.datetime[i] for i in ref_pts]

    refuel_df['fuel_VoltageJump'] = [(df_cleaned.fuelVoltage[i+1] - df_cleaned.fuelVoltage[i]) for i in ref_pts]
    refuel_df['fuel_VoltageJump(%)'] = [100 * (df_cleaned.fuelVoltage[i+1] - df_cleaned.fuelVoltage[i]) / (fuelMax - fuelMin) for i in ref_pts]
    refuel_df['fuel_VoltageJump(%)'] = refuel_df['fuel_VoltageJump(%)'].apply(lambda x: round(x,2))
    refuel_df['dist_jump(KM)'] = [(df_cleaned.distance[i + 1] - df_cleaned.distance[i]) * (.001) for i in ref_pts]
    refuel_df['time_jump'] = [(df_cleaned.datetime[i + 1] - df_cleaned.datetime[i]) for i in ref_pts]

    # result_df.to_csv(r"G:\Analytics\FuelAnalysis\results\reults.csv")

    # plt.plot(result_df.theft_time, result_df.FuelPerKM)
    # plt.semilogy()
    # plt.show()
    print(len(refuel_df))
    return refuel_df


####################################################################
#### Function to generate Smooth Curve
#####################################################################
def generate_SmoothCurve(df_clean):
    from scipy.fftpack import fft
    from scipy.signal import butter, lfilter, freqz

    normdata = df_clean.fuelVoltage / df_clean.fuelVoltage.max()
    yvolt = fft(normdata)

    def butter_lowpass_filter(data, cutoff, fs, order=5, ftype=False):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=ftype)
        y = lfilter(b, a, data)
        return y

    fs = 500  # Sampling Frequency
    order = 5  # Order of Filter
    cutoff = 1  # Filter Cut-off Frequency

    # Calling Butterworth filter
    y_smooth = df_clean.fuelVoltage.max() * (butter_lowpass_filter(normdata, cutoff, fs, order, ftype=False))
    df_clean['SmoothVoltage'] = y_smooth
    smooth_Df = df_clean[['datetime','fuelVoltage', 'SmoothVoltage']]

    return smooth_Df