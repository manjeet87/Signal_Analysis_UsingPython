import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import butter, lfilter

desired_width = 360
pd.set_option('display.width', desired_width)


def Clean_NoiseData(dff, fuelMax, fuelMin, nds):
    y = pd.Series(dff.fuelVoltage)

    ## Neighbourhood Distance
    Nds = abs(0.01*(fuelMax - fuelMin))
    if nds:
        Nds = abs(nds)

    dforward = pd.DataFrame()
    dbackward = pd.DataFrame()
    for i in range(1, 10):
        dforward['d' + str(i)] = abs(y - y.shift(i))
        dbackward['d' + str(i)] = abs(y - y.shift(-1 * i))
    dbackward = dbackward.fillna(0)
    dforward = dforward.fillna(0)

    dff2 = dff[(dforward.d1 <= Nds) & (dbackward.d1 <= Nds) & (dforward.d2 <= 2 * Nds) & (dbackward.d2 <= 2 * Nds) &
               (dforward.d3 <= 3 * Nds) & (dbackward.d3 <= 3 * Nds) & (dforward.d4 <= 4 * Nds) & (dbackward.d4 <= 4 * Nds) &
               (dforward.d5 <= 5 * Nds) & (dbackward.d5 <= 5 * Nds) & (dforward.d6 <= 6 * Nds) & (dbackward.d6 <= 6 * Nds) &
               (dforward.d7 <= 7 * Nds) & (dbackward.d7 <= 7 * Nds) & (dforward.d8 <= 8 * Nds) & (dbackward.d8 <= 8 * Nds)]
    dff2 = dff2.reset_index(drop=True)

    dff2.loc[:,('fuelVoltage_Percent')] = dff2.fuelVoltage.apply(lambda x: round((100*(x-fuelMin)/(fuelMax - fuelMin)),2))
    #print(dff2.head())
    return dff2


def jump_point(dff, level, fuelMax, fuelMin, neb_dist):

    if len(dff)<=200:
        raise Exception("Length of DataSet is too small for any theft & refuel point analysis.. Plz Enter valid Dataset")

    x = np.array(dff.index)
    y = np.array(dff.fuelVoltage)
    d = np.array(dff.distance)
    
    #######################################################################
    ## No. of Neighbourhood pts, 'n', dependent on sampling Rate of IoT device
    ## Criteria Set = Avg No. of points inside 10Km window will be considered

    n = int(8000/(dff.distance.max()/len(dff)))
    n = max(n,10)
    n = min(n,40)
    print('Points per 10Km: ', n)

    level = level*(fuelMax - fuelMin)
    if neb_dist:
        level = neb_dist
    theft_pts = []
    refpts = []
    rctr = 0
    ctr = 0
    i = 0

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
    refpts = pd.Series(refpts)
    shift = abs(refpts.shift(-1) - refpts)


    print(len(theft_pts))
    return theft_pts, refpts

def avg_decayRate(cleanDf,fuelMax, fuelMin):
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

################################################################################
#### Function to generate table for probable theft events with position and time
#################################################################################
def generate_TheftTable(df_cleaned, theft_pts, max_DecayRate, fuelMax, fuelMin):
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

###############################################################################
### Function to check for invalid Refuel Points and remove them
###############################################################################
def check_refpts_dropList(refdt):

    refdt['shift'] = abs(refdt['refuel_index'].shift(-1) - refdt['refuel_index'])
    #refdt['dist'] = pd.Series(df_clean2.distance[refpts].reset_index(drop=True))
    i = 0
    dropIndexList = []
    while i in range(len(refdt)):

        if refdt.loc[i, 'shift'] <= 5:
            i += 1
            while (refdt.loc[i, 'shift'] <= 5):
                dropIndexList.append(refdt.loc[i,'refuel_index'])
                i += 1
        i += 1

    refdt['drop'] = refdt.refuel_index.apply(lambda x: 'Y' if (x in dropIndexList) else 'N')
    #print (refdt)
    refdt = refdt[refdt['drop'] == 'N'].reset_index(drop= True)
    return refdt, dropIndexList


def generate_ReFuelTable(df_cleaned, ref_pts, fuelMax, fuelMin):

    refuel_df = pd.DataFrame()
    refuel_df['refuel_index'] = [df_cleaned.index[i] for i in ref_pts]
    refuel_df['lat'] = [df_cleaned.lat[i] for i in ref_pts]
    refuel_df['long'] = [df_cleaned.long[i] for i in ref_pts]
    refuel_df['refuel_time'] = [df_cleaned.datetime[i] for i in ref_pts]

    refuel_df, dropIndex = check_refpts_dropList(refuel_df.copy())
    df_cleaned = df_cleaned.drop(dropIndex)


    for i,j in zip(refuel_df.refuel_index, refuel_df.index):
        success = 0
        pos = 1
        while(success ==0):
            try:
                refuel_df.loc[j,'fuel_VoltageJump'] = (df_cleaned.fuelVoltage[i+pos] - df_cleaned.fuelVoltage[i])
                refuel_df.loc[j,'fuel_VoltageJump(%)'] = 100 * (df_cleaned.fuelVoltage[i+pos] - df_cleaned.fuelVoltage[i]) / (fuelMax - fuelMin)
                refuel_df.loc[j,'dist_jump(KM)'] = (df_cleaned.distance[i + pos] - df_cleaned.distance[i]) * (.001)
                refuel_df.loc[j,'time_jump'] = (df_cleaned.datetime[i + pos] - df_cleaned.datetime[i])
                success =1
            except:
                pos +=1

    refuel_df['fuel_VoltageJump(%)'] = refuel_df['fuel_VoltageJump(%)'].apply(lambda x: round(x,2))

    # result_df.to_csv(r"G:\Analytics\FuelAnalysis\results\reults.csv")

    # plt.plot(result_df.theft_time, result_df.FuelPerKM)
    # plt.semilogy()
    # plt.show()
    print(len(refuel_df))
    return refuel_df


####################################################################
#### Function to generate Smooth Curve
#####################################################################
def generate_SmoothCurve(dff, fuelMax, fuelMin):
    dff = dff.copy()

    #fmax = dff.fuelVoltage.max()
    # normdata = dff.fuelVoltage / fmax
    #   # add noise to the signal
    #
    # def butter_lowpass_filter(data, cutoff, fs, order=5, ftype=False):
    #     nyq = 0.5 * fs
    #     normal_cutoff = cutoff / nyq
    #     b, a = butter(order, normal_cutoff, btype='low', analog=ftype)
    #     y = lfilter(b, a, data)
    #     return y
    #
    # fs = 500  # Sampling Frequency
    # order = 5  # Order of Filter
    # cutoff = 10  # Filter Cut-off Frequency

    ### Calling Median Filter
    y_smooth = sp.signal.medfilt(dff.fuelVoltage, 99)
    dff['fuelVoltage'] = pd.Series(y_smooth)

    # # Calling Butterworth filter
    # y_smooth2 = fmax * (butter_lowpass_filter(normdata, cutoff, fs, order, ftype=False))
    # dff['SmoothVoltage2'] = y_smooth2
    #smooth_Df = df_clean[['datetime','fuelVoltage', 'SmoothVoltage']]
    dff = dff[abs(dff.fuelVoltage - dff.fuelVoltage.median()) <= 2.4*dff.fuelVoltage.std()]
    dff = dff.reset_index(drop=True)
    dff.loc[:, ('fuelVoltage_Percent')] = dff.fuelVoltage.apply(lambda x: round((100 * (x - fuelMin) / (fuelMax - fuelMin)), 2))

    return dff