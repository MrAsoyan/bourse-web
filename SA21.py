import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from scipy.signal import savgol_filter as savitzky_golay
from matplotlib.dates import date2num
from scipy.interpolate import interp1d
import time 
import ipywidgets as wg
from scipy.signal import savgol_filter
import yfinance as yf
import requests

def execution_time(st):
    print(f"\033[94mExecution time {time.time() - st:.1f} s\033[0m")

# import yfinance as yf

def date2angle(dates):
    # Ensure dates are in a pandas Series to handle both single values and arrays
    if not isinstance(dates, pd.Series):
        dates = pd.to_datetime(dates)
        
     # Calculate if each year is a leap year
    leap_year = (dates.year % 4 == 0) & (dates.year % 100 != 0) | (dates.year % 400 == 0)

    # Calculate the angle for each date
    angles = np.where(leap_year,                               # condition
                      dates.dayofyear * 2*np.pi/366,    # choix 1 si condition vrai
                      dates.dayofyear * 2*np.pi/365)    # choix 2 sinon
    return angles

def numofday2date(numofday, year=int(time.strftime("%Y"))):
    date = pd.to_datetime(f'{year}-01-01') + pd.to_timedelta(numofday - 1, unit='D') #
    date = date.strftime("%d-%b-%Y") if date.year==year else None
    return date

def normalizer(s):
    """
    s : serie, list, array
    
    Rnevoie s normalisée au format Pd.Series.
    """
    serie = pd.Series(s)
    serie_normalised = (serie - serie.min())/(serie.max()-serie.min())
    return serie_normalised

# # Récupérer les données historiques 
# def get_data(symbol, start, end, interval='1d'):
#     start_date = pd.to_datetime(start)
#     end_date = pd.to_datetime(end)
#     df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
#     df = df[['Close', 'Volume']].copy()
#     df['Angles'] = date2angle(df.index)
#     df['Close_norm'] = normalizer(df.Close)

#     return [symbol, df]

# Récupérer les données historiques 
def get_data(symbol, start, end, interval='1d'):
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
    df.columns = df.columns.get_level_values(0) # Keep only first level (features), NO MULTIPLE INDEX
    # df = df[['Close', 'Volume']].copy()
    df = pd.DataFrame ({'Price' : df[['Open', 'High', 'Low', 'Close']].mean(axis=1), 'Volume' : df['Volume']})
    df['Angles'] = date2angle(df.index)
    df['Price_norm'] = normalizer(df.Price)
    df['SavGol'] = savgol_filter(df.Price, window_length=30, polyorder=4)
    # df['SGMA'] = df['SavGol'].rolling(window=3, center=True).mean()

    return [symbol, df]

token_f="d0dmd21r01qm1l9vtk0gd0dmd21r01qm1l9vtk10"
def get_data_finnhub(symbol, start, end, token=token_f):
    # Step 1: Convert date to UNIX timestamps (required by Finnhub)
    start_ts = int(pd.to_datetime(start).timestamp())
    end_ts = int(pd.to_datetime(end).timestamp())

    # Step 2: Fetch OHLCV candles from Finnhub
    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": symbol,
        "resolution": "D",
        "from": start_ts,
        "to": end_ts,
        "token": token
    }

    response = requests.get(url, params=params).json()

    if response.get("s") != "ok":
        raise ValueError("Data fetch failed from Finnhub.")

    # Step 3: Create DataFrame
    df = pd.DataFrame({
        'Open': response['o'],
        'High': response['h'],
        'Low': response['l'],
        'Close': response['c'],
        'Volume': response['v']
    }, index=pd.to_datetime(response['t'], unit='s'))

    # Step 4: Match your original logic
    df['Price'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    df['Angles'] = date2angle(df.index)                     # keep your custom function
    df['Price_norm'] = normalizer(df['Price'])              # keep your normalizer function
    df['Savgol'] = savgol_filter(df['Price'], window_length=30, polyorder=4)

    return [symbol, df]

token = "b68dbe569c8242a6a44a1febe7ffef7f"
def get_data_twelvedata(symbol, start, end, api_key=token):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "start_date": start,
        "end_date": end,
        "apikey": api_key,
        "outputsize": 5000
    }

    response = requests.get(url, params=params).json()

    if "values" not in response:
        raise ValueError(f"Data fetch failed: {response.get('message', 'Unknown error')}")

    # Build DataFrame
    df = pd.DataFrame(response["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()

    # Convert price columns to float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Compute average price
    df["Price"] = df[["open", "high", "low", "close"]].mean(axis=1)

    # Add features
    df["Angles"] = date2angle(df.index)            # Your custom function
    df["Price_norm"] = normalizer(df["Price"])     # Your custom function
    df["SavGol"] = savgol_filter(df["Price"], window_length=30, polyorder=4)

    return [symbol, df]


def SL_limit(price, leverage=1):
    y1 = price*(1 - 1/(2*leverage)) #price = y0
    return y1

def SL_limit_SHORT(price, leverage=1):
    y1 = price*(1 + 1/(2*leverage)) #price = y0
    return y1

def polar_plot_df_graph_only(df, which='Price'):
    """
    Takes a pd.DataFrame and polarplots its 'Price' and 'SavGol' attributes.

    df : data frame that contains the dates, prices, SavGol of prices and the volume
    which : option that allows to choose 'Price', 'SavGol', 'Both', 'Volume'
    """
    
    # Prepare the polar plot 
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # Plot data for each year with increasing alpha for overlap effect
    # Each year must be ploted seperately to have that overlap effect
    years = df.index.year.unique() # years = [2017,...,2023,2024] 
    for year in years:
        theta = df.Angles.loc[df.index.year == year].values
        R_Price = df.Price.loc[df.index.year == year].values
        R_SavGol = df.SavGol.loc[df.index.year == year].values
        R_Volume = df.Volume.loc[df.index.year == year].values
        R_Vol_norm = (R_Volume - R_Volume.min())/(R_Volume.max()-R_Volume.min()) * R_Price.max()
        # R_Price_norm = 1 + df.Price_norm.loc[df.index.year == year].values # +1 pour afficher le long du cercle unité
        
        # ax[0,0].plot(theta, R_Price, '-', color='C0', linewidth=2)        
        if('Price' in which):
            if(year in [2021,2022]):
                ax.fill_between(theta, R_Price, color='red', alpha=0.25)
            else:
                ax.fill_between(theta, R_Price, color='C0', alpha=0.25)
        if('SavGol' in which):
            if(year in [2021,2022]):
                ax.fill_between(theta, R_SavGol, color='red', alpha=0.25)
            else:
                ax.fill_between(theta, R_SavGol, color='C0', alpha=0.25)
        if('Volume' in which):
            if(year in [2021,2022]):
                ax.fill_between(theta, R_Vol_norm, color='red', alpha=0.25)
            else:
                ax.fill_between(theta, R_Vol_norm, color='C0', alpha=0.25)
            
    
    # Set the tick positions for each month
    month_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Add the month labels to the plot
    ax.set_xticks(month_angles)
    ax.set_xticklabels(month_labels)
    # ax.set_yticklabels([])
    ax.set_thetamin(0)
    ax.set_thetamax(360)

    return fig, ax #to be able to add some other graphs 

def fill_nan(a):
    """
    Remplace toutes les valeurs NaN par la moyenne de la valeur précédente et suivante :
    - Les NaN au début de a sont remplis par la première valeur non-NaN.
    - Les NaN à la fin de a sont remplis par la dernière valeur non-NaN.
    - Les NaN (ou la séquence) au milieu de a sont remplis par la moyenne des valeurs encadrantes le/les NaNs.
    """
    if isinstance(a, pd.Series):
        a = a.to_numpy()
    
    # Remplacer le premier élément s'il est NaN
    if pd.isna(a[0]):
        for i in range(1, len(a)):
            if not pd.isna(a[i]):
                a[0] = a[i]
                break

    # Remplacer le dernier élément s'il est NaN
    if pd.isna(a[-1]):
        for i in range(len(a)-2, -1, -1):
            if not pd.isna(a[i]):
                a[-1] = a[i]
                break

    # Remplacer les NaN intermédiaires par la moyenne des éléments précédents et suivants
    i = 1
    while i < len(a) - 1:
        if pd.isna(a[i]):
            start = i
            while i < len(a) - 1 and pd.isna(a[i]):
                i += 1
            end = i
            if start > 0 and end < len(a):
                moyenne = (a[start-1] + a[end]) / 2
                for j in range(start, end):
                    a[j] = moyenne
        i += 1

    return a

def correlations(DataFrame, montrer_les_valeurs_sup_a=None):
    """
    DataFrame : doit contenir un indice contant les dates et au moins une colonne nommée 'Price'
    """
    df = DataFrame.copy()
    df['Year'] = df.index.year
    
    # Pivot the DataFrame to get a wide format where each column is a year
    df_pivot = df.pivot_table(index=df.index.dayofyear, columns='Year', values='Price') #years are integers
    
    # Calculate the correlation matrix between years
    correlation_matrix = df_pivot.corr()

    if(montrer_les_valeurs_sup_a != None):
        correlation_matrix = correlation_matrix[np.abs(correlation_matrix) > montrer_les_valeurs_sup_a]
    
    return correlation_matrix

def interference_index(df, option='sum', window=37, on='SavGol', win_step=1, smash_power=2):
    """
    df     : columns must containt index=dates, 'Price', 'SavGol', 'Volume'
    option : 'sum' or 'prod'
    window : window size for moving average for tendency
    on     : 'Price', 'SavGol', 'Volume'
    smash_power : if option='prod', interference pattern raise to this power

    returns the interference curve (366 points) normaliazed to the max value of the price.
    If the option is 'sum' : returns only interference pattern.
    If the option is 'prod' : returns the interference pattern with and an index which is the max or interf prod before scaling.
    """

    # Extract the year and month from the 'Date' column
    data = df.copy()
    data['Year'] = data.index.year
    # df['Month'] = df.index.month
    # df['Year-Month'] = df.index.to_period('M')
    
    # Pivot the DataFrame to get a wide format where each column is a specific Year-Month
    if(on=='SavGol'):
        kk = data.pivot_table(index=data.index.dayofyear, columns='Year', values='SavGol')
    elif(on=='Price'):
        kk = data.pivot_table(index=data.index.dayofyear, columns='Year', values='Price')
    elif(on=='Volume'):
        kk = data.pivot_table(index=data.index.dayofyear, columns='Year', values='Volume')
    # Create a complete index from 1 to 366
    complete_index = pd.Index(np.arange(1, 367), name=kk.index.name) # 366 si l'année n'est pas bisextile
    
    # Remplissage des dates manquantes _ Reindex the DataFrame to include any missing days
    kk = kk.reindex(complete_index)
    
    # remplissage des valeurs manquantes : preparation du tableau
    pp = pd.DataFrame(columns=kk.columns, index=kk.index)
    for i in data.index.year.unique(): #[2028-2023]
        pp[i] = fill_nan(kk[i].values)
    
    # interference index année par année
    # on fait : courbes - tendances, puis normalisation entre [0,2]
    tt = (pp-pp.rolling(window=window, center=True, step=win_step).mean())
    tt_copy = tt.copy()
    tt = 2*((tt-tt.min())/(tt.max()-tt.min()))
    tt = tt.fillna(0)

    # inticateur 1 : prod (un peu complexe : si le patterne d'intérfèrence multiplicative est en dessous de 1 => ANNULER !)
    interference_prod = tt.prod(axis=1)**smash_power
    interference_prod_index = np.max(interference_prod) # compris dans [0, 2^N], confiant si dans [1, 2^N], où N = années
    if(interference_prod_index > 1):
        interference_prod = interference_prod / np.abs(interference_prod).max() * data.Price.max() # Scaling [-Price.max, Price.max]
    else:
        interference_prod[:] = 0 
    if(option == 'prod'):
        return interference_prod

    # inticateur 2
    interference_sum = tt.sum(axis=1)
    interference_sum = interference_sum / np.abs(interference_sum).max() * data.Price.max() # Scaling [-Price.max, Price.max]
    if(option == 'sum'):
        return interference_sum

    # indicateur 3
    if(option == 'heart_attack'):
        interference_HA = 4*((tt_copy-tt_copy.min())/(tt_copy.max()-tt_copy.min())) - 2 #price -tendance
        interference_HA = interference_HA.fillna(0)
        interference_HA = interference_HA.prod(axis=1) # multiplie les années entre-elles
        interference_HA_index = np.max(interference_HA) # compris dans [0, 2^N], confiant si dans [1, 2^N], où N = années
        # print("Hesa le maxy : ", interference_HA_index)
        if(interference_HA_index > 1):
            interference_HA = interference_HA / np.abs(interference_HA).max() * data.Price.max() # Scaling [-Price.max, Price.max]
        else:
            interference_HA[:] = 0
        return interference_HA
    
    # ax.plot(np.linspace(0, 360, 367)[1:]*np.pi/180, interference, color='k') # erreur d'un jour à cause des années bisextiles

def spirality_index(df, option='M'):
    """
    df     : must contain dates in index and the 'Price'
    option : 'M', '3M' or 'A. Meaning Monthly, 3-Monthly and Annual
    on     : 'Price' or 'SavGol'

    returns the list of followings :
    0 : the spirality
    1 : daily angles of dates
    2 : daily values, 30 day moving average values 
    3 : monthly angles of dates
    4 : monthly values

    Plot    1 with 2 : daily   angles with daily values
    Scatter 3 with 4 : monthly angles with monthly values
    """
    kk_R = df.Price.copy()
    kk_theta = date2angle(df.index)
    
    # SpiralityM
    kk_mean30 = kk_R.rolling(window=30, center=True).mean()
    kk_monthly_means = kk_mean30.resample('ME').mean()
    kk_monthly_means.index = kk_monthly_means.index - pd.Timedelta(days=15) # Adjust the index to be the 15th of each month
    spiralityM = (kk_monthly_means.shift(-1)/kk_monthly_means).mean()
    if(option=='M'):
        return [spiralityM, kk_theta, kk_mean30, date2angle(kk_monthly_means.index), kk_monthly_means.values]
        
    # Spirality3M
    kk_mean90 = kk_R.rolling(window=90, center=True).mean()
    kk_3_monthly_means = kk_mean90.resample('3ME').mean()
    kk_3_monthly_means.index = kk_3_monthly_means.index + pd.Timedelta(days=15) # Adjust the index to be the 15th of each month
    spirality3M = (kk_3_monthly_means.shift(-1)/kk_3_monthly_means).mean()
    if(option=='3M'):
        return [spirality3M, kk_theta, kk_mean90, date2angle(kk_3_monthly_means.index), kk_3_monthly_means.values]
        
    # SpiralityA
    kk_meanA = kk_R.rolling(window=90, center=True).mean()
    kk_A_monthly_means = kk_meanA.resample('YE').mean()
    kk_A_monthly_means.index = kk_A_monthly_means.index + pd.Timedelta(days=180) # Adjust the index to be the 15th of each month
    spiralityA = (kk_A_monthly_means.shift(-1)/kk_A_monthly_means).mean()
    if(option=='A'):
        return [spiralityA, kk_theta, kk_meanA, date2angle(kk_A_monthly_means.index), kk_A_monthly_means.values]

def min_locator(f, spreading_level=30):  # only one minima in 30 day interval
    """
    f : pd.Series

    returns : serie of f.index, f.values of minima points
    """
    from scipy.signal import argrelextrema
    
    # Finding local minima
    minima_indices = argrelextrema(f.values, np.less, order=spreading_level)[0]

    f_copy = pd.Series(data=f.values[minima_indices], index=f.index[minima_indices])
    return f_copy
    #return f.index[minima_indices], f.values[minima_indices]

def max_locator(f, spreading_level=30):  # only one minima in 30 day interval
    from scipy.signal import argrelextrema
    
    # Finding local maxima
    maxima_indices = argrelextrema(f.values, np.greater, order=spreading_level)[0]
    
    f_copy = pd.Series(data=f.values[maxima_indices], index=f.index[maxima_indices])
    return f_copy
    # return f.index[minima_indices], f.values[minima_indices]


# x = np.linspace(0,12,1000)
# y = np.sin(x)

# plt.plot(x,y)
# plt.show()


@wg.interact(a = (1,5,1))
def run(a):
    print(a)