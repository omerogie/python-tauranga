# Press Shift+F10 to execute this script

import os
from methods.read_gw_data import read_gw_data
from methods.read_gw_data import read_rain_data
from methods.read_gw_data import make_coherence_spectra

# read gw data and output as dataframe
gw_data_df = read_gw_data(True)  # True: pickled dataframe will be read in. False: csv will be read in

# read rain data and output as dataframe
rain_data_df = read_rain_data(True)  # True: pickled dataframe will be read in. False: csv will be read in

# calculate coherence between rainfall and groundwater and write into new dataframe
df_coherence = make_coherence_spectra(rain_data_df, gw_data_df)
# transpose dataframe and write to csv
print('Writing to csv file...')
df_coherence.T.to_csv(os.path.join(os.getcwd(), 'files/coherences.csv'))
print("Finished!")

### --- optional and testing routines
# read in locations and write into datframe (not used now)
# from methods.read_gw_data import read_rain_locations
# rain_gauge_locs_df = read_rain_locations(True) # 'False' reads csv data and exports to df; 'True' reads in df

# from methods.read_gw_data import merge_dataframes
# merged_df = merge_dataframes(gw_data_df,rain_data_df,True)

# from methods.read_gw_data import gw_power_spectra
# gw_power_spectra()

# from methods.read_gw_data import rain_power_spectra
# name_df = 'rain_data_df'
# rain_power_spectra(name_df)

# print(os.getcwd())
# from methods.read_and_plot_time_series import lucas_script
# lucas_script()
### --- end of optional and testing routines


