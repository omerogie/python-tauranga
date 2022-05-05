# python-tauranga
main.py:
Calculates coherence between groundwater and rainfall data.
Author: Rogier Westerhoff

The main.py script currently does this:
 - read in groundwater level, from csv file or a stored (pickled) Pandas dataframe (read_gw_data);
 - read in rainfall data, from csv file or a stored (pickled) Pandas dataframe (read_rain_data);
 - calculate power spectra Px and Py and ultimately their coherence Cxy (make_coherence_spectra)
 - write results to a CSV file
I kept the intermediate testing files, which should also still work (no guarantee though). These are now #-commented.
