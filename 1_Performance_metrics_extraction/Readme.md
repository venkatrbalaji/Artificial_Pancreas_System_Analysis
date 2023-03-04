README

Steps to Execute Code:
1. Update the paths to datasets ÒCGMData.csvÓ and ÒInsulinData.csvÓ in main.py (line 10&11 respectively), if they are named different or not in the same directory as main.py,
2. Install ÒpandasÓ package (Òpip install pandasÓ) if not already installed,
3. Run the Ômain.pyÕ python script using CMD Òpython main.pyÓ

NOTE: 
1. The ÒDateÓ and ÒTimeÓ fields in the data sets have been combined to derive a new Index field called Òdate_time_stampÓ of type DateTime.
2. After filtering the necessary fields of CGM data onto a new pandas data frame, Ôgroup-byÕ and ÔslicingÕ techniques (over the date_time_stamp index) are followed for segmentation of the data frame based on MODE (Manual & Auto - slicing), date (Whole days Ð group by) and Time (daytime and overnight - slicing),
3. Percentage of in-range values for all six given cases are calculated for each segment (over the fixed 288 total number of values with missing values replaced by 0Õs) and are added to two result data frame Ð one for manual and one for auto mode.
4. As the last step, to get the overall average percentage of in-range values, the mean of result data frames is calculated and are exported to ÒResults.csvÓ.



