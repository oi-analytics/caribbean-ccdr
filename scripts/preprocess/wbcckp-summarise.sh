#!/bin/bash

#
# Summarise downloaded change factors
#
echo "variable,calculation,ssp,epoch,isoa3,country,value" > summary.csv

# pull relevant lines out of downloaded CSVs
grep Dominica scraped/*.csv >> summary.csv
grep Grenada scraped/*.csv >> summary.csv
grep "St. Lucia" scraped/*.csv >> summary.csv
grep "St. Vincent and the Grenadines" scraped/*.csv >> summary.csv

# find/replace various strings to turn this into a valid CSV
sed 's/scraped\/extremes_climatology_//' -i summary.csv
sed 's/_/,/g' -i summary.csv
sed 's/.csv:/,/' -i summary.csv
sed 's/median,annual,//g' -i summary.csv
