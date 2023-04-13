#!/bin/bash

#
# Download precipitation index change factors from World Bank Climate Change Knowledge Portal
#
rm -f get_all.sh
touch get_all.sh

for variable in "rx1day" "rx5day" "rxmonth"
do
  for iso in "DMA" "GRD" "LCA" "VCT"
  do
    for period in "2010-2039" "2035-2064" "2060-2089" "2070-2099"
    do
      for ssp in "ssp119" "ssp126" "ssp245" "ssp370" "ssp585"
      do
        for calculation in "changefactorfaep5yr" "changefactorfaep10yr" "changefactorfaep20yr" "changefactorfaep25yr" "changefactorfaep50yr" "changefactorfaep100yr"
        do
          echo "wget \"https://climateknowledgeportal.worldbank.org/climate_tab/extremes/climatology/$variable/median/annual/all/$calculation/$ssp/$period/country/$iso/85758\" \
            -O \"scraped/extremes_climatology_${variable}_median_annual_${calculation}_${ssp}_${period}_${iso}.csv\"" >> get_all.sh
        done
      done
    done
  done
done

# the loop above generates the temporary get_all.sh, which can then be piped to
# make requests in parallel
cat get_all.sh | parallel
rm get_all.sh
