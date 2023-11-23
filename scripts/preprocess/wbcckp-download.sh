#!/bin/bash

#
# Download precipitation index change factors from World Bank Climate Change Knowledge Portal
#
rm -f get_all.sh
touch get_all.sh

# clear any empty files
# find ./scraped -empty -delete

for var in "rx1day" "rx5day" "rxmonth"
do
  for stat in "median" "p10" "p90"
  do
    for iso in "DMA" "GRD" "LCA" "VCT"
    do
      for yrs in "2010-2039" "2035-2064" "2060-2089" "2070-2099"
      do
        for ssp in "ssp119" "ssp126" "ssp245" "ssp370" "ssp585"
        do
          for calc in "changefactorfaep5yr" "changefactorfaep10yr" "changefactorfaep20yr" "changefactorfaep25yr" "changefactorfaep50yr" "changefactorfaep100yr"
          do
            echo "wget -nc \"https://climateknowledgeportal.worldbank.org/climate_tab/extremes/climatology/$var/$stat/annual/all/$calc/$ssp/$yrs/country/$iso/85758\" \
              -O \"scraped/extremes_climatology_${var}_${stat}_annual_${calc}_${ssp}_${yrs}_${iso}.csv\"" >> get_all.sh
          done
        done
      done
    done
  done
done

# the loop above generates the temporary get_all.sh, which can then be piped to
# make requests in parallel
cat get_all.sh | parallel -j 8
rm get_all.sh
