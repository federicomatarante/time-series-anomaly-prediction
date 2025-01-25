#!/bin/sh

mkdir -p data/ESA-Mission1/
curl 5.75.134.176:3333/3_months.train.csv > data/ESA-Mission1/3_months.train.tsv
