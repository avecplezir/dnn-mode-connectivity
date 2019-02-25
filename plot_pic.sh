#!/usr/bin/env bash
for i in $(seq 3 4);
do
    python plane_plot.py --dir=plots/middle_point5051_2$i
done
