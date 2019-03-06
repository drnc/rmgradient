#! /bin/sh
cat test_data/skypoints | xargs python rmgradient.py -i test_data/sky.tif -o test_data/sky_out.tif -b test_data/sky_background.tif -s 3.0 -c 6 -p
