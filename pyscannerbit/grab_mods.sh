#!/bin/bash

# Grab modified pyscannerbit python files from site_packages directory
# During development it is faster to directly modify files there, and
# then just grab the changes back to here.

cp ~/anaconda3/envs/general/lib/python3.6/site-packages/pyscannerbit/*.py pyscannerbit/
#cp /home/farmer/anaconda3/lib/python3.7/site-packages/pyscannerbit/*.py pyscannerbit
