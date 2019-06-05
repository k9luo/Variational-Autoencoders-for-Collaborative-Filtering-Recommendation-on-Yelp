#!/usr/bin/env bash
cd ~/Recommendation-On-Yelp-Dataset
python tune_parameters.py --path data/yelp/ --save_path vae_tuning1.csv --grid config/vae-part1.yml
