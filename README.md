# Recommendation-On-Yelp-Dataset
This repo implements and evaluates Nearest Neighbor and Variational Autoencoders for Collaborative Filtering models on Yelp Academic Dataset. 

# Example Commands

### Data Split
```
python getyelp.py --enable_implicit --ratio 0.5,0.2,0.3 --data_dir data/yelp/ --data_name yelp_academic_dataset_review.json
```

### Single Run
```
python main.py --path data/yelp/ --model VAE-CF --epoch 200 --lamb 0.0000001 --rank 100
```

### Hyper-parameter Tuning

Split data in experiment setting, and tune hyper parameters based on yaml files in `config` folder. 

```
python getyelp.py --enable_implicit --ratio 0.5,0.2,0.3 --data_dir data/yelp/ --data_name yelp_academic_dataset_review.json
python tune_parameters.py --path data/yelp/ --save_path vae_tuning.csv --grid config/vae.yml --gpu
```

### Final Run
```
python final_performance.py --path data/yelp/ --problem yelp --name final_result.csv
```

### Reference
* Variational Autoencoders for Collaborative Filtering [[paper]](https://arxiv.org/pdf/1802.05814.pdf)
* Noise Contrastive Estimation for Scalable Linear Models for One-Class Collaborative Filtering [[paper]](https://arxiv.org/pdf/1811.00697.pdf)
