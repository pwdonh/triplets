model_comparison:
	fit_triplets_crossval.py --config_path configs/triplets3ab_crossval.cfg
	python proc_accents_1_model_comparison.py

model_exploration:
	fit_triplets.py --config_path configs/triplets3ab_crossval/best.cfg
	recon_embedding.py --config_path configs/triplets3ab_crossval/best.cfg
	python proc_accents_2_model_exploration.py

model_comparison_updated:
	fit_triplets_crossval.py --config_path configs/triplets3ab_crossval_rating.cfg
	python proc_accents_3_model_comparison_updated.py