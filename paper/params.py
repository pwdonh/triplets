import pandas as pd

config_path = 'configs/triplets3ab_crossval.cfg'

rater_groups = [
    'english_uk', 'german', 'french'
]

speaker_groups = [
    'english_us', 'english_uk',
    'dutch', 'german',
    'spanish', 'french',
    'polish', 'russian'
]

df_lang = pd.read_csv('data/languages.csv', index_col=0)
palette = dict(df_lang['color1'])

embs = ["emb0","emb1","emb2","emb3","emb4"]