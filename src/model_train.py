import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

class FinfrogTrainer:
    def __init__(self, data_path):
        features = ['id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
                    'installment', 'grade', 'sub_grade', 'emp_title', 'emp_length', 'home_ownership',
                    'annual_inc', 'verification_status','loan_status', 'dti', 'open_acc']
        self.df = pd.read_csv('data/accepted_2007_to_2018Q4.csv', usecols=features)
        self.model = None

    def preprocess(self):
        # On simplifie : On prédit si 'loan_status' est 'Fully Paid' ou 'Charged Off' (Défaut)
        # On modifie les variables non numérique  
        self.df = self.df[self.df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
        self.df['home_ownership'] = self.df['home_ownership'].apply(lambda x: 2 if x == 'OWN' else 1 if x == "MORTGAGE" else 0)
        self.df['target'] = self.df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
        self.df['emp_length'] = self.df['emp_length'].apply(lambda x: 10. if x == '10+ years' else 0. if x == '< 1 year' or type(x) == type(1.) else float(x.split()[0]))
        self.df['verification_status'] = self.df['verification_status'].apply(lambda x: 0 if x == 'Not Verified' else 1)
        

        features = ['loan_amnt', 'annual_inc', 'dti', 'home_ownership', 'open_acc','emp_length','verification_status']
        #features = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti', 'open_acc']
        # Note : Il faudrait encoder 'term' en numérique ici
        
        self.X = self.df[features]
        self.y = self.df['target']
        
        return train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train(self, X_train, y_train):
        # Hyperparamètres de base pour le risque de crédit
        # Exemple de configuration "Expert"
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.02,       # On baisse un peu pour être plus stable
            'num_leaves': 127,            # Plus de feuilles pour les nouvelles features
            'max_depth': 10,             # On limite la profondeur pour ne pas overfitter
            'min_data_in_leaf': 100,     # Plus robuste sur 1M de lignes
            'feature_fraction': 0.8,     # Chaque arbre ne regarde que 80% des colonnes
            'cat_smooth': 10,            # Gestion pro des catégories
            'is_unbalance': True,        # Toujours utile pour le déséquilibre des défauts
            'verbosity': -1
        }

        # Entraînement avec arrêt automatique
        
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test)
        self.model = lgb.train(params, 
                               train_data, 
                               valid_sets=[test_data],
                               num_boost_round=10000,
                               callbacks=[lgb.early_stopping(stopping_rounds=100)]
                                )                   
        print("Entraînement terminé !")
        # Sauvegarde au format texte (très léger et rapide)
        self.model.save_model('src/scoring_model.txt')
        print("Modèle sauvegardé avec succès !")

# --- Usage ---
trainer = FinfrogTrainer('data/accepted_2007_to_2018Q4.csv')
X_train, X_test, y_train, y_test = trainer.preprocess()

# On convertit les Series y en DataFrame pour le stockage Parquet
pd.DataFrame(X_train).to_parquet('data/X_train.parquet')
pd.DataFrame(X_test).to_parquet('data/X_test.parquet')
pd.DataFrame(y_train).to_parquet('data/y_train.parquet')
pd.DataFrame(y_test).to_parquet('data/y_test.parquet')

print("Fichiers de Split sauvegardés dans data/ !")
trainer.train(X_train, y_train)

