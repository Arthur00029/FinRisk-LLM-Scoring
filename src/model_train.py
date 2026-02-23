import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sentence_transformers import SentenceTransformer
import os

os.environ['USE_TF'] = 'NO'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Désactive les logs d'erreurs TF

nlp_model = SentenceTransformer('all-MiniLM-L6-v2')

class FinfrogTrainer:
    def __init__(self, data_path):
        features = ['id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
                    'installment', 'grade', 'sub_grade', 'emp_title', 'emp_length', 'home_ownership',
                    'annual_inc', 'verification_status','loan_status', 'dti', 'open_acc']
        self.df = pd.read_csv(data_path, usecols=features)
        self.model = None

    def preprocess(self):
        
        features = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term',
            'installment', 'emp_title', 'emp_length', 'home_ownership',
            'annual_inc', 'verification_status', 'dti', 'open_acc']
        ######################
            # On nettoie et on récupère uniquement les titres uniques pour gagner du temps
        unique_titles = self.df['emp_title'].astype(str).unique().tolist()
        
        # 2. Transformer le texte en vecteurs (coordonnées numériques)
        embeddings = nlp_model.encode(unique_titles, show_progress_bar=True)
        
        # 3. Créer un dictionnaire de correspondance {Titre: Vecteur}
        job_map = {title: emb for title, emb in zip(unique_titles, embeddings)}
        
        # 4. Appliquer au DataFrame (on crée une liste de vecteurs)
        # On réduit souvent la dimensionnalité ici pour ne pas avoir 384 colonnes
        # Mais pour commencer, on peut juste prendre la moyenne du vecteur ou 
        # les 5 premières composantes principales.
        
        # Exemple simple : on crée une feature "Job_Score" (moyenne du vecteur)
        # ou on concatène les 3 premières dimensions
        df_embeddings = self.df['emp_title'].map(job_map)
        
        # On transforme la liste de vecteurs en plusieurs colonnes
        matrix = np.array(df_embeddings.tolist())
        for i in range(5): # On ne prend que les 5 premières dimensions pour rester léger
            self.df[f'job_vector_{i}'] = matrix[:, i]
        
        ##########################
        
        features = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term',
            'installment', 'emp_length', 'home_ownership',
            'annual_inc', 'verification_status', 'dti', 'open_acc',
            'job_vector_0','job_vector_1','job_vector_2','job_vector_3','job_vector_4',]
    
        self.df = self.df[self.df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
        self.df = self.df[self.df['term'].isin([' 36 months', ' 60 months'])]
        self.df['term'] = self.df['term'].apply(lambda x: 36 if x == ' 36 months' else 60)
        self.df['target'] = self.df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
        # 1. Gestion de la Target (Toujours nécessaire en premier)
        if 'loan_status' in self.df.columns:
            self.df = self.df[self.df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
            self.df['target'] = self.df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})
            self.df = self.df.drop(columns=['loan_status'])

        # 2. Traitement spécifique des colonnes connues pour être pénibles
        if 'emp_length' in self.df.columns:
            self.df['emp_length'] = self.df['emp_length'].astype(str).str.replace(r'\+? years?', '', regex=True)
            self.df['emp_length'] = self.df['emp_length'].str.replace('< 1', '0', regex=False)
            self.df['emp_length'] = pd.to_numeric(self.df['emp_length'], errors='coerce').fillna(-1)

        if 'verification_status' in self.df.columns:
            verif_map = {'Not Verified': 0, 'Source Verified': 1, 'Verified': 2}
            self.df['verification_status'] = self.df['verification_status'].map(verif_map).fillna(-1)

        # 3. BOUCLE POUR LES STR (Grades, Verification, etc.)
        # On cible tout ce qui n'est pas un chiffre
        text_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        for col in text_cols:
            if col == 'emp_title':
                self.df[col] = self.df[col].astype(str).str.lower().fillna('unknown')
            else:
                # On force le type 'category' pour que LightGBM accepte 'A', 'B', 'C'...
                self.df[col] = self.df[col].astype('category')
                # ASTUCE : On récupère les codes numériques des catégories pour être sûr
                # Cela transforme 'A' en 0, 'B' en 1, etc. tout en gardant l'info
                self.df[col] = self.df[col].cat.codes 

        # 4. Remplissage des NaNs numériques
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median())

                
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

