from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd

X_test = pd.read_parquet('data/X_test.parquet')
y_test = pd.read_parquet('data/y_test.parquet')

# 1. Charger le modèle
model = lgb.Booster(model_file='src/finfrog_scoring_model.txt')

# 1. Prédiction sur le set de test
y_pred_proba = model.predict(X_test)
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"L'AUC du modèle est de : {auc_score:.4f}")

# 2. Matrice de confusion (Seuil par défaut 0.5)
y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred_proba]
cm = confusion_matrix(y_test, y_pred_binary)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédiction (0=Rembourse, 1=Défaut)')
plt.ylabel('Réalité')
plt.title('Matrice de Confusion - Scoring Finfrog')
plt.show()


def get_credit_score(data):
    """
    Prend un dictionnaire de données client et retourne un score de 0 à 100.
    """
    # Transformer les données en format accepté par LightGBM (2D array)
    features = np.array([data['loan_amnt'], 
                         data['annual_inc'], data['dti'], data['open_acc']]).reshape(1, -1)
    
    # Prédire la probabilité de défaut (PD)
    prob_defaut = model.predict(features)[0]
    
    # Transformer la PD en un Score de confiance (0 à 100)
    # Plus le score est haut, plus le client est fiable
    credit_score = (1 - prob_defaut) * 100
    
    return round(credit_score, 2)

# --- TEST SUR UN NOUVEAU CLIENT ---
nouveau_client = {
    'loan_amnt': 5000,
    'int_rate': 1.5,
    'annual_inc': 4500000,
    'dti': 18.5,
    'open_acc' : 5
}

score = get_credit_score(nouveau_client)
print(f"Le score de crédit du client est de : {score}/100")

if score > 80:
    print("Verdict : Prêt accordé automatiquement.")
elif score > 50:
    print("Verdict : Analyse manuelle requise.")
else:
    print("Verdict : Prêt refusé.")
    
# Test de sensibilité
client_pauvre = [5000, 20000, 30.0, 5]   # Petit revenu, gros endettement
client_riche  = [5000, 200000, 5.0, 5]   # Gros revenu, faible endettement

prob_pauvre = model.predict(np.array(client_pauvre).reshape(1, -1))[0]
prob_riche = model.predict(np.array(client_riche).reshape(1, -1))[0]

print(f"Probabilité Défaut (Pauvre): {prob_pauvre:.4f}")
print(f"Probabilité Défaut (Riche) : {prob_riche:.4f}")