import pandas as pd
from transformers import pipeline

class TransactionAnalyzer:
    """
    Analyseur de libellés bancaires utilisant un LLM pour le scoring de crédit.
    Objectif : Extraire des signaux de risque à partir de texte non structuré.
    """
    def __init__(self, model_name="facebook/bart-large-mnli"):
        # On utilise un modèle de classification Zero-Shot (pas besoin de réentraîner)
        print(f"Chargement du modèle {model_name}...")
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        

        self.candidate_labels = [
            "jeu d'argent et paris", 
            "frais d'incidents bancaires", 
            "revenu stable", 
            "remboursement de crédit",
            "dépense de consommation courante"
        ]

    def analyze_batch(self, descriptions):
        """
        Analyse une liste de libellés et retourne la catégorie la plus probable.
        """
        results = self.classifier(descriptions, self.candidate_labels)
        
        # On extrait le label avec le score de confiance le plus élevé
        structured_data = []
        for res in results:
            structured_data.append({
                "description": res['sequence'],
                "predicted_category": res['labels'][0],
                "confidence_score": round(res['scores'][0], 3)
            })
        
        return pd.DataFrame(structured_data)

# --- EXEMPLE D'UTILISATION ---
if __name__ == "__main__":
    # Simuler des libellés typiques qu'un algo Regex classique pourrait rater
    sample_tx = [
        "VIR SEPA RECU / DE LA PART DE : EMPLOYEUR SAS - SALAIRE JANVIER",
        "PRELEVEMENT WEB - WINAMAX PARIS SPORTIFS",
        "COMMISSION INTERVENTION - SOLDE INSUFFISANT",
        "TRANSFERT LYDIA - REMBOURSEMENT SOIREE POKER"
    ]
    
    analyzer = TransactionAnalyzer()
    df_insights = analyzer.analyze_batch(sample_tx)
    
    print("\n--- Insights extraits pour le Scoring ---")
    print(df_insights)