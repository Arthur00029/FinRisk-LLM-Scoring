import pandas as pd
import numpy as np

class RiskEngine:
    """
    Calculateur d'indicateurs de risque de crédit (Bâle III standards).
    """
    def __init__(self):
        pass

    def calculate_expected_loss(self, pd, ead, lgd):
        """
        Calcule la Perte Attendue (Expected Loss).
        Formule : EL = PD * EAD * LGD
        """
        return pd * ead * lgd

    def generate_loan_tape(self, n_loans=100):
        """
        Simule un 'Loan Tape' (portefeuille de prêts) pour le reporting.
        """
        np.random.seed(42)
        data = {
            'loan_id': [f'FR-{1000+i}' for i in range(n_loans)],
            'exposure_at_default': np.random.uniform(100, 1000, n_loans), # EAD : Montant prêté
            'probability_of_default': np.random.beta(2, 5, n_loans),      # PD : Risque (0 à 1)
            'loss_given_default': 0.85,                                  # LGD : Fixe à 85% pour du micro-crédit
        }
        
        df = pd.DataFrame(data)
        
        # Calcul de la perte attendue pour chaque ligne
        df['expected_loss'] = df.apply(
            lambda x: self.calculate_expected_loss(
                x['probability_of_default'], 
                x['exposure_at_default'], 
                x['loss_given_default']
            ), axis=1
        )
        
        return df

if __name__ == "__main__":
    engine = RiskEngine()
    loan_tape = engine.generate_loan_tape(10)
    
    print("--- Extrait du Reporting Prudentiel (Loan Tape) ---")
    print(loan_tape[['loan_id', 'exposure_at_default', 'probability_of_default', 'expected_loss']].head())
    
    total_el = loan_tape['expected_loss'].sum()
    print(f"\nPerte attendue totale sur le portefeuille : {total_el:.2f} €")