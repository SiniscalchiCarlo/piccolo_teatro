import pandas as pd

# Creazione di un esempio di DataFrame
data = {
    'Paese': ['Italia', 'Italia', 'Italia', 'Francia', 'Francia', 'Spagna', 'Spagna', 'Spagna'],
    'Città': ['Roma', 'Milano', 'Torino', 'Parigi', 'Lione', 'Madrid', 'Barcellona', 'Valencia'],
    'Popolazione': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
}

df = pd.DataFrame(data)

# Visualizzazione del DataFrame originale
print("DataFrame Originale:")
print(df)

# Utilizzo di groupby() senza aggregazione
# Iterazione sui gruppi
print("\nIterazione sui Gruppi:")
for paese, group in df.groupby('Paese'):
    print(f"Paese: {paese}")
    print(group)
    # Esempio di operazione personalizzata: ordinamento per popolazione
    sorted_group = group.sort_values('Popolazione')
    print("Gruppo Ordinato per Popolazione:")
    print(sorted_group)
    print("\n")

# Utilizzo di apply() per eseguire una funzione personalizzata su ogni gruppo
def custom_function(group):
    # Esempio di funzione che ritorna il gruppo con una nuova colonna
    group['Doppia_Popolazione'] = group['Popolazione'] * 2
    return group

# Applicazione della funzione personalizzata
result = df.groupby('Paese').apply(custom_function)

# Visualizzazione del risultato
print("Risultato con Nuova Colonna:")
print(result)
