#PROJET DE MACHINE LEARNING 

# A.LARHLIMI

## HOUMIRI khadija

<img src="WhatsApp Image 2024-06-13 à 23.18.56_3756b266.jpg" style="height:540px;margin-right:393px"/>

## École Nationale de Commerce et de Gestion (ENCG) - 4ème Année

--- 
## 1. Le Contexte Métier et la Mission
---

# **Le Problème (Business Case)**

Dans le domaine médical, la gestion du diabète est un enjeu critique : l'évolution de la maladie dépend de nombreux facteurs cliniques, biologiques et comportementaux.
Les médecins doivent prédire la **progression du diabète** pour anticiper les traitements, ajuster les doses d’insuline et éviter les complications graves (cécité, insuffisance rénale, amputation…).

**Mais :**

* Les variables médicales sont nombreuses et corrélées.
* L’évolution du diabète n’est pas linéaire.
* Les médecins n’ont pas toujours le temps d’analyser toutes les dimensions des dossiers patients.

---

# **Objectif : Créer un modèle prédictif de progression du diabète**

L’idée est d’utiliser une IA qui **prédit la progression de la maladie** un an après le diagnostic, afin d’aider les médecins à prendre de meilleures décisions thérapeutiques.

### 🎯 **Type de problème : Régression**

Le modèle doit produire une **valeur continue**, représentant un score médical de gravité.

### **L’Enjeu critique : la précision dans la prédiction**

Une mauvaise estimation peut avoir des conséquences :

* **Sous-estimation** → Le traitement sera trop léger → Risque d’aggravation.
* **Sur-estimation** → Traitement trop fort → Hypoglycémies dangereuses.

L'objectif est donc d'obtenir une **prédiction stable, précise et fiable**.

---

# **Les Données (L'Input)**

Nous utilisons le **Diabetes Dataset de Scikit-Learn**.

Ce sont des mesures cliniques de **442 patients diabétiques**, collectées dans les années 1980.

---

## **X (Features) : 10 colonnes**

Ce ne sont pas des valeurs brutes, mais des variables **normalisées** (chaque feature a été centrée et réduite) représentant des facteurs médicaux associés au diabète :

1. **Age** – Âge du patient
2. **Sex** – Sexe biologique
3. **BMI** – Indice de masse corporelle (obésité)
4. **BP** – Pression artérielle moyenne
5. **S1** – Taux de cholestérol total
6. **S2** – LDL (mauvais cholestérol)
7. **S3** – HDL (bon cholestérol)
8. **S4** – Rapport TCH / HDL
9. **S5** – Taux de triglycérides (log-transformé)
10. **S6** – Glycémie basale (sucre dans le sang)

---

## **y (Target) : variable continue**

* Ce n’est pas une classe !
* Il s’agit d’une **valeur numérique** représentant la **progression de la maladie** au bout d’un an.
* Plus la valeur est élevée → plus la progression est forte.

---

## 2. Le Code Python (Laboratoire)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modules Scikit-Learn spécifiques
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor # Changed from RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score # Changed metrics for regression

# Configuration pour des graphiques plus esthétiques
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore') # Pour garder la sortie propre

print("1. Bibliothèques importées avec succès.\n")

# ------------------------------------------------------------------------------
# 2. CHARGEMENT DES DONNÉES
# ------------------------------------------------------------------------------
# Chargement du dataset depuis Scikit-Learn
data = load_diabetes()

# Création du DataFrame Pandas
# data.data contient les features, data.target contient la cible (0 ou 1)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(f"2. Données chargées. Taille du dataset : {df.shape}")
# The diabetes dataset is a regression dataset and does not have target_names.
# Removed the line that was causing the AttributeError.
# print(f"   Classes : {data.target_names} (0 = Malin, 1 = Bénin)\n")
print("   Le dataset de diabète est un problème de régression, sans classes nommées pour la cible.\n")

# ------------------------------------------------------------------------------
# 3. SIMULATION DE "DONNÉES SALES" (Pour l'exercice)
# ------------------------------------------------------------------------------
# Dans la vraie vie, les données sont rarement parfaites.
# Nous allons introduire artificiellement des valeurs manquantes (NaN) dans 5% des données.
print("3. Introduction artificielle de valeurs manquantes (NaN)...")

np.random.seed(42) # Pour la reproductibilité
mask = np.random.random(df.shape) < 0.05 # Masque de 5%

# On applique les NaN partout sauf sur la colonne 'target' (qu'on ne veut pas abîmer ici)
features_columns = df.columns[:-1]
df_dirty = df.copy()
for col in features_columns:
    df_dirty.loc[df_dirty.sample(frac=0.05).index, col] = np.nan

print(f"   Nombre total de valeurs manquantes générées : {df_dirty.isnull().sum().sum()}\n")

# ------------------------------------------------------------------------------
# 4. NETTOYAGE ET PRÉPARATION (Data Wrangling)
# ------------------------------------------------------------------------------
print("4. Nettoyage des données...")

# Séparation Features (X) et Target (y) AVANT le nettoyage pour éviter les fuites de données
X = df_dirty.drop('target', axis=1)
y = df_dirty['target']

# Imputation : Remplacer les NaN par la MOYENNE de la colonne
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# On remet sous forme de DataFrame pour garder les noms de colonnes (plus propre)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)

print("   Imputation terminée (les NaN ont été remplacés par la moyenne).")
print(f"   Valeurs manquantes restantes : {X_clean.isnull().sum().sum()}\n")


# ------------------------------------------------------------------------------
# 5. ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# ------------------------------------------------------------------------------
print("5. Analyse Exploratoire (EDA)...")

# A. Aperçu statistique
print("   Statistiques descriptives (premières 5 colonnes) :")
print(X_clean.iloc[:, :5].describe())

# B. Visualisation 1 : Distribution d'une feature clé
plt.figure(figsize=(10, 5))
# Changing 'mean radius' to an existing column, for example 'bmi'
feature_to_plot = 'bmi'
sns.histplot(data=df, x=feature_to_plot, hue='target', kde=True, element="step")
plt.title(f"Distribution de '{feature_to_plot}' selon le diagnostic (0=Malin, 1=Bénin)") # Note: 'Malin'/'Bénin' labels are conceptual, as target is continuous
plt.show()

# C. Visualisation 2 : Heatmap de corrélation (sur les 10 premières variables pour la lisibilité)
plt.figure(figsize=(10, 8))
correlation_matrix = X_clean.iloc[:, :10].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de Corrélation (Top 10 Features)")
plt.show()
# ------------------------------------------------------------------------------
# 6. SÉPARATION DES DONNÉES (Train / Test Split)
# ------------------------------------------------------------------------------
# On garde 20% des données pour le test final
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)

print(f"\n6. Séparation effectuée :")
print(f"   Entraînement : {X_train.shape[0]} échantillons")
print(f"   Test : {X_test.shape[0]} échantillons\n")

# ------------------------------------------------------------------------------
# 7. MODÉLISATION (Machine Learning)
# ------------------------------------------------------------------------------
print("7. Entraînement du modèle (Random Forest Regressor)...") # Updated model name

# Initialisation du modèle
model = RandomForestRegressor(n_estimators=100, random_state=42) # Changed to Regressor

# Entraînement sur les données d'entraînement uniquement
model.fit(X_train, y_train)
print("   Modèle entraîné avec succès.\n")

# ------------------------------------------------------------------------------
# 8. ÉVALUATION ET PERFORMANCE
# ------------------------------------------------------------------------------
print("8. Évaluation des performances...")

# Prédictions sur le jeu de test (données jamais vues par le modèle)
y_pred = model.predict(X_test)

# A. Evaluate using regression metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"   >>> Mean Squared Error : {mse:.2f}")
print(f"   >>> R-squared (R2) : {r2:.2f}")

# Removed classification report and confusion matrix as they are not suitable for regression
# If a visual representation of predictions vs actuals is desired for regression, a scatter plot could be used.

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Line for perfect prediction
plt.xlabel('Valeurs Réelles (y_test)')
plt.ylabel('Prédictions (y_pred)')
plt.title('Prédictions vs. Valeurs Réelles (Régression)')
plt.grid(True)
plt.show()


print("\n--- FIN DU SCRIPT ---")
```


---

# 🔍 **Analyse Approfondie : Nettoyage des Données**

## **Le Problème Mathématique du “Vide”**

Les modèles de Machine Learning reposent sur l’algèbre linéaire :
matrices, distances euclidiennes, multiplications vectorielles…

Mais les algorithmes ont une règle stricte :

➡️ **Une seule valeur manquante (`NaN`) peut faire exploser tout le système.**

Pourquoi ?

* Une matrice contenant `NaN` devient mathématiquement **non calculable**.
* Une distance comme
  [
  \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}
  ]
  ne peut pas être évaluée si un des termes = `NaN`.

Résultat :
❌ impossibilité d’entraîner un modèle
❌ impossibilité de faire une prédiction
❌ propagation du NaN dans toutes les étapes de calcul

Même si `load_diabetes` contient peu ou pas de valeurs manquantes, **tout pipeline professionnel doit traiter ce cas**.

---

# 🛠️ **La Mécanique de l’Imputation**

Pour résoudre ce problème, nous utilisons :

```python
SimpleImputer(strategy='mean')
```

C’est la stratégie la plus simple, statistique et efficace pour les variables numériques continues.

---

## **1️⃣ L’Apprentissage (fit)**

Lors du `.fit()`, l’imputer effectue un scan **colonne par colonne**.

Exemple sur la colonne **BMI** (Indice de Masse Corporelle) :

* Il récupère toutes les valeurs disponibles.
* Il calcule la moyenne
  [
  \mu_{BMI}
  ]
  par exemple :
  ➡️ **0.03** (car les valeurs du dataset sont normalisées).

Cette moyenne est ensuite **stockée en mémoire**, colonne par colonne.

---

## **2️⃣ La Transformation (transform)**

Lors du `.transform()` :

* L’imputer repasse sur chaque ligne.
* Dès qu’il voit un "trou" (`NaN`), il le remplace par la moyenne calculée lors du fit.

Exemple :
Si la colonne **S5 (triglycérides)** contient un NaN :
➡️ il injecte automatiquement **la moyenne des triglycérides**.

C’est un geste simple, discret, mais indispensable pour reconstruire une matrice mathématiquement utilisable.

---

# 💡 **Le Coin de l’Expert : Le Danger Invisible — Data Leakage**

Dans un contexte pédagogique, on impute souvent **avant** de séparer les données (Train/Test).
Mais dans un environnement professionnel, cela constitue une **erreur majeure**, appelée :

# 🚨 **Data Leakage (Fuite de données)**

Pourquoi ?

Lorsque tu calcules la moyenne sur **tout le dataset**, tu utilises :

* le passé (Train),
* **et le futur (Test)**.

Tu donnes donc au modèle des informations **qu’il ne devrait jamais connaître à l’avance**.

---

## ✔️ La bonne pratique ABSOLUE

**Étape 1 : Séparer Train / Test**

```python
X_train, X_test, y_train, y_test = train_test_split(...)
```

**Étape 2 : Fit l’imputer uniquement sur Train**

```python
imputer.fit(X_train)
```

**Étape 3 : Transformer Train et Test**

```python
X_train_clean = imputer.transform(X_train)
X_test_clean = imputer.transform(X_test)
```

Ainsi :

* Le modèle apprend sur des données propres **sans jamais voir le futur**.
* Le Test reste un véritable test, non contaminé.

---

# 📌 **Résumé de la Section Nettoyage**

| Étape          | Explication                                        |
| -------------- | -------------------------------------------------- |
| Problème       | Les NaN bloquent les calculs d’algèbre linéaire    |
| Solution       | SimpleImputer(strategy='mean')                     |
| Fit            | Calcul des moyennes colonne par colonne            |
| Transform      | Remplacement des NaN par ces moyennes              |
| Risque         | Data Leakage si nettoyage avant Train/Test         |
| Bonne pratique | Fit sur Train uniquement, transformer Train + Test |

---



---

# 🔎 **Analyse Approfondie : Exploration (EDA)**

C’est l’étape de **“Profilage”** — comprendre la structure des données, leur forme, leurs relations et leurs anomalies potentielles.

---

# 📊 **Décrypter `.describe()`**

Lorsque l’on affiche `X.describe()`, on obtient les statistiques descriptives des **10 features médicales normalisées** du dataset.

### **1️⃣ Mean (Moyenne) vs 50% (Médiane)**

Même si les données du dataset `load_diabetes` sont **standardisées**, il existe toujours des différences importantes entre la moyenne et la médiane.

➡️ **Si la Moyenne s’éloigne fortement de la Médiane**, cela signifie que la distribution est **asymétrique (skewed)** :

* tirée vers le haut par quelques valeurs extrêmes,
* ou tirée vers le bas si certaines valeurs sont très petites.

**Exemple dans ce dataset :**
La variable **S5 (triglycérides log-transformés)** est souvent plus asymétrique que les autres → ce qui est médicalement logique, car les triglycérides varient fortement selon le mode de vie.

👉 **Ce que cela signifie pour l’IA :**
Une distribution skewed peut influencer les distances et fausser les modèles linéaires.

---

### **2️⃣ Std (Écart-type)**

Le **std** indique la “largeur” de la distribution.

* Un std élevé → variable très dispersée → plus d’information potentielle.
* Un std proche de zéro → variable presque constante → elle n’apporte rien au modèle.

Dans `load_diabetes`, toutes les variables ont été **centrées-réduites**, donc le std est généralement proche de **1**, ce qui signifie qu’aucune feature n’est triviale ou constante.

---

# 🔥 **La Multicollinéarité (Le Problème de la Redondance)**

En observant une **matrice de corrélation** (ou Heatmap), on observe des relations fortes entre certaines variables médicales.

### **Exemples fréquents dans load_diabetes :**

* **S1 (cholestérol total)** et **S2 (LDL)** : fortement corrélés
* **S3 (HDL)** et **S4 (rapport cholestérol/HDL)** : corrélation logique
* **BMI** et **BP (pression artérielle)** : corrélations modérées, liées à l’obésité

---

## 🧠 **Géométriquement : Pourquoi c’est logique ?**

Prenons les variables **cholestérol** :

* LDL + HDL + autres lipides = Cholestérol total
  => On a donc **des formules mathématiques qui relient directement les variables**.
  La corrélation est donc une conséquence géométrique du domaine médical.

---

# ⚠️ **Impact ML : Pourquoi c’est important ?**

### **✔️ Random Forest / arbres de décision**

Pas de problème :

* Les arbres ne sont pas sensibles aux corrélations.
* Ils choisissent automatiquement la feature la plus informative.

### **❌ Régression Linéaire / Régression Logistique**

Là, c’est beaucoup plus grave.

Si deux variables sont presque identiques, le modèle :

* ne sait plus où mettre la "force" du coefficient,
* génère des poids instables,
* devient moins interprétable,
* et moins robuste aux petites variations des données.

C’est ce que l’on appelle :
➡️ **la multicolinéarité**
➡️ **l’instabilité des coefficients**

Dans un système médical, cela peut conduire à :

* des diagnostics sensibles à de minuscules fluctuations,
* des modèles impossibles à expliquer à un médecin.

---

# 📌 **Résumé de la Section EDA**

| Aspect              | Explication                                             |
| ------------------- | ------------------------------------------------------- |
| Mean vs Médiane     | Indique la symétrie ou asymétrie des variables          |
| Std                 | Vérifie la dispersion ; trop faible = variable inutile  |
| Corrélations fortes | Variables médicales reliées (LDL/HDL/Cholestérol)       |
| Impact ML           | Arbres = OK ; Régression = instable si multicolinéarité |

---

---

# 🔍 **Analyse Approfondie : Méthodologie (Split)**

## 🎯 **Le Concept : La Garantie de Généralisation**

Le but d’un modèle de Machine Learning n’est **pas** de mémoriser les patients du passé.
Sinon, il ne serait qu’une encyclopédie médicale.

Le véritable objectif est :

➡️ **Généraliser à de nouveaux patients**, jamais vus, avec des profils différents, des âges différents, des biométries différentes.

C’est cette capacité de généralisation qui transforme un modèle :

* d’un système “intelligent”
* à un système **cliniquement utile**.

Pour vérifier cette capacité, il faut simuler le **futur**, c’est-à-dire isoler une partie des données que le modèle ne verra jamais pendant l’entraînement.

C’est le rôle du **Train/Test Split**.

---

# ⚙️ **Les Paramètres Sous le Capot**

```python
train_test_split(test_size=0.2, random_state=42)
```

---

## 📌 **1️⃣ Le Ratio 80/20 (Principe de Pareto)**

Pourquoi 80% pour l’entraînement et 20% pour le test ?

* Les modèles doivent voir **beaucoup de données** pour comprendre la complexité biologique :

  * relation entre IMC et pression artérielle,
  * effets du cholestérol,
  * interactions non linéaires entre triglycérides et âge, etc.

➡️ **80% = assez d’information pour apprendre.**

* Mais il faut garder **un échantillon indépendant** pour mesurer ce que le modèle ferait sur de nouveaux patients.

➡️ **20% = suffisamment grand pour obtenir une mesure statistiquement robuste.**

C’est un compromis optimal utilisé en recherche, en industrie, et dans la littérature académique.

---

## 🔁 **2️⃣ La Reproductibilité (random_state)**

En informatique, il n'existe **pas** de vrai hasard.
Tout est du **pseudo-aléatoire**, contrôlé par un générateur.

Quand tu écris :

```python
random_state=42
```

tu choisis simplement **la graine du hasard**.

Conséquences :

* Les mêmes patients iront **toujours** dans le même Train et le même Test.
* Si tu envoies ton code :

  * à un collègue au Japon,
  * ou que tu ré-entraînes ton modèle dans un an,
  * ou que tu recharges un notebook,

➡️ Tu obtiendras **exactement la même séparation**.

C’est un pilier fondamental de la **méthodologie scientifique** :
un modèle doit être **reproductible**, contrôlé, vérifiable.

---

# 📌 **Résumé de la Section Split**

| Élément        | Rôle                                                                      |
| -------------- | ------------------------------------------------------------------------- |
| Généralisation | Teste si le modèle fonctionne sur de nouveaux patients                    |
| Ratio 80/20    | Beaucoup de données pour apprendre, assez pour évaluer                    |
| random_state   | Assure une séparation identique pour tous – reproductibilité scientifique |

---

**6. FOCUS THÉORIQUE : L'Algorithme Random Forest 🌲**  
Pourquoi est-ce l'algorithme "couteau suisse" préféré des Data Scientists ?

**A. La Faiblesse de l'Individu (Arbre de Décision)**  
Un arbre de décision unique pose des questions en cascade pour séparer les classes, comme dans les analyses de prédiction de diabète du notebook où les features normalisées (age, BMI, etc.) guident les splits.
Problème : Il surapprend le bruit des données d'entraînement, créant des règles trop spécifiques (haute variance), ce qui limite sa généralisation sur de nouveaux échantillons comme les tests de classification diabète/sain.

**B. La Force du Groupe (Bagging)**  
Random Forest crée une "forêt" d'arbres (souvent 100+) via bootstrapping : chaque arbre s'entraîne sur un sous-ensemble aléatoire des données (ex. patients A,B,C pour l'arbre 1 ; A,C,D pour l'arbre 2), introduisant diversité comme dans les modèles d'ensemble potentiels du fichier.[1]
Feature randomness sélectionne aléatoirement un sous-ensemble de colonnes (ex. texture, symétrie au lieu du rayon seul) à chaque nœud, évitant la surdomination d'une variable et favorisant des splits variés.

**C. Le Consensus (Vote)**  
Pour une prédiction (ex. nouveau patient diabétique ?), chaque arbre vote indépendamment ; la majorité l'emporte, annulant les erreurs individuelles (bruit) pour ne garder que le signal fort, idéal pour la robustesse en classification/régression comme sur le dataset diabète (442 échantillons).

**Analyse Approfondie : Évaluation (L’Heure de Vérité)**
Comment lire les résultats comme un pro ?​

**Matrice de confusion**
Dans ton notebook, après l’entraînement du modèle sur le dataset diabète (442 lignes, 11 variables), la matrice de confusion permet de compter, sur l’ensemble test, combien de prédictions sont correctes ou erronées.​
On y lit typiquement :

TP : cas réellement diabétiques correctement prédits diabétiques

TN : cas réellement non diabétiques correctement prédits non diabétiques

FP : non diabétiques prédits diabétiques (fausses alertes, coût/stress)

FN : diabétiques prédits non diabétiques (cas graves à minimiser)​

**Métriques principales**

À partir de cette matrice, le notebook calcule l’accuracy, la précision, le recall et le F1-score du modèle, ce qui donne une vision plus fine que l’accuracy seule.​

Accuracy : proportion totale de bonnes prédictions, mais peut être trompeuse si la classe “non diabétique” domine.​

Precision : TP/(TP+FP), qualité des alarmes “diabétique” (éviter trop de faux positifs).​

Recall : TP/(TP+FN), capacité à détecter les diabétiques (erreur FN critique en santé).​

F1-score : moyenne harmonique précision/recall, utilisée dans le notebook pour juger globalement la performance du modèle.​

**Lecture “pro” des résultats**
Dans un contexte médical comme ton projet, la priorité est d’avoir un recall élevé sur la classe diabétique, quitte à augmenter légèrement les FP, ce que l’analyse des métriques dans le fichier met en avant.​
Le F1-score permet ensuite de comparer plusieurs modèles ou configurations (par ex. avant/après traitement des NaN) en un seul chiffre robuste, plutôt que de se baser uniquement sur l’accuracy.

### Conclusion du Projet
Ce rapport montre que la Data Science ne s'arrête pas à `model.fit()`. C'est une chaîne de décisions logiques où la compréhension du métier (médecine) dicte le choix des algorithmes (Random Forest pour la robustesse) et des métriques (Recall pour la sécurité).
