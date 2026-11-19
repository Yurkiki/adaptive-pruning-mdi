import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
co = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Importance des variables : pourquoi la mesure standard est fausse, et comment la réparer

**Ce notebook ne suppose aucune connaissance préalable.** On part de zéro : ce qu'est un arbre de
décision, ce qu'il mesure, pourquoi sa mesure d'importance est biaisée, et les deux remèdes que ce
projet combine.

Chaque idée est montrée, pas seulement décrite : tout le code tourne, toutes les figures sont
produites par les cellules.

> **Pour exécuter** : lancez depuis la racine du dépôt (`jupyter lab notebooks/cours.ipynb`).
> Dépendances : `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`.

---

## Plan

| | |
|---|---|
| 1 | La question qu'on se pose |
| 2 | Un arbre de décision, construit pas à pas |
| 3 | L'impureté : la seule chose que l'arbre optimise |
| 4 | Le MDI, mesure d'importance standard |
| 5 | **Le bug** : de l'importance venue de nulle part |
| 6 | Pourquoi : une identité en trois lignes |
| 7 | Le second biais : juger sur sa propre copie |
| 8 | Remède 1 : mesurer sur des données non vues |
| 9 | Remède 2 : élaguer l'arbre |
| 10 | Comment on sait qui a raison (et les scénarios disponibles) |
| 11 | Le plan d'expérience |
| 12 | Ce qu'on a trouvé |
""")

co(r"""
import sys, warnings
from pathlib import Path

# marche qu'on lance depuis la racine du depot ou depuis notebooks/
racine = Path.cwd()
while not (racine / "src" / "simulate.py").exists() and racine != racine.parent:
    racine = racine.parent
sys.path.insert(0, str(racine / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# palette unique pour tout le notebook
INK, SOFT, GRID = "#16191f", "#4e545f", "#dfe2e9"
BLEU, ORANGE, VERT, PRUNE = "#2a78d6", "#eb6834", "#1baf7a", "#3a4a86"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": SOFT, "text.color": INK,
    "xtick.color": SOFT, "ytick.color": SOFT, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 110,
})
print("prêt")
""")

# ---------------------------------------------------------------- 1
md(r"""
---
## 1. La question qu'on se pose

Vous avez un tableau. Chaque ligne est une observation, chaque colonne une **variable**, et une
colonne à part — notée $y$ — est ce qu'on cherche à **prédire**.

Exemple : des quartiers de Californie. On connaît le revenu médian, l'âge des maisons, le nombre de
pièces… et on veut prédire le **prix médian** du quartier.

| quartier | revenu | âge | pièces | → | **prix** |
|---|---|---|---|---|---|
| A | 1,5 | 40 | 4,2 | | **80 k\$** |
| B | 6,0 | 20 | 7,1 | | **350 k\$** |

Une fois qu'un modèle prédit correctement, une seconde question arrive, souvent plus intéressante
que la prédiction elle-même :

> **Quelles variables comptent vraiment ?**

C'est l'*importance des variables*. Elle sert à comprendre un phénomène, à éliminer des mesures
coûteuses et inutiles, ou à vérifier qu'un modèle ne s'appuie pas sur n'importe quoi.

Ce notebook montre que la mesure la plus utilisée pour ça est **fausse**, de deux manières
indépendantes — et ce qu'on peut y faire.
""")

# ---------------------------------------------------------------- 2
md(r"""
---
## 2. Un arbre de décision, construit pas à pas

Un arbre pose des questions binaires sur les variables et découpe les données en groupes.

Fabriquons un jeu minuscule : le prix dépend du revenu, et **rien d'autre**.
""")

co(r"""
revenu = np.array([1.5, 1.8, 2.0, 6.0, 6.5, 7.0])
age    = np.array([40, 35, 30, 20, 25, 15], dtype=float)   # sans rapport avec le prix
prix   = np.array([80, 95, 110, 350, 380, 400], dtype=float)

print(f"{'quartier':>9} {'revenu':>7} {'age':>5} {'prix':>6}")
for q, r_, a_, p_ in zip("ABCDEF", revenu, age, prix):
    print(f"{q:>9} {r_:>7.1f} {a_:>5.0f} {p_:>6.0f}")
print(f"\nmoyenne des prix : {prix.mean():.0f} k$")
""")

md(r"""
L'arbre cherche **une question** qui sépare au mieux les prix. Ici, « revenu ≤ 4 ? » coupe le groupe
en deux moitiés très homogènes.
""")

co(r"""
fig, ax = plt.subplots(figsize=(8.2, 4.2))

def boite(x, y, titre, detail, couleur):
    ax.add_patch(FancyBboxPatch((x - .72, y - .30), 1.44, .60,
                 boxstyle="round,pad=0.04", fc=couleur, ec=GRID, lw=1.2, zorder=3))
    ax.text(x, y + .08, titre, ha="center", va="center", fontsize=9.5, color=INK, zorder=4)
    ax.text(x, y - .12, detail, ha="center", va="center", fontsize=8.5, color=SOFT, zorder=4)

boite(0, 1.0, "les 6 quartiers", "moyenne = 236 k$", "#f3f4f7")
boite(-1.6, 0, "revenu ≤ 4", "A, B, C — moyenne 95 k$", "#eaf2fd")
boite( 1.6, 0, "revenu > 4", "D, E, F — moyenne 377 k$", "#eaf2fd")
for dx in (-1.6, 1.6):
    ax.plot([0, dx], [0.70, 0.30], color=SOFT, lw=1.1, zorder=1)
ax.text(-0.95, .56, "oui", fontsize=8.5, color=SOFT)
ax.text( 0.80, .56, "non", fontsize=8.5, color=SOFT)
ax.set_xlim(-2.7, 2.7); ax.set_ylim(-.55, 1.5); ax.axis("off")
ax.set_title("Un arbre à une seule coupure", fontsize=11, color=INK, pad=10)
plt.tight_layout(); plt.show()
""")

md(r"""
**Le vocabulaire, une fois pour toutes :**

- un **nœud** est un groupe de lignes ; la **racine** contient tout le monde ;
- une **coupure** envoie chaque ligne à gauche ou à droite selon une question ;
- une **feuille** est un nœud qu'on ne coupe plus ;
- pour **prédire**, on fait descendre la nouvelle ligne et on répond **la moyenne de sa feuille**.

Un nouveau quartier avec un revenu de 1,9 part à gauche → prédiction = 95 k\$.

Le point essentiel : **un nœud contient des lignes, pas des variables.**
""")

# ---------------------------------------------------------------- 3
md(r"""
---
## 3. L'impureté : la seule chose que l'arbre optimise

Comment l'arbre a-t-il *su* que « revenu ≤ 4 » était une bonne question ?

Il mesure à quel point les prix sont **mélangés** dans un groupe. Cette mesure s'appelle
l'**impureté**, et en régression c'est simplement la **variance** :

$$H(\text{nœud}) = \text{moyenne des } (y_i - \bar{y}_{\text{nœud}})^2$$

- prix tous proches → impureté **faible** (groupe homogène) ;
- prix dispersés → impureté **élevée** ;
- tous identiques → impureté **nulle**, le nœud est dit *pur*.
""")

co(r"""
def impurete(v):
    return float(np.mean((v - v.mean())**2))

gauche, droite = prix[:3], prix[3:]
H_avant = impurete(prix)
H_apres = (len(gauche)/len(prix))*impurete(gauche) + (len(droite)/len(prix))*impurete(droite)

print(f"impureté AVANT la coupure : {H_avant:8.1f}")
print(f"impureté APRÈS (pondérée) : {H_apres:8.1f}")
print(f"{'':>26} {'-'*8}")
print(f"baisse d'impureté         : {H_avant - H_apres:8.1f}   <- le gain de cette coupure")
""")

co(r"""
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)

axes[0].scatter(np.zeros(6), prix, s=90, color=BLEU, zorder=3, edgecolor="white", lw=1.5)
axes[0].axhline(prix.mean(), color=SOFT, lw=1.2, ls="-")
axes[0].text(.12, prix.mean()+12, f"moyenne {prix.mean():.0f}", fontsize=8.5, color=SOFT)
axes[0].set_title(f"avant — impureté {H_avant:.0f}", fontsize=10)
axes[0].set_xlim(-.5, .9); axes[0].set_xticks([])

for i, (grp, lab) in enumerate([(gauche, "revenu ≤ 4"), (droite, "revenu > 4")]):
    axes[1].scatter(np.full(len(grp), i), grp, s=90, color=VERT, zorder=3,
                    edgecolor="white", lw=1.5)
    axes[1].plot([i-.25, i+.25], [grp.mean()]*2, color=SOFT, lw=1.4)
axes[1].set_title(f"après — impureté {H_apres:.0f}", fontsize=10)
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["revenu ≤ 4", "revenu > 4"], fontsize=9)

for a in axes:
    a.grid(axis="y"); a.set_ylabel("prix (k$)")
fig.suptitle("Une bonne coupure rassemble les points", fontsize=11, color=INK)
plt.tight_layout(); plt.show()
""")

md(r"""
**L'algorithme complet tient en une phrase :** essayer toutes les variables et tous les seuils,
garder la coupure qui fait le plus baisser l'impureté, puis recommencer dans chaque moitié.

Une remarque qui servira beaucoup plus loin : **une coupure ne peut jamais faire *monter*
l'impureté.** Dans le pire des cas elle ne change rien. Donc le gain est toujours ≥ 0 —
**même pour une coupure sur une variable complètement inutile.**
""")

# ---------------------------------------------------------------- 4
md(r"""
---
## 4. Le MDI, mesure d'importance standard

L'idée est naturelle : si une variable fait beaucoup baisser l'impureté, c'est qu'elle est utile.

> **MDI d'une variable** = somme des baisses d'impureté de **toutes les coupures faites sur cette
> variable**, chacune pondérée par le nombre de lignes concernées.

MDI veut dire *Mean Decrease in Impurity*. C'est la mesure que renvoie scikit-learn sous le nom
`feature_importances_`, et de très loin la plus utilisée en pratique.

Vérifions-la sur notre petit exemple, où l'on **sait** que seul le revenu compte.
""")

co(r"""
from sklearn.tree import DecisionTreeRegressor

X = np.column_stack([revenu, age])
arbre = DecisionTreeRegressor(max_depth=1).fit(X, prix)

imp = arbre.tree_.compute_feature_importances(normalize=False)
for nom, v in zip(["revenu", "âge   "], imp):
    print(f"  {nom} : {v:9.1f}")
print(f"\n  variance totale des prix : {impurete(prix):9.1f}")
""")

md(r"""
Parfait : tout le crédit va au revenu, l'âge obtient zéro. La mesure fait exactement ce qu'on
attend.

**Sur cet exemple.** Voyons ce qui se passe quand on laisse l'arbre grandir.
""")

# ---------------------------------------------------------------- 5
md(r"""
---
## 5. Le bug : de l'importance venue de nulle part

Voici l'expérience décisive. On fabrique des données où **rien n'a de rapport avec rien** :

- $y$ est du bruit pur ;
- les 5 variables sont du bruit pur ;
- aucune relation, par construction.

La bonne réponse est donc : **importance nulle pour les cinq variables.**
""")

co(r"""
rng = np.random.default_rng(0)
X_bruit = rng.normal(size=(100, 5))      # 5 variables de bruit pur
y_bruit = rng.normal(size=100)           # y sans aucun rapport

complet = DecisionTreeRegressor(random_state=0).fit(X_bruit, y_bruit)   # arbre complet
imp = complet.tree_.compute_feature_importances(normalize=False)

print("vérité :  0.000  pour les cinq variables\n")
for j, v in enumerate(imp):
    print(f"  variable {j} : {v:.3f}")
print(f"\n  SOMME             : {imp.sum():.3f}")
print(f"  variance de y     : {y_bruit.var():.3f}   <- identique !")
print(f"  feuilles de l'arbre : {complet.get_n_leaves()}  (pour 100 points)")
""")

co(r"""
fig, ax = plt.subplots(figsize=(7.4, 3.6))
b = ax.bar(range(5), imp, color=ORANGE, width=.6, zorder=3)
ax.axhline(0, color=INK, lw=1.2, zorder=4)
ax.plot([-.5, 4.5], [0, 0], color=VERT, lw=2.4, zorder=5, label="la vérité : 0")
for r_, v in zip(b, imp):
    ax.text(r_.get_x()+r_.get_width()/2, v+.004, f"{v:.3f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(range(5)); ax.set_xticklabels([f"variable {j}" for j in range(5)])
ax.set_ylabel("importance MDI"); ax.grid(axis="y")
ax.set_ylim(top=imp.max()*1.35)
ax.legend(frameon=False, loc="upper left")
ax.set_title("Cinq variables sans aucun lien avec y — et pourtant…", fontsize=11, color=INK)
plt.tight_layout(); plt.show()
""")

md(r"""
**Chaque variable reçoit une importance franchement positive, alors que la vérité est zéro pour
toutes.** Rien dans ces données ne prédit quoi que ce soit, et la mesure affirme le contraire.

Remarquez aussi le second chiffre : la **somme des importances vaut exactement la variance de
$y$**. Ce n'est pas une coïncidence — c'est l'explication complète du phénomène.
""")

# ---------------------------------------------------------------- 6
md(r"""
---
## 6. Pourquoi : une identité en trois lignes

Additionnons les MDI de **toutes** les variables. Chaque nœud interne apparaît deux fois dans la
somme : une fois en positif (comme parent), une fois en négatif (comme enfant de son propre
parent). Tout ce qui est au milieu **s'annule**, et il ne reste que les extrémités :

$$\sum_j \text{MDI}_j \;=\; \underbrace{H(\text{racine})}_{\text{variance de } y} \;-\; \underbrace{\sum_{\text{feuilles}} \frac{n_\ell}{n} H(\ell)}_{\text{erreur d'entraînement}}$$

Or un arbre qu'on laisse pousser jusqu'au bout met **un seul point par feuille**. Et la variance
d'un point tout seul vaut… **zéro** : une valeur ne s'écarte pas d'elle-même.

L'erreur d'entraînement disparaît, et il reste :

$$\sum_j \text{MDI}_j \;=\; \text{variance de } y \;=\; \underbrace{\text{signal}}_{\text{explicable}} + \underbrace{\text{bruit}}_{\textbf{inexplicable}}$$

**L'arbre distribue donc le bruit entre les variables comme si c'était de l'information.**
Dans l'exemple précédent il n'y avait *que* du bruit — d'où une importance entièrement fabriquée.
""")

co(r"""
somme_mdi = imp.sum()
var_y     = float(np.mean((y_bruit - y_bruit.mean())**2))
err_train = float(np.mean((complet.predict(X_bruit) - y_bruit)**2))

print(f"  somme des MDI                      : {somme_mdi:.6f}")
print(f"  variance de y − erreur d'entraînement : {var_y - err_train:.6f}")
print(f"  écart                              : {abs(somme_mdi - (var_y - err_train)):.2e}")
print(f"\n  erreur d'entraînement de l'arbre complet : {err_train:.2e}  (nulle : il mémorise)")
""")

md(r"""
L'identité est vérifiée à la précision machine. **Ce n'est pas un défaut d'implémentation de
scikit-learn, c'est une propriété mathématique de la mesure elle-même.**

C'est le résultat central de Scornet (2023), qui en tire une consigne sans détour : *il ne faut
jamais calculer le MDI sur un arbre entièrement développé*.
""")

# ---------------------------------------------------------------- 7
md(r"""
---
## 7. Le second biais : juger sur sa propre copie

Il y a un deuxième problème, **indépendant du premier**.

Rappelez-vous comment une coupure est choisie : l'arbre essaie **toutes** les possibilités et garde
celle qui a l'air la meilleure **sur les données d'entraînement**. Puis le MDI mesure à quel point
cette coupure était bonne… **sur ces mêmes données**.

> C'est noter un élève sur un examen qu'il a lui-même rédigé après avoir vu le corrigé.

Le score est gonflé par construction. Et le biais frappe plus fort les variables qui offrent
**plus de coupures possibles** : une variable continue, ou catégorielle à 50 niveaux, donne plus
d'occasions de tomber par chance sur une séparation qui « marche ».

**Les deux biais sont distincts, et appellent deux remèdes différents :**

| biais | cause | remède |
|---|---|---|
| 1 | l'arbre est trop profond pour la quantité de données | **élaguer** |
| 2 | les mêmes données choisissent *et* notent | **mesurer sur des données non vues** |
""")

# ---------------------------------------------------------------- 8
md(r"""
---
## 8. Remède 1 : mesurer sur des données non vues

L'idée : garder l'arbre tel qu'il a été appris, mais recalculer les baisses d'impureté sur un
**jeu de validation** — des lignes mises de côté, que l'arbre n'a jamais vues.

Pourquoi ça marche : pour une variable de bruit, partir à gauche ou à droite n'a aucun rapport avec
$y$. Sur des données fraîches, les contributions positives et négatives **se compensent**, et
l'importance tombe vers 0. Elle peut même devenir **négative** — ce qui est une information utile,
et pourquoi on ne tronque jamais à zéro.

Deux mesures existent ; le projet les compare :

- **MDI-oob** (Li et al., 2019) — la même formule, évaluée ailleurs ;
- **UFI** (Zhou & Hooker, 2021) — la baisse d'entraînement **plus** celle mesurée en validation.

Utilisons l'implémentation du projet.
""")

co(r"""
import simulate, models, mdi

scenario = simulate.scenario_s1()          # 5 vraies variables + 5 de bruit
d = scenario.n_features
verite = scenario.true_importance
bruit  = scenario.is_noise

rng = np.random.default_rng(0)
X_all, y_all = simulate.sample(scenario, 1200, rng)

# 2/3 pour construire l'arbre, 1/3 mis de côté pour mesurer
ordre = rng.permutation(1200); n_val = 400
val, tr = ordre[:n_val], ordre[n_val:]
centre = float(y_all[tr].mean())                      # on centre y (voir note plus bas)
X_tr, y_tr = X_all[tr], y_all[tr] - centre
X_val, y_val = X_all[val], y_all[val] - centre

cfg = models.ModelConfig(kind="tree")
modele = models.grow(X_tr, y_tr, cfg, seed=0)

mesures = mdi.all_measures(modele, X_val, y_val, d)

print(f"{'variable':>10} {'vérité':>8} {'mdi_train':>11} {'mdi_oob':>9} {'ufi_half':>9}")
for j in range(d):
    etiq = f"bruit {j-4}" if bruit[j] else f"signal {j+1}"
    print(f"{etiq:>10} {verite[j]:>8.3f} {mesures['mdi_train'][j]:>11.3f} "
          f"{mesures['mdi_oob'][j]:>9.3f} {mesures['ufi_half'][j]:>9.3f}")

print(f"\n{'SOMME sur le bruit (vérité = 0)':>36}")
for nom in ("mdi_train", "mdi_oob", "ufi_half"):
    print(f"{nom:>26} : {mesures[nom][bruit].sum():+8.3f}")
""")

co(r"""
fig, ax = plt.subplots(figsize=(7.6, 3.6))
noms = ["mdi_train\n(classique)", "mdi_oob\n(validation)", "ufi_half\n(validation)"]
vals = [mesures[k][bruit].sum() for k in ("mdi_train", "mdi_oob", "ufi_half")]
b = ax.bar(noms, vals, color=[ORANGE, BLEU, VERT], width=.55, zorder=3)
ax.axhline(0, color=INK, lw=1.4, zorder=4)
for r_, v in zip(b, vals):
    ax.text(r_.get_x()+r_.get_width()/2, v + (.012 if v >= 0 else -.03),
            f"{v:+.3f}", ha="center", fontsize=9.5, color=INK)
ax.set_ylabel("importance donnée aux 5 variables de bruit")
ax.grid(axis="y")
ax.set_title("La vérité est 0. Seules les mesures sur données non vues s'en approchent.",
             fontsize=10.5, color=INK)
plt.tight_layout(); plt.show()
""")

md(r"""
**Le remède fonctionne.** Le bruit passe d'une masse franchement positive à quelque chose
d'indiscernable de zéro.

Deux détails techniques que le projet a dû établir, et qui ne sont dans aucun des deux articles :

> **Centrer $y$.** La formule du MDI-oob multiplie par $y_i$. Si $y$ n'est pas centré (des prix,
> par exemple), un terme parasite s'ajoute et détruit la mesure. On soustrait la moyenne
> d'**entraînement** — jamais celle de validation, qu'on n'a pas le droit de regarder.
> L'UFI, elle, est insensible à ce décalage.
>
> **Diviser l'UFI par 2.** Ses deux termes convergent vers *la même* quantité, donc leur somme vaut
> deux fois la vraie baisse. Zhou & Hooker ne s'en servent que pour **classer** les variables, où
> une constante multiplicative est invisible — la question ne se posait pas chez eux.
""")

# ---------------------------------------------------------------- 9
md(r"""
---
## 9. Remède 2 : élaguer l'arbre

L'autre biais vient de la **profondeur**. Un arbre qui mémorise chaque point apprend le bruit.
**Élaguer**, c'est supprimer les coupures qui n'apprennent rien de réel.

Le projet utilise l'**alpha-trimming** (Surjanovic et al., 2024). À chaque nœud on se demande :
*les données justifient-elles statistiquement cette coupure ?* Le critère compare deux modèles —
une seule moyenne contre deux — et pénalise le second parce qu'il a plus de paramètres.

Un unique réglage $\alpha$ contrôle la sévérité : $\alpha = 0$ ne coupe rien, $\alpha$ grand élague
fort. Et surtout, **l'élagage se fait après la croissance** : un seul arbre construit suffit pour
toute la grille.

Ce qui le rend *adaptatif* : le gain d'une coupure grandit comme $|N|$, la pénalité seulement comme
$\log|N|$. Donc on élague **fort là où les données sont rares**, et peu là où elles sont nombreuses.
""")

co(r"""
import prune

base = models.grow(X_tr, y_tr, cfg, seed=0, min_samples_leaf=5)
grille = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)

print(f"{'alpha':>6} {'feuilles':>9}   {'erreur de prédiction':>21}")
feuilles, erreurs = [], []
for a in grille:
    m = prune.trim(base, a)
    err = float(np.mean((m.predict(X_val) - y_val)**2))
    feuilles.append(m.mean_leaves); erreurs.append(err)
    print(f"{a:>6.2f} {m.mean_leaves:>9.0f}   {err:>21.4f}")
""")

co(r"""
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))

axes[0].plot(grille, feuilles, marker="o", color=PRUNE, lw=1.8, ms=6,
             mec="white", mew=1.4, zorder=3)
axes[0].set_xlabel("alpha (sévérité de l'élagage)"); axes[0].set_ylabel("nombre de feuilles")
axes[0].set_title("L'arbre rétrécit", fontsize=10); axes[0].grid()

axes[1].plot(grille, erreurs, marker="o", color=ORANGE, lw=1.8, ms=6,
             mec="white", mew=1.4, zorder=3)
k = int(np.argmin(erreurs))
axes[1].scatter([grille[k]], [erreurs[k]], s=150, facecolor="none",
                edgecolor=VERT, lw=2.2, zorder=4)
axes[1].annotate("meilleure prédiction", (grille[k], erreurs[k]),
                 textcoords="offset points", xytext=(12, 14), fontsize=9, color=VERT)
axes[1].set_xlabel("alpha"); axes[1].set_ylabel("erreur de prédiction")
axes[1].set_title("…et prédit mieux, jusqu'à un point", fontsize=10); axes[1].grid()

fig.suptitle("Élaguer aide la prédiction — mais aide-t-il la mesure d'importance ?",
             fontsize=11, color=INK)
plt.tight_layout(); plt.show()
""")

md(r"""
L'erreur descend puis remonte : il existe une quantité d'élagage optimale. **Trop d'arbre** apprend
le bruit, **trop peu** rate le signal.

Et c'est exactement la question du projet : **ce réglage optimal pour *prédire* est-il aussi le bon
pour *mesurer l'importance* ?** Rien ne l'impose.
""")

# ---------------------------------------------------------------- 10
md(r"""
---
## 10. Comment on sait qui a raison

Problème de méthode : sur des données réelles, **personne ne connaît la vraie importance**. On ne
peut donc pas dire si une mesure se trompe.

La solution : fabriquer un monde dont on connaît les règles.

$$y = f_1(x_1) + f_2(x_2) + \dots + \text{bruit}$$

On **choisit** les fonctions $f_j$, donc on peut calculer exactement la contribution de chaque
variable. C'est sa variance :

$$I_j = \text{Var}\big(f_j(x_j)\big)$$

L'intuition : *de combien cette variable fait-elle bouger $y$ ?* Une variable dont la contribution
ne bouge jamais ne distingue personne — importance nulle. Les variables de bruit ont $f_j = 0$,
donc $I_j = 0$ **exactement** : ce sont nos pièges.

Que ce soit la bonne cible est un théorème de Scornet (2023), valable quand le modèle est additif
et les variables indépendantes.

> **Pourquoi pas des données réelles ?** Parce qu'on ne pourrait rien conclure. Scornet est
> explicite : *« données simulées pour le moment : comme la véritable importance sur les jeux de
> données réelles est inconnue, il sera plus difficile de commenter les résultats. »* Le code pour
> les données réelles existe (`src/realdata.py`) mais reste en attente.

### Les scénarios disponibles

Deux familles, toutes additives à variables indépendantes — la seule situation où $I_j$ existe.

| | d'où ça vient | $d$ | signaux |
|---|---|---|---|
| `S1` | fabriqué pour ce projet : cinq formes de tailles comparables | 10 | 5 |
| `S4` | idem, mais les importances couvrent deux ordres de grandeur | 10 | 5 |
| `DS1`, `DS3`, `DS4`, `DS6`, `DS8` | **Duroux & Scornet (2016)**, section 4 | 30 à 1000 | 2 à 10 |

Les modèles Duroux–Scornet sont ceux que Scornet nous a indiqués. Ils sont plus larges et plus
parcimonieux : `DS8` a 1000 variables dont **4** portent du signal.

> **Trois des huit modèles du papier sont inutilisables ici.** Les modèles 2, 5 et 7 contiennent
> des produits ($\tilde{X}_1\tilde{X}_2$) et des indicatrices à plusieurs variables. Dans ces cas,
> $\mathrm{Var}(f_j(X_j))$ n'est la MDI de population d'**aucun** arbre — il n'y a donc pas de
> vérité contre laquelle mesurer un biais. Le code les refuse explicitement.
""")

co(r"""
# les modeles additifs du papier de Duroux & Scornet
print(f'{"modele":>7} {"d":>6} {"signaux":>9} {"sum I_j":>9}   indices porteurs de signal')
for tag in ("DS1", "DS3", "DS4", "DS6", "DS8"):
    sc_ds = simulate.duroux(tag)
    porteurs = np.flatnonzero(~sc_ds.is_noise)
    apercu = str(porteurs.tolist()[:6]) + ("…" if porteurs.size > 6 else "")
    print(f'{tag:>7} {sc_ds.n_features:>6} {porteurs.size:>9} '
          f'{sc_ds.true_importance.sum():>9.3f}   {apercu}')

for refuse in ("DS2", "DS5", "DS7"):
    try:
        simulate.duroux(refuse)
    except ValueError as e:
        print(f'\n  {refuse} -> refusé : {e}')
""")

co(r"""
print(f"{'variable':>12} {'fonction':>14} {'I_j (vérité)':>13}")
for j, c in enumerate(scenario.components):
    print(f"{('bruit' if bruit[j] else 'signal'):>7} {j+1:<4} {c.name:>14} {c.variance:>13.4f}")

explicable = verite.sum()
print(f"\n  variance explicable  sum(I_j) : {explicable:.4f}")
print(f"  variance du bruit    sigma^2  : {scenario.sigma**2:.4f}")
print(f"  variance totale de y          : {explicable + scenario.sigma**2:.4f}")
print(f"  vérifiée sur l'échantillon    : {y_all.var():.4f}")
""")

co(r"""
fig, ax = plt.subplots(figsize=(6.4, 5.0))
hi = verite.max()*1.18
ax.plot([0, hi], [0, hi], color=SOFT, lw=1.2, zorder=2)
ax.text(hi*.60, hi*.66, "mesure parfaite", fontsize=9, color=SOFT, rotation=41)
for nom, coul in [("mdi_train", ORANGE), ("mdi_oob", BLEU), ("ufi_half", VERT)]:
    ax.scatter(verite, mesures[nom], s=85, color=coul, label=nom, zorder=3,
               edgecolor="white", lw=1.5)
ax.set_xlabel("vraie importance  $I_j$"); ax.set_ylabel("importance estimée")
ax.legend(frameon=False); ax.grid()
ax.set_title("Les points à gauche (x = 0) sont les variables de bruit", fontsize=10.5, color=INK)
plt.tight_layout(); plt.show()
""")

md(r"""
Cette figure résume tout le problème.

- **À gauche** ($x = 0$, les variables de bruit) : l'orange décolle du zéro, le bleu et le vert y
  restent collés. C'est le biais, visible à l'œil.
- **À droite** (les vraies variables) : l'orange suit plutôt bien la diagonale ; le bleu et le vert
  sont **en dessous** — ils sous-estiment.

Chaque mesure est donc juste là où l'autre se trompe. C'est pourquoi un seul chiffre résumé ne
suffit jamais à départager.
""")

# ---------------------------------------------------------------- 11
md(r"""
---
## 11. Le plan d'expérience

Pour attribuer un écart à une cause, il faut ne faire varier **qu'une chose à la fois**. D'où sept
manières de construire l'arbre — les **bras** — toutes comparées sur les mêmes données.

| bras | l'arbre | ce qu'il sert à isoler |
|---|---|---|
| `standard` | complet, sur **toutes** les données | la pratique courante, la référence à battre |
| `full` | complet, sur l'entraînement | l'effet des données séparées, sans élagage |
| `global` | taille de feuille minimale réglée | élagage simple, non adaptatif |
| `ccp` | coût-complexité réglé | élagage classique de CART |
| `alpha0` | départ de l'alpha-trimming, **sans** élagage | le point de comparaison exact de `alpha` |
| `alpha` | alpha-trimming réglé | **la méthode étudiée** |
| `oracle` | alpha-trimming réglé **avec la vérité** | la borne : le mieux que l'élagage pourrait faire |

Deux subtilités qui font toute la rigueur du dispositif :

> **Pourquoi `alpha0` et pas `full` ?** L'alpha-trimming a besoin de partir d'un arbre dont les
> feuilles contiennent ≥ 5 points (sur des feuilles pures, son critère diverge et il ne fait rien).
> Son point de départ n'est donc pas l'arbre complet. Comparer `alpha` à `full` mélangerait
> l'élagage et la différence de profondeur de départ ; `alpha0` est le **même** arbre de départ,
> non élagué.
>
> **Pourquoi un `oracle` ?** Le réglage est choisi par validation croisée, qui optimise la
> **prédiction**. Si `alpha` ne bat pas `alpha0`, deux explications restent : l'élagage n'aide pas,
> ou le réglage était mauvais. L'oracle, qui a le droit de tricher en regardant la vérité, tranche.
""")

# ---------------------------------------------------------------- 12
md(r"""
---
## 12. Ce qu'on a trouvé

Montrons la question centrale directement : parcourons **tous** les niveaux d'élagage et traçons,
côte à côte, l'erreur de **prédiction** et l'erreur d'**importance**.
""")

co(r"""
err_importance, err_prediction = [], []
for a in grille:
    m = prune.trim(base, a)
    est = mdi.all_measures(m, X_val, y_val, d)["ufi_half"]
    err_importance.append(float(((est - verite)**2).sum()))
    err_prediction.append(float(np.mean((m.predict(X_val) - y_val)**2)))

i_pred = int(np.argmin(err_prediction))
i_imp  = int(np.argmin(err_importance))

# deux grandeurs d'unites differentes : deux panneaux empiles partageant l'axe des
# alpha, chacun a son echelle. Superposer sur deux axes verticaux ferait croire a
# des hauteurs comparables.
fig, (haut, bas) = plt.subplots(2, 1, figsize=(8.0, 5.6), sharex=True,
                                gridspec_kw={"hspace": 0.18})

for ax, serie, coul, k, titre, ylab in (
    (haut, err_prediction, ORANGE, i_pred, "Prédire : élaguer aide", "erreur de prédiction"),
    (bas,  err_importance, PRUNE,  i_imp,  "Mesurer l'importance : élaguer nuit",
     "erreur d'importance"),
):
    ax.plot(grille, serie, marker="o", color=coul, lw=2.2, ms=6.5, mec="white", mew=1.5, zorder=3)
    ax.scatter([grille[k]], [serie[k]], s=200, facecolor="none", edgecolor=coul, lw=2.4, zorder=4)
    ax.axvline(grille[k], color=coul, ls=":", lw=1.4, zorder=2)
    ax.annotate(f"minimum : alpha = {grille[k]}", (grille[k], serie[k]),
                textcoords="offset points", xytext=(14, 10), fontsize=9, color=coul)
    ax.set_ylabel(ylab, fontsize=9.5)
    ax.set_title(titre, fontsize=10, color=INK, loc="left")
    ax.grid()

bas.set_xlabel("alpha (sévérité de l'élagage)")
fig.suptitle("Les deux objectifs ne demandent pas le même réglage", fontsize=11, color=INK)
plt.tight_layout(); plt.show()

print(f"  meilleur alpha pour PRÉDIRE            : {grille[i_pred]}")
print(f"  meilleur alpha pour MESURER L'IMPORTANCE : {grille[i_imp]}")
""")

md(r"""
> **Lecture.** Deux panneaux plutôt qu'une superposition : les deux grandeurs n'ont pas la même
> unité, et les mettre sur un même graphe ferait croire que leurs hauteurs se comparent. Ce qui
> compte est **où chaque courbe atteint son minimum** — et les deux pointillés ne tombent pas au
> même endroit.

### Les résultats du protocole complet

Sur l'ensemble des scénarios et des niveaux de bruit (20 répétitions, deux tailles d'échantillon) :

**1 · Le remède « données séparées » marche, et c'est le résultat le plus solide.**
L'importance attribuée à des variables prouvées inutiles chute d'un facteur allant jusqu'à 40, et
ce sur toute la gamme de bruit testée.

**2 · L'élagage n'ajoute rien par-dessus.** Aucun niveau d'élagage n'améliore la mesure sur données
séparées. L'oracle lui-même — qui a le droit de tricher — refuse d'élaguer.

**3 · Et il y a une raison mécanique.** Les deux remèdes s'attaquent à **la même chose** : le bruit
absorbé par un arbre trop profond. L'élagage le retire en raccourcissant l'arbre ; la mesure sur
données séparées le retire en refusant de créditer une coupure qui ne se reproduit pas. Appliquez
le second, il ne reste plus rien à retirer au premier — qui ne fait alors que coûter de la
résolution.

> **Ce sont des substituts, pas des partenaires.** C'est une réponse complète à la question posée,
> et elle est négative — ce que la formulation « évaluer la pertinence » invitait explicitement.

**4 · Le résultat le plus prometteur pour la suite** est celui de la figure ci-dessus : le réglage
bon pour prédire n'est pas celui qui mesure le mieux. Un critère de choix pensé pour l'importance,
et non pour la prédiction, serait une contribution à part entière.

---

### Pour aller plus loin

- `PROTOCOL.md` — le protocole complet et les définitions formelles
- `src/mdi.py` — les trois mesures, une dizaine de lignes chacune
- `src/prune.py` — l'alpha-trimming, transcrit de l'algorithme 1 de l'article
- `tests/test_all.py` — 51 tests, dont l'identité exacte qui valide tout le reste

**Les trois articles**

1. E. Scornet (2023), *Trees, forests, and impurity-based variable importance in regression* — pourquoi les arbres profonds se trompent
2. Z. Zhou & G. Hooker (2021), *Unbiased measurement of feature importance in tree-based methods* — mesurer sur des données non vues
3. N. Surjanovic, A. Henrey & T. M. Loughin (2024), *Alpha-Trimming* — l'élagage adaptatif
""")

nb["cells"] = C
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
import os
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/cours.ipynb")
print(f"écrit : notebooks/cours.ipynb — {len(C)} cellules")
