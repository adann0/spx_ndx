# Architecture

Le système décide d'être investi (IN) ou en cash (OUT) à chaque mois, en combinant une trentaine de signaux techniques, macro et fondamentaux via un pipeline à 3 niveaux (traders > groupes > ensemble). Les combinaisons de signaux les plus robustes survivent à chaque étape, se combinent entre elles, et votent collectivement.

## Dénomination

- Indicateur : signal binaire à chaque date (ex: RSI>30, CPI<4). ~30 indicateurs.
- Trader : (N, (sig_1, ..., sig_k)), investi si au moins N signaux sont ON. 2 à 5 signaux, tous les N possibles.
- Groupe : paire de 2 traders. top 50 gardés par ensemble.
- Ensemble : un pool de groupes filtrés par un seuil CAGR donné (8% à 10%). 3 ensembles, chacun produit un 0 ou 1.
- Consensus : majorité des ensembles (≥ 2/3) : décision finale IN/OUT.

## Architecture Globale

- Génération de toutes les combinaisons de trader (`(N, (sig_1, sig_2, ..., sig_k))`) 
- Découpage en folds du dataset, expanding window, K folds de 2 ans, train commence en 1993, premier test en 1999.
- Itération sur chaque fold
  - Gridsearch sur les hyperparamètres (vote_threshold, trader_min_rtr)
  - Selection du meilleur combo d'hyperparamètres
  - Test du meilleur combo
  - Results, Explainability & Metrics Export
- Global Results, Explainability & Metrics Export

## Gridsearch sur les hyperparamètres

1. Évaluation des ~835K traders sur le train. On génère toutes les combinaisons de 2 à 5 signaux, avec le seuil de vote N testé de 1 à k. Chaque trader (N, (sig_1, ..., sig_k)) est investi quand au moins N de ses signaux sont ON. Ex : (2, (SMA_10m, VIX<25, RSI<70)) = IN si au moins 2/3 actifs. Évalué via Numba parallèle. On calcule son CAGR, RTR, stability, etc.
2. Filtre : RTR >= trader_min_rtr (gridsearch).
3. Pour chaque seuil CAGR (8%, 9%, 10%) (3 seuils pour éviter de hardcoder un unique filtre et pondérer les stratégies les plus populaires et efficientes) :
  - Filtre les traders par cagr >= seuil.
  - Tri par ("stability", "cagr") (top 100 trader différent par seuil : un trader a 8% CAGR peut être plus stable qu'un trader a 10% CAGR).
  - Forme l'ensemble de tout les groupes possibles à partir de ces traders (`(N, (trader_0, ..., trader_k))`).
  - Evalue chaque groupe sur le train. Chaque groupe est investi si au moins N de ses traders sont ON. On calcule le CAGR, RTR, stability, etc de chaque groupe. 
  - Garde le top 50, filtré par RTR >= 0 et CAGR >= 0 (seuil de sécurité), trié par ("stability", "cagr"). Poids de vote proportionnels au CAGR train de chaque groupe.
  - Ça donne un ensemble.
4. Évaluation des ensembles sur le train : les 3 ensembles votent. Majorité 2/3. Chaque ensemble dit 0 ou 1 en fonction du seuil d'hyperparamètre gridsearch `vote_threshold` (compris dans (0.5, 0.6, 0.7)). Si la majorité des ensembles retournent 1, le consensus envoi un signal d'achat, sinon il envoi un signal de vente. (Note : l'évaluation peut se faire sur un split validation séparé du train via ADAPTIVE_VAL_YEARS. Actuellement désactivé).
5. On retourne le CAGR du consensus avec ses hyperparamètres sur le train.
6. Sélection du meilleur combo d'HP : trié par (cagr, rtr) sur le train, filtré par stability >= baseline (mediane des hyperparams), médiane entre les hyperparams ex-æquo si plusieurs hyperparams ont le même (cagr, rtr).
7. On fait ensuite le test sur le meilleur combo avec le même principe que l'évaluation des ensembles sur le train.

## Explainability

Décomposition statique des poids le long du graphe groupes > traders > signaux. Le poids de chaque groupe est réparti uniformément entre ses traders, puis entre les signaux de chaque trader. Normalisé par pipeline, moyenné sur les 3 pipelines. Donne une approximation linéaire `vote ≈ w₁·sig₁ + w₂·sig₂ + ...`.

## Sécurités

1. Shift(1) des signaux : les indicateurs calculés au mois M sont décalés d'un mois > utilisés pour la décision du mois M+1. Empêche le look-ahead bias.
2. Dataset month-end uniquement : resample("ME").last() + drop de la dernière ligne si incomplète. Empêche d'utiliser des données partielles en milieu de mois.                     
3. Expanding window : le train ne voit jamais le futur. Chaque fold n'utilise que les données antérieures au test.
4. Majorité dynamique : `ceil(n_valid/2)`. Crash si 0 valides (n'arrive jamais en pratique, crash la pipeline GitHub actions et retourne une erreur si jamais).
5. Filtre stability >= baseline (mediane des hyperparamètres) : empêche de sélectionner un combo d'HP rentable en train mais instable.
6. Médiane des combo ex-æquo : évite de sélectionner un HP extrême par chance (ou le premier qui arrive dans la liste par "hasard").
7. Seuils de sécurité groupes : RTR >= 0 et CAGR >= 0 (élimine les groupes à rendement négatif).

### Stress Tests

**Significance**: Permutation test (N=1000), bootstrap (IC 95% sur CAGR/Sharpe).

**Robustness**: Signal noise injection (flip 1-30%), return noise injection, sensibilité aux coûts de transaction.

**Stability**: Vintage analysis, decade split, regime split (VIX/bull/bear/crisis), drawdown analysis, rolling Sharpe 36m, cumulative alpha.

**Complementary**: Block bootstrap (blocs 6m), corrélation B&H, Monte Carlo, rolling alpha 3Y/5Y, baseline signal correlation, cross-index NDX/MSCI World.
  
## Limites et biais résiduels

- Overfitting temporel : 3 crises en 25 ans (DotCom, GFC, COVID).
- Publication bias : Les signaux sont des indicateurs connus ; le risque qu'ils aient été sélectionnés parce qu'ils ont fonctionné historiquement ne peut pas être éliminé.
- Regime shift : Les relations macro peuvent changer structurellement (post-2008, post-COVID).
- Biais bull market : La sélection favorise les signaux ON 70-80% du temps. Ils sortent bien en crise mais sont aveugles à la recovery : ils attendent la confirmation du bull avant de re-rentrer.
- Fréquence monthly : Impossible de capter les V-shaped recoveries. Les SMA mettent 4-5 mois à se retourner après un bottom. On évite les gros drawdowns mais on paie en retard au retournement (le prix doit rattraper une SMA qui descend).
- Signaux rares éliminés : Les signaux de recovery (credit spread, NFCI, unemployment) fire 3-4 fois en 25 ans et perdent la compétition face aux SMA/VIX utiles partout.
- Modèle unique : Un seul ensemble monthly essaie de tout faire (bull, crisis, recovery). Des modèles séparés par régime seraient plus adaptés.
- Limites d'indicateurs et pondération : Dans la vraie vie, un trader n'est pas limité à 5 indicateur et il pondère chaque indicateur en fonction du contexte (ce n'est pas un IN/OUT fixe sur 20 ans avec "si j'ai 3 indicateurs sur 5 ON je suis IN") (_TODO en V2_).
- Execution price : Le backtest simule une entrée/sortie au close du mois, mais en live on exécute au début de la période suivante (si on attend d'avoir les données de la fin du mois, on peut pas éxécuter au close du mois, on est obligé d'éxécuter a l'open du mois suivant ou dans la journée). Améliorer en mesurant le return open/open plutôt que close/close et intégrer la stratégie d'éxécution intraday (VWAP, etc).

## Hyperparamètre

| HP | Valeurs testées | Raison d'exclusion | Valeur fixée |
|---|---|---|---|
| `vote_threshold` | 0.5, 0.6, 0.7 | **Dans la grid**: varie légitimement par fold | grid |
| `trader_min_rtr` | 0.5, 0.6, 0.7 | **Dans la grid**: varie légitimement par fold | grid |
| `top_traders` | 50 > 1000 | −1.5pp CAGR, +8.2pp MaxDD ; valeurs dispersées par fold = overfitting inner_val | 100 |
| `top_groups` | 10, 25, 50, 75, 100 | −0.9pp CAGR, +1.3pp MaxDD ; même pattern d'overfitting | 50 |
| `group_min_cagr` | 0.01, 0.05, 0.08 | −0.7pp CAGR, +3.3pp MaxDD ; seuils variables par fold = overfitting | 0 |
| `ensemble_majority` | 2, 3, 4 | +5.1pp MaxDD ; remplacé par `ceil(n_valid/2)` dynamique | dynamique |
| `max_traders_per_group` | 2, 3 | −0.5pp CAGR ; triplets trop contraignants, moins de groupes valides | 2 |
| `min/max_signals_per_trader` | [1–3] × [3–7] | max > 5 : combos trop spécifiques au train (fold 11 : 21.6% > 13.8%) | min=2, max=5 |
| `cagr_thresholds` | 3 plages | Paramètre structurel (définit les 3 pipelines), pas un HP à sweeper | 7%–11% |
| `trader_sort` | 6 combos | Résultats = baseline ; médiane sur tuples de strings sans sens sémantique | (stability, cagr) |
| `group_sort` | 5 combos | Idem | (stability, cagr) |
| `aggregation` | 5 méthodes | −0.2pp CAGR ; médiane alphabétique sans sens | cagr_weighted |
| `weight_exponent` | 1, 2, 3 | CAGRs des groupes trop proches: amplifier les écarts ne change rien | 1 |
| `round_sort` | False, True | −0.3pp CAGR ; l'arrondi détruit la discrimination au 4e décimale | False |
| `group_min_sharpe` | 0.01, 0.1, 0.3 | Sélectionne toujours la médiane (0.1) ; filtre inactif en pratique | 0.01 |
| `trader_min_stability` | | Redondant : le tri par stability primaire élimine déjà les instables | |
| `trader_min_calmar` | | Redondant : le tri par (stability, cagr) fait le ménage | |
| `group_min_stability` | | Redondant : même logique | |
| `group_min_calmar` | | Redondant : idem | |
| `min_traders_per_group` | 1, 2, 3 | Sélectionne toujours 2 ; résultats = baseline | 2 |
