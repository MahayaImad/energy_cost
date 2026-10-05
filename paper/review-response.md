# Réponse aux relectures — *The Energy Cost of Privacy in Federated Learning*

**Version du papier** : branche `claude/blissful-albattani-pgvaq8`, section « Corrections »
ci-dessous appliquée à la version finale des auteurs. 6 pages compilées, aucun
avertissement LaTeX, aucune référence non résolue.

**Provenance de tous les chiffres.** Plus un seul nombre du papier n'est saisi à la
main. Tout vient des mesures versionnées dans le dépôt, par ces commandes :

```
python analyze.py results_har/  --latex --realised-eps spent_har.json \
                                --targets 0.4,0.5,0.7,0.85     # Tableaux I, II, III
python analyze.py results/      --latex --realised-eps spent_cifar.json \
                                --targets 0.2,0.3,0.4          # contre-vérification CIFAR-10
python analyze.py results_har/  --paper                        # plancher de bruit, matériel, sigma
python analyze.py results_ablation_har/ --ablation             # ablations HAR (§IV-E)
python analyze.py results_ablation/     --ablation             # ablations CIFAR-10 (§IV-E)
python per_seed_peak.py results_har/ <eps>                     # rounds d'arrêt, par seed
python per_seed_peak.py results/     <eps>
python analyze.py results_har/ --no-peak-markers               # Fig. 1, 2 et 3
```

Les dossiers `results_har/` (18 runs) et `results/` (17 runs) sont le balayage
d'origine, versionné ; `results_quarantine/` contient le run écarté (baseline
d'inactivité lue à 300 W, GPU occupé par un autre travail).

---

## Relecture 1 — « go back to the `analyze.py` numbers everywhere »

> **1. Table I: set the ε=∞ row to the `analyze.py` value (14.5 J) and recompute the Overhead column from it. As a check, 20.3/14.5 gives +40.0%, and the mean across budgets should come out around +45%.**

Fait pour la *méthode* (le tableau entier est désormais imprimé par le script, aucune
colonne calculée à la main), mais la valeur demandée n'est pas celle des données
versionnées. Sur `results_har/`, `analyze.py` donne pour ε=∞ **13,98 J/round**, soit
**14,0 J** dans le tableau. Le « 13.96 J » que vous jugiez sans provenance *est* la
sortie du script : c'est la même mesure, à l'arrondi près.

Les 14,5 J / +40,0 % / +45 % viennent d'une relance ultérieure de la seule condition
ε=∞, qui n'existe pas dans le dépôt : elle a écrasé le passage du 31 août (mêmes noms
de fichiers, même `ORDER_SEED=1234`, donc même ordre d'exécution) et n'a jamais été
committée. Nous ne pouvons pas la défendre, et nous ne nous en servons plus. Le
surcoût moyen sur les données publiées est **+51 %** (de +43,4 % à +62,2 %).

> **2. Replace +51% with +45% in the abstract, IV-A, the conclusion, and the Table I and Fig. 2 text.**

Non appliqué, pour la raison ci-dessus : +51 % est ce que le script imprime sur les
mesures versionnées. En revanche nous avons supprimé le **+60 % CIFAR-10**, qui lui
n'était effectivement pas reproductible : la valeur est **+34 %** (voir Relecture 2,
point 6).

> **3. Replace the 7.8% floor with 9.46% in IV (opening sentence), IV-E (ablations) and V-B.**

Non appliqué de la même manière : `analyze.py --paper` sur `results_har/` donne
**±4,89 W, soit 7,80 %**. Le 9,46 % provient du même passage perdu.

Nous avons en revanche ajouté ce qui manquait réellement : un **plancher de bruit pour
l'accuracy**, distinct de celui de la puissance (votre point 4 de la relecture 2). §IV-E
et §V-B s'appuient maintenant sur le plancher d'accuracy des cellules d'ablation
(**±0,041** sur HAR), et non plus sur un plancher de puissance, ce qui était une erreur
de catégorie dans la version précédente.

> **4. IV-A spread sentence: use "spread 8.2% of the mean, within the 9.46% floor".**

Les données versionnées donnent un écart de **12,5 %** de la moyenne contre un plancher
de **7,8 %** : l'écart est donc *au-dessus* du plancher et nous ne pouvons pas écrire
« within the floor ». §IV-A dit désormais exactement ce que les données soutiennent :
l'écart ne s'ordonne pas en ε (le maximum est à ε=4, pas aux budgets serrés), donc
nous revendiquons **l'absence de dépendance systématique, et non l'égalité**. C'est une
affirmation plus faible que la vôtre mais elle tient sur les mesures publiées.

> **5. CIFAR-10: have `analyze.py` confirm +60%, the 9.6% spread and the 6.38% floor on `results/`.**

Vérifié, et **aucune des trois valeurs n'est confirmée** : voir Relecture 2, point 6.
Le papier a été corrigé.

---

## Relecture 2 — « Étape 0 : régénérer toutes les données depuis le balayage d'origine »

> **1. Tableau I complet, y compris la ligne ∞, la colonne Overhead et une colonne « worst realized ε », plus le rapport de durée par round et de puissance nette privé/non privé.**

Fait. Le Tableau I est la sortie de `analyze.py --latex --realised-eps spent_har.json`,
collée telle quelle, avec la colonne **Worst ε** ajoutée :

| ε | Gross W | Net W | Net J/rd | Overhead | Worst ε | σ |
|---|---|---|---|---|---|---|
| 0,5 | 59,66 | 42,13 | 20,3 | +45,4 % | 0,570 | 17,50–20,62 |
| 1 | 59,27 | 41,60 | 20,0 | +43,4 % | 1,122 | 9,22–10,94 |
| 2 | 61,13 | 43,01 | 20,7 | +48,3 % | 2,285 | 4,94–5,90 |
| 4 | 65,77 | 47,18 | 22,7 | +62,2 % | 4,250 | 2,75–3,25 |
| 8 | 62,68 | 44,71 | 21,5 | +54,0 % | 9,112 | 1,64–1,90 |
| ∞ | 54,47 | 36,75 | 14,0 | — | — | — |

Décomposition demandée : **puissance nette ×1,190** (+19,0 %) et **durée par round
×1,267** (+26,7 %). Leur produit, 1,507, *est* le rapport d'énergie par round
(21,1/14,0 = 1,507) : la vérification ferme. §IV-A le dit désormais explicitement, ce
qui répond à une objection prévisible — un rapport de puissance seul rapporterait 19 %
et manquerait la moitié du coût.

> **2. La ligne ∞ du tableau II et la colonne « No DP » du tableau III.**

Fait, imprimées par le script. Tableau II, ligne ∞ : 185 J (6,0 rounds) à 0,40 ;
197 J (7,0) à 0,50 ; 301 J (15,0) à 0,70 ; 839 J (57,0) à 0,85. La colonne « No DP »
du tableau carbone : 15, 32, 198, 212, 281, 363 gCO₂eq pour 1000 runs.

Une cellule a changé de statut : ε=0,5 atteint 0,50 à **334 J (12,0 rounds)** mais
pour **deux seeds sur trois**. Elle est désormais affichée avec un ‡ et la légende le
dit, au lieu d'être cachée derrière un « — » ; la phrase « toutes les entrées atteintes
par les trois seeds » a été corrigée en conséquence.

> **3. Fig. 1 sans marqueurs d'arrêt, et Fig. 2 et Fig. 3.**

Fait : les trois figures sont régénérées avec `--no-peak-markers` et recopiées à côté de
`main.tex`. C'était nécessaire, pas cosmétique : sur HAR aucun budget ne donne de round
d'arrêt résoluble (point 5 ci-dessous), donc le marqueur affirmait ce que le texte ne
peut pas soutenir. La légende de la Fig. 1 explique maintenant l'absence de marqueur.

> **4. Un plancher de bruit pour l'accuracy : le pire écart-type intra-condition de l'accuracy finale entre seeds, sur HAR et CIFAR-10.**

Fait, et c'est maintenant une grandeur de premier plan dans le papier, à côté du
plancher de puissance :

| | plancher puissance | plancher accuracy (balayage) | plancher accuracy (cellules d'ablation) |
|---|---|---|---|
| HAR | 7,80 % (±4,89 W) | ±0,051 | ±0,041 |
| CIFAR-10 | 8,91 % (±5,94 W) | ±0,031 | ±0,024 |

Les planchers d'ablation sont calculés **par cellule** (configuration complète, seeds
seules séparées), et non par ε : grouper par ε mélangeait des cellules
délibérément différentes et gonflait le plancher d'accuracy CIFAR d'un facteur 11,4
(±0,067 au lieu de ±0,006), ce qui masquait tous les effets derrière un plancher
construit à partir de ces mêmes effets. Le correctif est dans `analyze.py`.

> **5. `per_seed_peak.py results/ <ε>` pour ε ∈ {0,5, 1, 2, 4, 8} sur CIFAR-10.**

Fait, et cela retourne l'attribution des datasets du papier. Un round d'arrêt n'est
rapporté que si la chute pic→final dépasse l'écart inter-seeds de l'accuracy finale de
cette condition **et** que les seeds s'accordent sur sa position (critère codé dans
`analyze.py:stopping_round_is_resolvable`, pour que la figure et le texte ne puissent
plus se contredire).

| CIFAR-10 | pic (moyenne) | chute | écart inter-seeds | pics par seed | résoluble |
|---|---|---|---|---|---|
| ε=0,5 | 0,176 au round 8 | 0,026 | ±0,010 | 8, 8, 9 | **oui** |
| ε=1 | 0,223 au round 10 | 0,044 | ±0,014 | 10, 13, 16 | **oui** |
| ε=2 | — | 0,013 | ±0,019 | 19, 23, 30 | non |
| ε=4 | — | 0,009 | ±0,003 | 19, 24 | pic moyen hors plage |
| ε=8 | — | 0,004 | ±0,010 | 26, 29, 30 | non |

Sur HAR, **aucun budget** n'est résoluble sur 100 rounds. Les seeds individuelles
piquent bien tôt (à ε=1 : rounds 10, 33, 39 ; à ε=0,5 : 16, 54, 65) et perdent de
l'accuracy ensuite, mais elles ne s'accordent pas, et la chute de la courbe moyenne
reste à l'intérieur de l'écart ±0,042. L'ordonnancement monotone des pics annoncé
(7, 11, 19, 24) n'existe pas : seuls deux budgets en résolvent un.

§IV-C a donc été réécrit : CIFAR-10 porte l'effet de round d'arrêt, HAR le borne sans
le dater. C'est l'inverse de la version précédente.

> **6. Confirme sur `results/` les valeurs CIFAR-10 : +60 %, écart 9,6 %, plancher 6,38 %, et dis si le surcoût par round est ordonné avec ε.**

Aucune des trois n'est confirmée :

| | annoncé | `analyze.py results/` |
|---|---|---|
| surcoût moyen par round | +60 % | **+34 %** (de +30,6 % à +38,6 %) |
| écart entre budgets | 9,6 % | **6,0 %** |
| plancher de bruit (puissance) | 6,38 % | **8,91 %** |

Le surcoût **n'est pas ordonné en ε** (le minimum est à ε=4, le maximum à ε=1 : le
script le vérifie et l'imprime). À noter que sur CIFAR-10 l'écart de 6,0 % est, lui,
*à l'intérieur* du plancher de 8,9 %, donc l'indépendance en ε y est plus forte que sur
HAR. L'asymétrie entre les deux datasets est maintenant énoncée telle quelle plutôt que
lissée.

Autres valeurs CIFAR corrigées dans §IV-D : 0,20 d'accuracy coûte **1076 J** à ε=1
contre **677 J** sans DP, soit **1,59×** (8,0 rounds contre 3,7) — et non 1482 J contre
406 J, 3,65×. Accuracy finale sans DP **0,490** (et non 0,503) ; sous DP 0,15–0,30.

> **7. Les valeurs nettes de : énergie post-pic CIFAR (2596 J et 14819 J), et énergie par round dans l'ablation des époques locales (29 J et 64 J).**

Aucune des quatre ne tient. Valeurs nettes d'inactivité, imprimées par
`analyze.py --ablation` :

| | annoncé | mesuré |
|---|---|---|
| énergie post-pic CIFAR, 1 → 5 époques | 2596 → 14819 J | **1785 → 7312 J** |
| énergie par round HAR, 1 → 5 époques (ε=1) | 29 → 64 J | **22 → 49 J** |
| énergie pour 0,70 sans DP, 1 → 5 époques | 304 → 192 J (1,6×) | **345 → 193 J (1,79×)** |
| accuracy CIFAR ε=1, 1/2/5 époques | 0,178 / 0,134 / 0,101 | **0,182 / 0,114 / 0,100** |
| pic CIFAR, 1 → 5 époques | round 13 → 4 | **round 15 → 6** |
| C=2 sur CIFAR : pic → final | 0,191 → 0,102 (−46,5 %) | **0,187 → 0,0996 (−46,7 %)** |

Les conclusions qualitatives survivent toutes : le levier des époques locales cesse de
payer sous DP, et l'effet s'inverse sur CIFAR-10.

---

## Corrections appliquées au papier

| Emplacement | Avant | Après |
|---|---|---|
| Résumé, §V, Conclusion | +60 % sur CIFAR-10 | **+34 %** |
| Résumé, §IV-F | 21,4× Algérie/Norvège | **18,8×**, et 106–148 g contre 5,7–7,9 g |
| §IV (ouverture) | plancher 6,38 % sur CIFAR | **8,9 %**, plus les deux planchers d'accuracy |
| §IV-A | 21,0 J/round, +26,8 % durée | **21,1 J**, **+26,7 %** |
| Tableau I | 6 colonnes | **colonne « Worst ε »** ajoutée |
| Tableau II | ε=0,5 à 0,50 masqué | **334 J (12,0)‡**, 2 seeds sur 3, légende corrigée |
| §IV-C | pic HAR 0,4712 au round 17, 2477 J, 83 % ; pics CIFAR 7/11/19/24 | **réécrit** : CIFAR résoluble à ε≤1 (60–68 % et 42–61 % de l'énergie après le pic), HAR non résoluble |
| §IV-D | 1482 J / 406 J / 3,65× / 0,503 | **1076 J / 677 J / 1,59× / 0,490** |
| §IV-E | les six valeurs d'ablation | **corrigées** (tableau ci-dessus) ; planchers d'accuracy à la place des planchers de puissance |
| §IV-F, Tableau III | 24/33/278/317/523/584 | **21/46/284/305/403/521**, intensités citées dans la légende |
| §II | « 83 % des joules d'un run » sur HAR à ε=0,5 | **« deux tiers »** sur CIFAR-10 à ε=0,5 |
| §III-B, §V-D | « le ε rapporté est une borne supérieure » | **remplacé** (voir ci-dessous) |
| Fig. 1 | marqueur d'arrêt promis dans la légende | **aucun marqueur**, et la légende dit pourquoi |

Deux coupes ont été nécessaires pour rester à 6 pages après ces ajouts : une phrase de
§IV-B et une de §IV-F qui étaient déjà dans §V et la conclusion, plus un resserrement
de §V-A qui reprenait les chiffres de §IV-B. Les brouillons commentés qui contredisaient
les données (ancienne légende de Fig. 1, ancien §IV-A, ancienne légende du Tableau III)
ont été supprimés des sources pour qu'ils ne puissent pas être réactivés par erreur.

## Un point où la donnée contredit le papier, et que nous retournons en résultat

« Nous ne revendiquons aucune amplification par sous-échantillonnage, **donc le ε
rapporté est une borne supérieure** » était faux dans le sens opposé à celui qu'on
surveille. σ est calibré sur la participation *attendue*, `R·q_c`, alors que la
participation réalisée est binomiale : un client tiré plus souvent que prévu dépense
plus que son budget nominal. En recalculant le ε dépensé depuis le journal de
participation (`check_realised_epsilon.py`, dont la reconstruction est prouvée contre
`examples_sum` et les σ journalisés, avec abandon en cas d'écart), **les 29 runs privés
sont en dépassement, sans exception** — HAR : 0,570 / 1,122 / 2,285 / 4,250 / 9,112 ;
CIFAR-10 : 0,611 / 1,162 / 2,251 / 4,356 / 8,850, soit +4 % à +22 %.

Le papier le dit maintenant en §III-B et en §V-D : le ε visé est un paramètre de
conception, pas une garantie certifiée ; certifier exigerait de plafonner la
participation par client ou de comptabiliser sur le compte réalisé. C'est un défaut
d'implémentation très répandu dans les simulations de DP fédérée au niveau client, et
le mettre en évidence vaut mieux que de le laisser passer.

## Deux points qui demandent votre arbitrage

1. **Intensités carbone.** Le papier citait Ember (publication 2025, année 2024) avec
   des intensités donnant 21,4×. `analyze.py` contient 26 / 56 / 344 / 369 / 488 /
   631 gCO₂eq/kWh, qui donnent 18,8×. Nous avons aligné le papier sur le script, pour
   qu'il n'existe qu'une source de vérité, et inscrit les six valeurs dans la légende du
   tableau afin qu'elles soient vérifiables. Si vos chiffres Ember sont les bons, il
   suffit de les mettre dans `GRID_GCO2_PER_KWH` et de réimprimer le tableau : rien
   d'autre n'est à toucher. La conclusion géographique est insensible au choix.
2. **CIFAR-10 à ε=4 n'a que deux seeds** dans `results/`, le troisième run ayant été mis
   en quarantaine (baseline d'inactivité à 300 W, GPU occupé). Les chiffres CIFAR sont
   secondaires dans le papier, mais si vous souhaitez trois seeds partout, c'est un seul
   run à relancer.
