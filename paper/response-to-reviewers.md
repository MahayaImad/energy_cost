# Response to Reviewers
### *The Energy Cost of Privacy in Federated Learning* — cSIS 2026, Track 1

We thank both reviewers. Two of the findings are substantial and we have
accepted them in full: the privacy guarantee was mislabelled (R2.3), and our
first contribution rested on an unreconciled discrepancy between two
measurement passes (R2.1). Both are now fixed in a way that makes the error
impossible to repeat, and one of them has become a result in its own right.

**Where every number now comes from.** The paper no longer contains a figure
typed by hand. All tables and all in-text values are printed by the analysis
script from the same arrays, on the committed measurements:

```
python analyze.py results_har/ --latex --realised-eps spent_har.json \
                  --targets 0.4,0.5,0.7,0.85            # Tables I, II, III
python analyze.py results/     --latex --realised-eps spent_cifar.json \
                  --targets 0.2,0.3,0.4                 # CIFAR-10 check
python analyze.py results_har/ --paper                  # floors, hardware, sigma
python analyze.py results_ablation_har/ --ablation      # Section IV-E, HAR
python analyze.py results_ablation/     --ablation      # Section IV-E, CIFAR-10
python per_seed_peak.py  results_har/ <eps>             # stopping rounds
python per_seed_peak.py  results/      <eps>
python check_realised_epsilon.py results_har/ --worst-case
python analyze.py results_har/ --no-peak-markers        # Figures 1 and 2
```

The paper is now **6 pages**, verified by compilation, with no LaTeX warnings
and no unresolved references.

---

## Reviewer 1

> **The paper runs one page over the six-page limit.**

Fixed: 6 pages. The cuts are the ones R2.6 identified — Section V-A reduced to
its interpretive content, V-B's restatement of IV-E removed, the carbon figure
dropped because Table III is the same six numbers, captions shortened, and the
overlap between the methodology's limitations and Section V-E's threats merged
into one place. Section V-D — the measurement hazard, numbered V-E in the submitted version —
is untouched; its three controls now cross-reference the methodology instead
of restating them.

> **Uses "Index Terms" rather than "Keywords."**

Fixed with `\renewcommand{\IEEEkeywordsname}{Keywords}`.

> **Sets the title in non-bold font.**

Fixed; the title is now bold.

> **Figure and table captions are too long.**

Every caption is now at most three lines, with the numbers moved into the
text where they can be read against the surrounding argument.

> **Two numbers in the summary of our contribution have changed.**

We note them so the summary can be re-read against the corrected paper: the
moderate-target cost is **5.7×** (1710/301 = 5.68, as R2.2 computed), and the
carbon factor is **18.8×**, now presented as context rather than as a result
about privacy, for the reason R2.4 gives.

---

## Reviewer 2

### 1. Two baselines and two noise floors for the same measurement

The reviewer is right, and the discrepancy had two independent causes.

**The baselines.** Figure 2 and the opening of Section IV were taken from a
later re-run of the $\varepsilon=\infty$ condition alone, which overwrote the
original sweep's files — identical filenames, identical run-order seed — and
was never committed. Table I was computed from the original sweep. We can
defend only the committed data, so the re-run is gone and the whole paper now
comes from the original sweep: baseline **13.98 J/round** (14.0 in Table I),
mean overhead **+51%**, range +43.4% to +62.2%, board-power floor **7.80%**.
The 14.5 J / +45% / 8.2% / 9.46% set is withdrawn.

**The floor comparison was the wrong instrument**, and this is the more
important half of the reviewer's point. Two things were wrong with it:

- The 12.5% spread is a spread in **net energy per round**, while the 7.8%
  floor is a spread in **board power**. The two differ by the round duration,
  which also varies across seeds. Every floor in the paper is now measured on
  the quantity it bounds. On energy per round the floor is **±2.58 J/round,
  12.0%** — not 7.8% — so the 12.5% spread sits essentially on it.
- Even with matched units, a floor is the spread of *one* condition's seeds,
  whereas the claim is about the spread of *five condition means*, which with
  three seeds each is narrower by about $\sqrt{3}$. So we test the claim
  instead of comparing it to a floor: permuting the seed labels across the
  five private budgets gives **F = 1.46, p = 0.27** on HAR and **F = 0.46,
  p = 0.79** on CIFAR-10.

On the specific contrast the reviewer raises, +43.4% at $\varepsilon=1$ against
+62.2% at $\varepsilon=4$: it is not resolvable. The three seeds of the
$\varepsilon=4$ condition span 20.9 to 24.1 J/round on their own, and the
overheads are not ordered in $\varepsilon$ ($\varepsilon=8$ sits below
$\varepsilon=4$).

Section IV-A is retitled *The Per-Round Cost Does Not Track the Budget* and
states the weaker claim the data supports: no resolvable dependence on
$\varepsilon$, and explicitly **no claim of equality** — three seeds per budget
bound a budget effect at about the size of the overhead's own scatter, not at
zero.

### 2. Headline figures not reproducible from the tables

Every one is corrected, and the text values are now printed from the same
arrays as the tables rather than transcribed.

| Claim | Submitted | Corrected |
|---|---|---|
| ratio at the 0.50 target | 1.17–2.10 | **1.28–1.79** (as the reviewer computed) |
| moderate-target cost | 5.8× | **5.7×** (1710/301 = 5.68) |
| non-private at 0.85 | 878 J, 57.7 rounds | **839 J, 57.0 rounds** |
| post-peak energy, HAR $\varepsilon=0.5$ | 2477 J over 83 rounds | **withdrawn** (see point 4) |
| ablation per-round at $\varepsilon=1$ | 29 J | **22 J** |

On the 2477 J: the reviewer's arithmetic is what exposed it. 2477 J over 83
rounds is 29.8 J/round against Table I's 20.3 J net, because that number came
from the withdrawn pass. On the committed data HAR has no resolvable stopping
round at any budget, so the claim is gone rather than rescaled.

On the ablation's 29 J: the corrected value is 22.03 ± 0.36 J/round, and the
residual 2 J against Table I's 20.05 ± 0.39 is a genuine between-session
offset — the ablation sweep ran separately, with its own idle baseline. It is
larger than either sweep's across-seed spread, so we now say in Section IV-E
that ablation energies are compared **within** the ablation sweep only, and we
make no cross-sweep energy comparison anywhere.

On gross versus net: stated once in Section III-C ("every energy figure in
this paper is net of the idle baseline unless a column is labelled Gross") and
again at the head of Section IV. Table I labels its gross column; every other
energy in the paper is net.

### 3. The guarantee is mislabelled, and $q_c$ assumes what we disclaim

Accepted in full, on both halves.

**The label.** Opacus clips and noises per-sample gradients inside local
training, so the guarantee is **example-level**, applied locally, with the
budget held per client over the whole run. It is not the client-level
(user-level) DP of [2] and [11], which noises the entire client update and
conceals participation itself. We have relabelled the abstract, Section I,
Section II-D, Section III-B, Section V-E and the conclusion, and Section III-B
now states the relation explicitly: our guarantee is weaker, and the per-round
overhead we measure is the per-sample gradient work that both mechanisms need,
so the energy result is orthogonal to the choice between them.

**The $q_c$ factor.** Also accepted: $q_c$ makes $s_{\text{total}}$ an
expected participation count, nothing in the protocol bounds the realised one,
and the reported $\varepsilon$ is therefore **optimistic rather than an upper
bound**. We had already begun auditing this and the audit agrees with the
reviewer in every run. Recomputing each client's spent $\varepsilon$ from the
participation logs, the worst client overspends in **all 29 private runs**, by
4% to 22%: HAR 0.570 / 1.122 / 2.285 / 4.250 / 9.112 against targets 0.5 / 1 /
2 / 4 / 8, CIFAR-10 0.611 / 1.162 / 2.251 / 4.356 / 8.850.

Table I now carries a *Worst $\varepsilon$* column with those values, and
Section III-B states the remedy the reviewer names: a guarantee that must hold
unconditionally has to drop $q_c$ and calibrate against
$R\,E\,\lceil n/B \rceil$, or cap participation per client. The audit script
has a `--worst-case` mode that prints the $\varepsilon$ of participation in
every round, which is the figure such a calibration would have to use.

We would rather report this than hide it: the same optimistic calibration is
common in federated DP simulations, and the audit is five lines of logging
plus a replay.

### 4. Two results carry less information than their framing

Accepted on both.

**The post-peak energy share.** The reviewer is right that with per-round
energy flat, "83% of the joules" is 83 of 100 rounds restated, and right that
the declines defining those peaks sat below our own floors. A peak is now
reported only where the peak-to-final drop clears that condition's across-seed
spread in final accuracy **and** the individual seeds agree on where the peak
falls. Under that test:

- **HAR fails at every budget.** The largest drop in a mean curve is 0.032 at
  $\varepsilon=1$ against a ±0.042 spread, and the per-seed peak rounds
  disagree (10, 33, 39 at $\varepsilon=1$; 16, 54, 65 at $\varepsilon=0.5$).
  The HAR stopping-round claim is withdrawn and Figure 1 is redrawn with no
  marker, its caption saying why.
- **CIFAR-10 passes at the two tightest budgets.** $\varepsilon=0.5$: drop
  0.026 against ±0.010, seeds peaking at rounds 8, 8, 9. $\varepsilon=1$: drop
  0.044 against ±0.014, seeds between 10 and 16. At $\varepsilon\ge2$ the
  drops fall inside their spreads and we claim nothing.

Section IV-C now leads with the peak's position — round 8 of 30 at
$\varepsilon=0.5$ — and says in the text that the energy fraction tracks the
fraction of the schedule after the peak precisely because per-round energy is
flat, so the joules price the tail of a round budget the designer chose. The
monotone peak ordering 7/11/19/24 is withdrawn: only two budgets resolve a
peak at all.

**The carbon factor.** Also accepted. The ratio is the two grid intensities
and would be identical for non-private training, and the paper now says so in
Section IV-F. The abstract and Section IV-F lead instead with the quantity
that is about privacy: the premium attributable to privacy alone, 106–148
gCO$_2$eq per 1000 runs on Algeria's grid against 5.7–7.9 g in Norway. The
geographic point is now framed as context for a federated deployment, which
cannot relocate the way a data centre can.

### 5. Reference verification and IEEE formatting

Thank you for checking these. All fixed except two, which we have marked in
the source and list below.

- **One author plus "et al." where six are permitted.** This was IEEEtran's
  default name limit, not per-entry data. An `@IEEEtranBSTCTL` entry now sets
  the limit to six names and `main.tex` cites it before the first citation;
  eight entries now show six authors before "et al."
- **Missing volume, issue and pages.** The entries that lacked them were
  preprints typed as journal articles, which is why the fields were empty.
  The seven preprints are now `@misc` entries carrying the arXiv identifier,
  which IEEEtran prints, so nothing claims a volume it does not have. Two
  IEEE entries without volume or pages are genuinely early access and are now
  marked as such.
- **Institutional authors** (NVIDIA, Ember) were being initialised to
  "N. Corporation" and "E. Ember"; both are now braced.
- **Anguita et al.** carried a spurious "and others" — the paper has exactly
  five authors — and the venue was abbreviated to "Esann". Both corrected.

Two items we could not close:

1. **Page numbers for [16] Anguita et al.** The paper is es2013-84 in the
   ESANN 2013 proceedings; we cite it by that paper number pending the
   publisher record rather than repeat the page range that circulates in
   secondary sources.
2. **The DOI for [24] Aroussi et al.** We were unable to confirm the record
   independently either. Because it carries the closest competing claim, we
   will either supply the DOI with the camera-ready or remove the citation and
   the comparison that rests on it, rather than leave an unverifiable
   reference in that position.

### 6. Page limit

The submission is now 6 pages, and we took the cut where the reviewer directed
it. Section V-A is reduced to its interpretive content, V-B no longer restates
Section IV-E, the carbon figure is dropped because Table III carries the same
six numbers, the captions are shortened, and the duplication between the
methodology's limitations and Section V-E's threats is merged. The
bibliography is tightened in form rather than shortened, since every entry is
cited.

**Section V-D, the accountant-in-the-measurement-window hazard (V-E in the
submitted version), is untouched**
and we have made it cheaper to act on: its three controls now point to the
methodology that implements them, so the finding and its remedy are stated
once each.

---

## Summary of changes to the paper

| Location | Change |
|---|---|
| Abstract, §I, §II-D, §III-B, §V-E, Conclusion | guarantee relabelled example-level, applied locally, per-client budget over the run |
| Abstract, §IV-A, §V, Conclusion | CIFAR-10 per-round overhead +60% → **+34%**; permutation-test $p$-values added |
| §IV opening | floors restated on the quantity each bounds; CIFAR power floor 6.38% → **8.9%** |
| §IV-A | retitled; the floor comparison replaced by a permutation test; the $\varepsilon=4$ contrast addressed |
| §III-C, §IV opening | gross versus net stated explicitly |
| Table I | *Worst $\varepsilon$* column added; caption shortened |
| Table II | $\varepsilon=0.5$ at the 0.50 target shown as 334 J (12.0)$\ddagger$, two of three seeds, instead of hidden |
| §IV-B | 1.17–2.10 → **1.28–1.79**; 5.8× → **5.7×**; 878 J/57.7 → **839 J/57.0** |
| §IV-C | rewritten: HAR stopping round withdrawn, CIFAR-10 resolvable at $\varepsilon\le1$, peak ordering withdrawn, energy share reframed |
| §IV-D | 1482 J / 406 J / 3.65× / 0.503 → **1076 J / 677 J / 1.59× / 0.490** |
| §IV-E | six ablation values corrected; between-sweep offset stated; accuracy floors used for accuracy claims |
| §IV-F, Table III | premium leads, ratio 21.4× → **18.8×** and framed as the grids'; intensities stated in the caption |
| §II | "83% of a run's joules" replaced by the peak's position |
| Figures | carbon figure dropped; Fig. 1 redrawn without a stopping marker; captions shortened |
| refs.bib | six-name limit, preprints as `@misc` with arXiv ids, braced institutional authors, Anguita corrected |

---

## À vérifier avant d'envoyer cette réponse

Trois points qui demandent votre décision, parce qu'ils ne se tranchent pas
depuis les données :

1. **La référence [24] (Aroussi et al., IRASET 2026).** Le relecteur n'a pas pu
   la vérifier, et moi non plus. Si vous avez le DOI ou la page IEEE Xplore,
   ajoutez-les dans `refs.bib` ; sinon il faut retirer la citation et la phrase
   de §II-B qui s'appuie dessus. Laisser une référence invérifiable à l'endroit
   du travail concurrent le plus proche est le pire des trois choix.
2. **Les pages de [16] (Anguita, ESANN 2013).** Je cite le papier es2013-84
   sans pages plutôt que de recopier l'intervalle qui circule dans les sources
   secondaires. Si vous avez les actes, mettez les pages.
3. **Les intensités carbone.** Le papier citait Ember avec des valeurs donnant
   21,4× ; `analyze.py` contient 26 / 56 / 344 / 369 / 488 / 631 gCO₂eq/kWh,
   qui donnent 18,8×. J'ai aligné le papier sur le script pour qu'il n'y ait
   qu'une source de vérité, et inscrit les six valeurs dans la légende du
   tableau. Si vos chiffres Ember sont les bons, mettez-les dans
   `GRID_GCO2_PER_KWH` et réimprimez le tableau : la conclusion géographique
   ne change pas.

Un point de forme : la réponse est écrite à la première personne du pluriel,
en votre nom. Les engagements qu'elle prend pour la version finale (le DOI,
les pages) sont les vôtres à tenir, donc relisez-les avant envoi.
