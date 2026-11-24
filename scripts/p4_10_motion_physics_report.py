"""P4-10: motion + physics report for the manager (the presentation's centre). Every result number is read from JSON at build time. -> tasks/P4_10_MOTION_PHYSICS_REPORT.md"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

RAW = "https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/"
M = "outputs/p4_10/meeting/"
B = ("25-100", "100-200", "200-300")
now = subprocess.run(["date", "+%Y-%m-%d %H:%M ET"], capture_output=True, text=True).stdout.strip()
J = lambda p: json.loads(Path(p).read_text())  # noqa: E731
G = lambda n, e="png", t=None: "[{}]({}{}{}.{})".format(t or n, RAW, M, n, e)  # noqa: E731
R = lambda p, t=None: "[{}]({}{})".format(t or p.split("/")[-1], RAW, p)  # noqa: E731
t3 = lambda x, d=3: " / ".join("{:.{}f}".format(v, d) for v in x)  # noqa: E731
ci = lambda v, d=3: "{:+.{}f} [{:+.{}f}, {:+.{}f}]".format(v[0], d, v[1], d, v[2], d)  # noqa: E731
ci3 = lambda D, d=3: " / ".join(ci(D[b], d) for b in B)  # noqa: E731

v1 = J("outputs/p4_10/verdict_R1_150K.json")
sc = J("outputs/p4_10/scorecard_source_R0_CTL_RCP.json")["families"]
s1 = J("outputs/p4_10/scorecard_R1_full.json")["families"]
p2 = J("outputs/p4_10/p2/report.json")
pt = J("outputs/p4_10/pt/read_152K.json")["reads"]
rp = J("outputs/p4_10/r1p_0d/report.json")
eb = J("outputs/p4_10/eb/report.json")
e2 = J("outputs/p4_10/e2/report.json")["strata"]
ee = J("outputs/p4_10/ee/report.json")
eeh = J("outputs/p4_10/ee/posthoc_clearance.json")
m1a, m1b, m1m = J(M + "M1_source_three_recipes_50Hz.json"), J(M + "M1_source_three_recipes_125Hz.json"), J(M + "M1m_source_three_recipes_125Hz.json")
m2, m3, m4, m5 = J(M + "M2_reciprocity_maps.json"), J(M + "M3_walk_moved_source.json"), J(M + "M4_capture_then_move.json"), J(M + "M5_energy_flow_moved_source.json")
v4o, v4s, v5 = J(M + "V4_walk_original.json"), J(M + "V4_walk_stretched.json"), J(M + "V5_energy_flow.json")
cau, rec, we = p2["causality"], p2["reciprocity"], p2["wave_equation"]["bands"]
lat = {x["edit"]: x["net_1rx_601"] for x in J("outputs/p4_9/latency.json")["rows"]}
inf, uns = v1["RS1"]["in_family_full"], v1["RS2"]["unseen_full"]
sf = lambda m, g, d=s1: t3([d["Source|" + g][m]["edited_r"][b][0] for b in B])  # noqa: E731
ig = lambda g: t3([s1["Source|" + g]["R1_150K"]["ignore_r"][b][0] for b in B])  # noqa: E731
phys = lambda g: t3([s1["Source|" + g]["R1_150K"]["m1_r"][b][0] for b in B])  # noqa: E731
rcp_rec = J("outputs/p4_9/recip_gap/RCP_150K.json")["summary"]["in_family15"]


def strip(d, f):
    o = {}
    for p, v in d["panels"].items():
        o.setdefault(p.split("|")[1].split(":")[0], []).append(v["r_at_{}Hz".format(f)])
    return o


def m1rows(d, f):
    return "; ".join("{}: {}".format(k, " / ".join("{:.2f}".format(x) for x in v)) for k, v in strip(d, f).items())


def recip_row(m, label):
    r = rec[m]["in_family15"]
    u = rec[m]["unseen16"]
    return "| {} | {} | {} | {} | {} |".format(label, t3([r[b]["normal"][0] for b in B]), t3([r[b]["swapped"][0] for b in B]), ci3({b: r[b]["swapped_minus_normal"] for b in B}), t3([u[b]["swapped"][0] for b in B]))


def we_row(b):
    r = we[b]
    return "| {} | {:.4f} | {} | {} |".format(b, r["truth"][0], " | ".join("{:.1f}".format(r[m + "|raw"]["over_truth"]) for m in ("R0_120K", "R1_150K", "R1S_150K", "R1M_150K")), " | ".join("{:.1f}".format(r[m + "|proj"]["over_truth"]) for m in ("R0_120K", "R1_150K", "R1S_150K", "R1M_150K")))


hf, uf = pt["1_static"]["HF"], pt["1_static"]["UF"]
res, e3 = pt["2_3_residual"], pt["4_e3_63hz"]
d1, dr = pt["4_d1"], pt["5_direction_raw_fd"]
ce = eb["CE1"]
pooled = J("outputs/p4_10/ee/partial_posthoc.json")["partial_controlling_clearance"]
E3T = ", ".join("{} {}".format(m.split("_")[0].replace("R1S", "R1-S").replace("R1M", "R1-M"), x) for m, x in __import__("re").findall(r"^\| (\S+) \| 63 \| .*? \| .*? \| ([\d.]+) \| \d+ \|", Path("outputs/p4_10/e3/report.md").read_text(), flags=__import__("re").M))
wpart = eeh["within_room_partial"]
zero_d = __import__("re").findall(r"^\| single \| \S+ \| .*? \| .*? \| .*? \| ([\d.]+) \|", Path("outputs/p4_10/r1p_0d/report.md").read_text(), flags=__import__("re").M)
ZD = " / ".join("{:.2f}".format(float(x)) for x in zero_d)
src = eb["families"]["Source|third"]["bands"]
val = lambda k, b="25-100": eb["families"]["Source|third"]["bands"][b]["C"][k][0]  # noqa: E731

txt = f"""# Motion and physics report for the manager (built {now}; meeting 2026-11-24 about 16:00 ET)

For the manager finishing the deck. It gathers everything on **moving the source, moving the listener, and the physics experiments**: figures, numbers, wording, what is final, and what not to say. The general presentation report (stretch, materials, decay, slide order) is {R('tasks/P4_10_PRESENTATION_REPORT.md', 'P4_10_PRESENTATION_REPORT.md')}. Every result number below is read from JSON by `scripts/p4_10_motion_physics_report.py`; times, ranks and status are text.

Triples are 25-100 / 100-200 / 200-300 Hz. r = off-lattice accuracy from the frozen scorer (spatial correlation of the sound-level map with the truth, dense receivers). "In-family (held-out)" = 15 test rooms of the training family; "unseen" = 16 rooms of families never trained on. Brackets are 95% intervals (paired bootstrap over rooms).

**Status:** everything in this report is FINAL except R1-SM (the combined model, still training; its verdict comes after the meeting and is not quoted here).

## 1. What to say, in order

1. **A model trained with one source per room cannot move its source.** R0 reads {sf('R0_120K', 'test', sc)} on moved sources in-family, below leaving the source where it was ({ig('test')}).
2. **Source-listener swaps alone help at 25-100 Hz only** (RCP {sf('RCP_150K', 'test', sc)}; leaving the source still {ig('test')}: about level at 100-200 Hz, below it at 200-300 Hz).
3. **Four source positions per room plus swaps (R1) makes the moved source usable at low frequency:** {sf('R1_150K', 'test')} in-family, {sf('R1_150K', 'unseen')} in unseen rooms (leaving the source still: {ig('unseen')}). Above 100 Hz it is far from solved.
4. **Moving the listener is cheap and works at the model's own accuracy:** {1000 * lat['geometry change']:.0f} ms per listener after a geometry change on one A5000.
5. **The collaborator's physics idea was tried and it costs accuracy at this setting.** A mean-value (wave-equation) penalty lowered the wave-equation error but cost high-frequency accuracy and made decay worse, so its pre-registered stop rule ended the run.
6. **Physics audit of the trained models:** energy before the direct sound is {min(cau['pooled'][m]['ratio_of_means'] for m in ('R0_120K','R1_150K','R1S_150K','R1M_150K')):.2f} to {max(cau['pooled'][m]['ratio_of_means'] for m in ('R0_120K','R1_150K','R1S_150K','R1M_150K')):.2f} times the truth's (the causality condition of 2 times is not met), the swapped render is far less accurate than the normal one, and the wave-equation residual is {min(we[b][m + '|raw']['over_truth'] for b in B for m in ('R0_120K', 'R1_150K', 'R1S_150K', 'R1M_150K')):.1f} to {max(we[b][m + '|raw']['over_truth'] for b in B for m in ('R0_120K', 'R1_150K', 'R1S_150K', 'R1M_150K')):.1f} times the truth's (section 6.2).

## 1b. Proposed slide order (motion first, physics after; the general report has the full deck)

| # | slide | figure | status |
|---|---|---|---|
| 1 | Moving the source: three recipes, 50 Hz | {G('M1_source_three_recipes_50Hz')} | FINAL |
| 2 | The same at 125 Hz, with the median room | {G('M1_source_three_recipes_125Hz')}, {G('M1m_source_three_recipes_125Hz')} | FINAL |
| 3 | Why: swap source and listener | {G('M2_reciprocity_maps')} | FINAL |
| 4 | Motion animation (optional) | {G('V2_source_animation', 'mp4')} | FINAL |
| 5 | The listener's walk, source moved | {G('M3_walk_moved_source')} (and {G('V4_walk_original')} for the still source) | FINAL |
| 6 | Where the energy flows, source moved | {G('M5_energy_flow_moved_source')} (and {G('V5_energy_flow')}) | FINAL |
| 7 | Capture, then move the source | {G('M4_capture_then_move')} (and {G('EB_capture_then_edit')}) | FINAL, negative |
| 8 | How physical are our fields? | {G('P2_how_physical')} | FINAL |
| 9 | Adding physics to training | {G('P1F_physics_penalty')} | FINAL, stopped |
| backup | Unseen room types | {G('V2u_source_unseen_50Hz')}, {G('V2u_source_unseen_125Hz')} | FINAL |

## 2. Moving the source

### 2.1 Three training recipes on one line (slide 1)

Figures: {G('M1_source_three_recipes_50Hz')}, {G('M1_source_three_recipes_125Hz')}; second example (median room) {G('M1m_source_three_recipes_125Hz')}; animation of V2 {G('V2_source_animation', 'mp4')}; V2 strips {G('V2_source_50Hz')}, {G('V2_source_125Hz')}; unseen backup {G('V2u_source_unseen_50Hz')}, {G('V2u_source_unseen_125Hz')}.

Accuracy r at each of six positions from the original source (start) to the third source, in the in-family (held-out) showcase room:
- **50 Hz** (room ranks {m1a['ranks']['rank']} of {m1a['ranks']['of']} at 25-100 Hz): {m1rows(m1a, 50)}
- **125 Hz** (room ranks {m1b['ranks']['rank']} of {m1b['ranks']['of']} at 100-200 Hz, so the median room is shown too): {m1rows(m1b, 125)}
- **125 Hz, median room** L6.06_W4.52_d1.08_w2.44: {m1rows(m1m, 125)}

**One line:** "At 50 Hz the one-source model loses the moved source (its accuracy falls to about zero at the far end), swaps keep part of it, and training on four source positions keeps most of it. At 125 Hz all three are much weaker."

### 2.2 The numbers behind it (all rooms, full set)

| | R0 (one source) | RCP (one source + swaps) | R1 (four sources + swaps) | leaving the source still | physics at the same edit (m1, reference) |
|---|---|---|---|---|---|
| source-family accuracy, in-family 15 | {sf('R0_120K', 'test', sc)} | {sf('RCP_150K', 'test', sc)} | {sf('R1_150K', 'test')} | {ig('test')} | {phys('test')} |
| source-family accuracy, unseen 16 | {sf('R0_120K', 'unseen', sc)} | {sf('RCP_150K', 'unseen', sc)} | {sf('R1_150K', 'unseen')} | {ig('unseen')} | {phys('unseen')} |

("Source family" pools the second and third source positions. "Leaving the source still" is the original-source truth scored against the moved truth: a reference, not a model.) **R1 by position, in-family:** second {t3(inf['second']['r'])} (ignoring the move {t3(inf['second']['ign'])}); third {t3(inf['third']['r'])} (ignoring {t3(inf['third']['ign'])}). **Unseen:** second {t3(uns['second']['r'])} (ignoring {t3(uns['second']['ign'])}); third {t3(uns['third']['r'])} (ignoring {t3(uns['third']['ign'])}).

**Pre-registered verdicts (final):** R1 fails RS1 (gap to its own original-position accuracy: second {t3([inf['second'][b][0] for b in B])}, third {t3([inf['third'][b][0] for b in B])}; bar 0.10 in every band), RS2 (one of six cells: second source, 200-300 Hz) and RS3; RC1 (no harm at the original source) holds. Approved wording: **"Moved sources are far better than before, not solved."**

### 2.3 Why, and what was tried

- **Reciprocity maps:** {G('M2_reciprocity_maps')}. Swap source and listener: a physical field does not change. In the showcase room at 50 Hz: {', '.join('{} {:.2f}'.format(k.replace('|', ' ').replace(' 50Hz', ''), [x for q, x in v.items() if q.startswith('r_at')][0]) for k, v in m2['panels'].items() if '|50Hz' in k)}; R0's swapped map is noise. It ranks {m2['ranks']['rank']} of {m2['ranks']['of']} on R1's swapped-minus-normal accuracy at 25-100 Hz ({m2['ranks']['swapped_minus_normal_r_25-100']:+.2f}; median room {m2['ranks']['median']:+.2f}).
- **Reciprocity over all rooms:** table in section 6.2. RCP's swapped accuracy (in-family) is {t3([rcp_rec['swapped'][b] for b in B])}, the same as R1's: the swaps set it.
- **Averaging the normal and swapped renders is not a candidate** (E2, all 31 rooms; it fails its pre-registered reading, which needs a gain above 100 Hz at the third source and almost no loss at the original one). It helps the second source below 200 Hz and loses at the third above 100 Hz: third source, average minus normal {ci3(e2['all 31|third']['c_minus_a'])}; second source {ci3(e2['all 31|second']['c_minus_a'])}.
- **Does accuracy follow where training sources were?** (E-E, {ee['n_positions']} positions, 8 in-family rooms.) Yes: more training sources nearby goes with higher accuracy (Spearman within 0.5 m {' / '.join('{:+.2f}'.format(ee['spearman'][b + '|n05'][0]) for b in B)}; within-room mean {' / '.join('{:+.2f}'.format(ee['within_room_mean_spearman'][b + '|n05'][0]) for b in B)}). **Read it as an association:** density and wall clearance go together (Spearman {min(eeh['density_vs_clearance_spearman'].values()):+.2f} to {max(eeh['density_vs_clearance_spearman'].values()):+.2f} depending on the radius). In two post-hoc checks that control for clearance (not pre-registered) the link stays positive in every band: pooled over rooms {min(pooled.values()):+.2f} to {max(pooled.values()):+.2f}, within room {min(wpart.values()):+.2f} to {max(wpart.values()):+.2f}. **The two checks disagree about which band carries it** (pooled: stronger above 100 Hz; within room: stronger below), so do not name a band. The p-values in the E-E report ignore that nearby positions resemble each other. An experiment that varies the training sources would be needed to call it a cause.
- **Withdrawn:** that wall clearance explains the second source (D209 corrected). The original source is as close to a wall and is the most accurate position.

## 3. Moving the listener

Figures: {G('V4_walk_original')}, {G('V4_walk_stretched')} (source at the original place); {G('M3_walk_moved_source')} (source moved to the third position).

| walk | source | R1 | R0 |
|---|---|---|---|
| original room | original place | {t3([v4o['R1_r_off_lattice_dense'][b] for b in B], 2)} | not drawn |
| room stretched 1.10 | original place | {t3([v4s['R1_r_off_lattice_dense'][b] for b in B], 2)} | not drawn |
| original room | third position | {t3([m3['R1_r_off_lattice_dense'][b] for b in B], 2)} | {t3([m3['R0_r_off_lattice_dense'][b] for b in B], 2)} |

(Accuracy over the room's off-lattice receivers at that source, 25-100 / 100-200 / 200-300 Hz; the walk is y = 1.0 m.) **One line:** "Moving the listener costs {1000 * lat['listener move']:.0f} ms per position; with the source moved, R1's field stays useful at low frequency and R0's does not." The only real-time claim is one listener: {1000 * lat['geometry change']:.0f} ms after a geometry change, {1000 * lat['listener move']:.0f} ms for a listener move (one RTX A5000; a full grid takes about {J('outputs/p4_9/latency.json')['rows'][0]['net_grid_601']:.0f} s).

## 4. Where the energy flows

Figures: {G('V5_energy_flow')} (original source, 50 and 100 Hz), {G('M5_energy_flow_moved_source')} (source at the third position, 50 Hz).

| source | frequency | model | direction agreement (mean cosine) | median angle |
|---|---|---|---|---|
| original | 50 Hz | R1 | {v5['numbers']['50Hz']['energy_flow_direction_agreement_mean_cosine']:.2f} | {v5['numbers']['50Hz']['median_angle_deg']:.0f} deg |
| original | 100 Hz | R1 | {v5['numbers']['100Hz']['energy_flow_direction_agreement_mean_cosine']:.2f} | {v5['numbers']['100Hz']['median_angle_deg']:.0f} deg |
| third | 50 Hz | R1 | {m5['numbers']['R1']['energy_flow_direction_agreement_mean_cosine']:.2f} | {m5['numbers']['R1']['median_angle_deg']:.0f} deg |
| third | 50 Hz | R0 | {m5['numbers']['R0']['energy_flow_direction_agreement_mean_cosine']:.2f} | {m5['numbers']['R0']['median_angle_deg']:.0f} deg |

At the third position the source radiates little at 50 Hz (net outward flow 0.15 m around it {m5['net_outward_flow_at_0.15m']['show_p11']:.3f}, against {m5['net_outward_flow_at_0.15m']['show_s1.000']:.3f} at the original place), so those arrows show mostly circulating energy. The showcase ranks {v5['ranks']['rank_50']} of {v5['ranks']['of']} on direction agreement at 50 Hz and {v5['ranks']['rank_100']} at 100 Hz (median room {v5['ranks']['median_value_50']:.2f} and {v5['ranks']['median_value_100']:.2f}); M5 has no rank of its own.

## 5. Capture, then move the source (E-B, M4)

Figures: {G('EB_capture_then_edit')}, {G('M4_capture_then_move')}. Model R1-M, 32 microphones, 16 unseen rooms, off-lattice, the 32 microphones left out.

- **Moving the source after capture (16 rooms, source at the third position):** zero-shot move {val('zero_edit'):.2f} / {val('zero_edit', '100-200'):.2f} / {val('zero_edit', '200-300'):.2f}; captured then moved {val('cap_edit'):.2f} / {val('cap_edit', '100-200'):.2f} / {val('cap_edit', '200-300'):.2f}; captured and not moved {val('cap_unedited'):.2f}; KRR on the same 32 microphones {val('krr'):.2f} (25-100 Hz); ignoring the move {val('ignore'):.2f}. **Capturing first makes the source move worse than doing it zero-shot.**
- **M4 shows one such room** (unseen, 50 Hz): original source zero-shot {m4['r_at_50Hz']['zero|orig']:.2f}, captured {m4['r_at_50Hz']['cap|orig']:.2f}; source moved zero-shot {m4['r_at_50Hz']['zero|third']:.2f}, captured then moved {m4['r_at_50Hz']['cap|third']:.2f}, KRR {m4['r_at_50Hz']['krr|third']:.2f}. The re-fit and E-B's stored fit give identical numbers.
- **CE1 (geometry and material edits, survival >= 0.5 at 25-100 Hz):** {'; '.join('{} {:.2f} [{:.2f}, {:.2f}], {}'.format(k.split('|')[0] + ' ' + k.split('|')[1], x['survival'], *x['ci'], 'holds' if x['holds'] else 'fails') for k, x in ce.items())}. The passes mostly reflect the capture itself: a captured model with no edit already scores {eb['families']['Wall|L+0.3']['bands']['25-100']['not_edited_on_C']['value']:.2f} (Wall) and {eb['families']['Material|all0.30']['bands']['25-100']['not_edited_on_C']['value']:.2f} (all walls 0.30) on the same scale, and interpolating the microphones scores as high on Wall and Material.

## 6. The physics experiments

### 6.1 P-1, adding physics to training (slide "We tried a physics penalty")

Figure: {G('P1F_physics_penalty')}. R1 fine-tuned with a mean-value (wave-equation) penalty for up to 10,000 steps, against the same run without it. This is an exploratory re-registration (D217); **the pre-registered arm R1-P was never launched: its gate failed (D205).** One read (+2,500 steps), 15 in-family and 16 unseen rooms, off-lattice, paired; **the pre-registered stop rule fired (HF more than 0.02 below the control with the interval below zero) and the arm was cancelled at 11:30.** (All 15 in-family rooms were lower; recomputed from the stored renders by the session that ran P-1.) The control ran to the end of its 10,000 steps.

| measure (+2,500 steps) | penalty | control | penalty minus control [95%] |
|---|---|---|---|
| HF: in-family accuracy at 200-300 Hz | {hf['pt']:.3f} | {hf['ctl']:.3f} | {ci(hf['delta'])} |
| UF: unseen accuracy 25-100 Hz | {uf['25-100']['pt']:.3f} | {uf['25-100']['ctl']:.3f} | {ci(uf['25-100']['delta'])} |
| UF: 100-200 Hz | {uf['100-200']['pt']:.3f} | {uf['100-200']['ctl']:.3f} | {ci(uf['100-200']['delta'])} |
| UF: 200-300 Hz | {uf['200-300']['pt']:.3f} | {uf['200-300']['ctl']:.3f} | {ci(uf['200-300']['delta'])} |
| wave-equation residual over the truth's, 25-100 Hz | {res['25-100']['pt_raw_over_truth']:.2f} | {res['25-100']['ctl_raw_over_truth']:.2f} | residual {ci(res['25-100']['pt_minus_ctl'], 4)} |
| wave-equation residual over the truth's, 100-200 Hz | {res['100-200']['pt_raw_over_truth']:.2f} | {res['100-200']['ctl_raw_over_truth']:.2f} | residual {ci(res['100-200']['pt_minus_ctl'], 4)} |
| wave-equation residual over the truth's, 200-300 Hz | {res['200-300']['pt_raw_over_truth']:.2f} | {res['200-300']['ctl_raw_over_truth']:.2f} | residual {ci(res['200-300']['pt_minus_ctl'], 4)} |
| band-limited residual over the truth's, 25-100 Hz | {res['25-100']['pt_bl_over_truth']:.2f} | {res['25-100']['ctl_bl_over_truth']:.2f} | (0d for R0 was 2.78) |
| 63 Hz energy common to the room, over the truth's | {e3['pt_over_truth_of_means']:.2f} | {e3['ctl_over_truth_of_means']:.2f} | level {ci(e3['pt_minus_ctl_ratio'])} |
| early-decay (EDT) error 63 / 125 / 250 Hz | {' / '.join('{:.0%}'.format(d1[o]['EDT_err']['pt']) for o in ('63', '125', '250'))} | {' / '.join('{:.0%}'.format(d1[o]['EDT_err']['ctl']) for o in ('63', '125', '250'))} | {' / '.join('{:+.1%}'.format(d1[o]['EDT_err']['delta'][0]) for o in ('63', '125', '250'))} |
| late floor 63 / 125 / 250 Hz | {' / '.join('{:.1f}'.format(d1[o]['floor']['pt']) for o in ('63', '125', '250'))} dB | {' / '.join('{:.1f}'.format(d1[o]['floor']['ctl']) for o in ('63', '125', '250'))} dB | {' / '.join('{:+.1f}'.format(d1[o]['floor']['delta'][0]) for o in ('63', '125', '250'))} dB |
| raw direction (finite differences) 25-100 Hz | {dr['25-100']['pt'][0]:.3f} | {dr['25-100']['ctl'][0]:.3f} | {ci(dr['25-100']['delta'])} |

**Say:** "We tried a physics penalty on the trained model. It pulls the field closer to the wave equation (residual {res['25-100']['ctl_raw_over_truth']:.2f} to {res['25-100']['pt_raw_over_truth']:.2f} times the truth's at 25-100 Hz, {res['100-200']['ctl_raw_over_truth']:.2f} to {res['100-200']['pt_raw_over_truth']:.2f} at 100-200 Hz, {res['200-300']['ctl_raw_over_truth']:.2f} to {res['200-300']['pt_raw_over_truth']:.2f} at 200-300 Hz) but costs accuracy at 200-300 Hz and makes the decay worse. Our stop rule ended the run after 2,500 steps. This is one arm, one read, one setting." **The manager's expectation** (written before the numbers): the residual falls (it did); static r stays within 0.01 of the control (it did not at 200-300 Hz); direction improves at 25-100 Hz (by {ci(dr['25-100']['delta'])}); EDT does not change (it got worse); whether the floor falls was open (it rose by {min(d1[o]['floor']['delta'][0] for o in ('63', '125', '250')):.1f} to {max(d1[o]['floor']['delta'][0] for o in ('63', '125', '250')):.1f} dB).

### 6.2 P-2, how physical are our fields? (zero training; final)

Figure: {G('P2_how_physical')}. 15 in-family rooms, R0 / R1 / R1-S / R1-M against the truth.

**Wave equation (mean-value residual over the truth's; 1 = as physical as the truth):**

| band | truth | raw R0 | raw R1 | raw R1-S | raw R1-M | band-limited R0 | band-limited R1 | band-limited R1-S | band-limited R1-M |
|---|---|---|---|---|---|---|---|---|---|
{we_row('25-100')}
{we_row('100-200')}
{we_row('200-300')}

**Reciprocity (accuracy of the normal and the swapped render; in-family 15; the last column is the unseen swapped accuracy):**

| model | normal | swapped | swapped minus normal | unseen, swapped |
|---|---|---|---|---|
{recip_row('R0_120K', 'R0')}
{recip_row('R1_150K', 'R1')}
{recip_row('R1S_150K', 'R1-S')}
{recip_row('R1M_150K', 'R1-M')}

**Causality (energy before the direct sound, over the direct-sound energy; condition = at least 2 times the truth's with its interval above zero):** truth {cau['pooled']['truth'][0]:.3f}; R0 {cau['pooled']['R0_120K']['level'][0]:.3f} ({cau['pooled']['R0_120K']['ratio_of_means']:.2f} times), R1 {cau['pooled']['R1_150K']['level'][0]:.3f} ({cau['pooled']['R1_150K']['ratio_of_means']:.2f}), R1-S {cau['pooled']['R1S_150K']['level'][0]:.3f} ({cau['pooled']['R1S_150K']['ratio_of_means']:.2f}), R1-M {cau['pooled']['R1M_150K']['level'][0]:.3f} ({cau['pooled']['R1M_150K']['ratio_of_means']:.2f}). **Not met by any model, so the Kramers-Kronig term is not revisited.** Limit: the truth's own level is the band limit's pre-ringing, so non-causality smaller than that cannot be seen.

### 6.3 R1-P's gate and the late floor (for questions)

- **R1-P (the pre-registered penalty) was not launched** because its gate 0d failed (D205): raw residual over band-limited {ZD} (needs 3).
- **E3:** the excess room-uniform late energy is at 63 Hz only (times the truth's: {E3T}). P-1 raised it, not lowered it ({e3['pt_over_truth_of_means']:.2f} against {e3['ctl_over_truth_of_means']:.2f}).

## 7. Do not claim

- That the source-motion problem is solved, or that the moved source is accurate above 100 Hz.
- That capture then moving the source works; that material or geometry edits transfer to unseen rooms.
- That wall clearance explains the second source, or that training-source density is the cause (E-E is an association).
- That the physics penalty helps the field or the decay; or that R1-P was launched. P-1 is one arm, one seed, one read.
- That our fields are physical: the wave-equation residual is {min(we[b][m + '|raw']['over_truth'] for b in B for m in ('R0_120K', 'R1_150K', 'R1S_150K', 'R1M_150K')):.1f} to {max(we[b][m + '|raw']['over_truth'] for b in B for m in ('R0_120K', 'R1_150K', 'R1S_150K', 'R1M_150K')):.1f} times the truth's raw, and reciprocity is poor above 100 Hz.
- Anything about R1-SM's gains; anything real-time except one listener ({1000 * lat['geometry change']:.0f} ms).

## 8a. Needed but not available today

- Source-motion numbers for the combined model (R1-SM) beyond its lite moved-source gap: the full scorecard is after the meeting.
- RCP and R0 moved-source maps in an unseen room: not drawn (V2u shows R1 and R0 only). RCP's unseen accuracy is in the table in 2.2.
- A walk (M3) in an unseen room, and a walk with RCP: not drawn.
- A 15-room rank for M5 (no per-room truth gradients at a moved source).
- A second penalty weight or a longer P-1 run: the stop rule ended the only run.

## 8. Still coming, and when

| item | state | expected |
|---|---|---|
| R1-SM (combined model) | training; newest read +10K, harm check not triggered; no verdict | +15K about 12:15, +20K about 13:15; +25K after the cut-off; verdict after the meeting |
| P-1 +5K read | both arms' checkpoints exist, unread; the manager decides if it is wanted | on request |
| E-E | ran | done (section 2.3) |

Nothing else on motion or physics is running. New files are announced in {R('tasks/P4_10_STATUS_REPORT.md', 'the status report')} and {R('outputs/p4_10/meeting/MEETING_PACK.md', 'the meeting pack')}.

## 9. Where the numbers live

{R('outputs/p4_10/verdict_R1_150K.md', 'R1 verdict')} · {R('outputs/p4_10/scorecard_source_R0_CTL_RCP.md', 'source scorecard R0 / CTL / RCP')} · {R('outputs/p4_10/scorecard_R1_full.md', 'R1 scorecard')} · {R('outputs/p4_10/p2/report.md', 'P-2')} · {R('outputs/p4_10/pt/read_152K.md', 'P-1 read')} · {R('outputs/p4_10/eb/report.md', 'E-B')} · {R('outputs/p4_10/e2/report.md', 'E2')} · {R('outputs/p4_10/ee/report.md', 'E-E')} · {R('outputs/p4_10/r1p_0d/report.md', 'gate 0d')} · {R('tasks/P4_10_PHYSICS_ADDON.md', 'physics add-on')} · {R('tasks/P4_10_MOTION_ADDON.md', 'motion add-on')} · {R('outputs/p4_10/ee/posthoc_clearance.json', 'E-E post-hoc')} · figure sidecars are beside each PNG as `.json`.
"""
Path("tasks/P4_10_MOTION_PHYSICS_REPORT.md").write_text(txt)
print(len(txt.split()), "words")
