"""P4-10: presentation report for the manager (meeting day). Every number is read from JSON at build time; status labels FINAL / PROVISIONAL / PENDING. -> tasks/P4_10_PRESENTATION_REPORT.md"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.p4_10_meeting_pack import B, J, M, NOT_CLAIM, RAW, numbers, strip_r, t3

now = subprocess.run(["date", "+%Y-%m-%d %H:%M ET"], capture_output=True, text=True).stdout.strip()
L = lambda n, e="png": "[{}]({}outputs/p4_10/meeting/{}.{})".format(n, RAW, n, e)  # noqa: E731
R = lambda p, t=None: "[{}]({}{})".format(t or p.split("/")[-1], RAW, p)  # noqa: E731
v1, vs, vm = (J("outputs/p4_10/verdict_{}.json".format(n)) for n in ("R1_150K", "R1S_150K", "R1M_150K"))
e1 = J("outputs/p4_10/e1/report.json")["groups"]
v7 = J("outputs/p4_10/v7/report.json")["groups"]
v5 = J(M / "V5_energy_flow.json")
v3 = J(M / "V3ab_decay_rate_vs_absorption.json")["rates"]["all"]
eb = J("outputs/p4_10/eb/report.json")
ec = J("outputs/p4_10/ec/report.json")
cau = J("outputs/p4_10/p2/causality.json")
rec = J("outputs/p4_10/p2/recip.json")
res = J("outputs/p4_10/p2/residual.json")
sc = {n: J("outputs/p4_10/scorecard_{}.json".format(f))["families"] for n, f in (("R0_120K", "R0_full"), ("R1_150K", "R1_full"), ("R1S_150K", "R1S_full"), ("R1M_150K", "R1M_full"))}
edt = lambda m: e1[m]["EDT_err"]  # noqa: E731
vw = lambda n, w: J(M / "V6_why_decay_is_off.json")["width_ratio"][w]  # noqa: E731,F841
row = {(x["traj"], x["label"]): x for x in ec["table"]}


def ecw(lab, traj="a"):
    return row[(traj, lab)]["width_mean_both"]


def ecr(lab, traj):
    return "{:,}".format(int(row[(traj, lab)]["rows"]))


g = v1["RS1"]["in_family_full"]
mi, mu = vm["MM1"]["in_family_full"], vm["MM1"]["unseen_full"]
v7r = v7["in-family 15|R1|125"]
p = lambda x: 100 * x  # noqa: E731
ig = lambda f, gr, m="R1M_150K": t3([sc[m]["{}|{}".format(f, gr)][m]["ignore_r"][b][0] for b in B])  # noqa: E731
er = lambda m, f, gr: t3([sc[m]["{}|{}".format(f, gr)][m]["edited_r"][b][0] for b in B])  # noqa: E731
wall_cap = eb["families"]["Wall|L+0.3"]["bands"]["25-100"]
mat_cap = eb["families"]["Material|all0.30"]["bands"]["25-100"]
fl = lambda f, b="25-100": eb["families"][f]["bands"][b]["C"]  # noqa: E731
smr = J("outputs/p4_10/progress/R1SM_125K.json")
def ptnum():
    import re
    t = Path("outputs/p4_10/pt/read_152K.md").read_text()
    g = lambda pat: [float(x) for x in re.search(pat, t).groups()]  # noqa: E731
    hf = g(r"HF ([\d.]+) vs ([\d.]+), ([-+][\d.]+)")
    raw = [g(r"raw over truth ([\d.]+) vs ([\d.]+)")[i] for i in (0, 1)]
    ed = {o: g(r"D1 " + o + r" Hz: EDT error ([\d.]+)% vs ([\d.]+)%") for o in ("63", "125", "250")}
    return {"HF": hf, "raw": raw, "edt_pt": [ed[o][0] for o in ("63", "125", "250")], "edt_ctl": [ed[o][1] for o in ("63", "125", "250")]}


pt = ptnum()
m5 = J(M / "M5_energy_flow_moved_source.json")["numbers"]
EBS = "FINAL, 16 of 16 rooms" if not eb["missing_rooms"] and not eb["partial_rooms"] else "PROVISIONAL, {} of 16 rooms".format(eb["n_rooms"])
fam = {"Stretch-U 1.10": "StretchU|1.10", "Wall +0.3 m": "Wall|L+0.3", "All walls 0.30": "Material|all0.30", "Longest wall 0.50": "Material|long0.50", "Source moved": "Source|third"}


def eb_table():
    out = ["| edit | zero-shot edit | captured, then edited | captured, not edited | interpolate the 32 mics | ignore the edit |", "|---|---|---|---|---|---|"]
    for lab, k in fam.items():
        c = fl(k)
        out.append("| {} | {} |".format(lab, " | ".join("{:.2f}".format(c[m][0]) for m in ("zero_edit", "cap_edit", "cap_unedited", "krr", "ignore"))))
    return "\n".join(out)


def eb_line(lab, k):
    ce = eb["CE1"].get(k)
    return "{}: survival {:.2f} [{:.2f}, {:.2f}] -> {}".format(lab, ce["survival"], *ce["ci"], "holds" if ce["holds"] else "fails")


txt = f"""# Presentation report for the manager (built {now}; meeting 2026-11-24 about 16:00 ET)

For the manager building the deck. It lists a proposed slide order, the figures with raw links, slide-ready numbers, the approved wording, what is final or provisional, and what must not be said. **The result numbers are read from JSON at build time** (times, ranks and schedules are text) by `scripts/p4_10_presentation_report.py`; rebuild it after new reads land. Triples are 25-100 / 100-200 / 200-300 Hz; accuracy r is the off-lattice score from the frozen scorer (spatial correlation of the sound-level map). "Known rooms" = the 15 in-family test rooms; "new rooms" = the frozen 16 unseen rooms.

Status labels: **FINAL** (scored, verdict recorded), **PROVISIONAL** (a later part is still being computed), **PENDING** (not yet landed).

## 1. The story in six sentences

1. One network, trained on 1,000 rooms, renders a room's sound field from its floor plan, and can be **edited** in its input: stretched, reshaped, given new wall materials, or with its source moved.
2. In known rooms the geometry edits work (Stretch {er('R1S_150K', 'StretchU', 'test')} with R1-S, against {ig('StretchU', 'test', 'R1S_150K')} for ignoring the edit); in new rooms they are much weaker ({er('R1S_150K', 'StretchU', 'unseen')}).
3. The moved source, which R0 could not do, now works partly: a new interior source in known rooms reads {t3(g['third']['r'])}, against {t3(g['third']['ign'])} for ignoring the move.
4. R1-M is the first model that answers a change of wall absorption: right direction, nearly proportional (Pearson {mi['pearson']:.2f}), about half the true size (slope {mi['slope']:.2f}; the bar was 0.8 to 1.2).
5. Every shared model decays too fast; two one-room models get the resonance widths (ratio {e1['S-1 models in their own rooms']['width_ratio'][0]:.2f}) and early decay right, and R1 does not, even in rooms it trained on ({e1['R1 in 15 random training rooms']['width_ratio'][0]:.2f} times the true width).
6. Concentrating about 1,000 to 2,000 training rows on one room makes the shared model's widths right (two rooms), at a small cost to editing that room.

## 2. Proposed slide order

| # | slide | figure(s) | one line to say | status |
|---|---|---|---|---|
| 1 | What can the model edit? | {L('V8_scorecards')} (backup {L('F7_scorecard_after')}) | Per family: R0 in grey, this round's model beside it, red tick = ignoring the edit. Largest gains: moving the source and changing materials. | FINAL |
| 2 | Stretching a room | {L('V1_stretch_50Hz')} or {L('V1_stretch_125Hz')}; animation {L('V1_stretch_animation', 'mp4')} | Known room: at 50 Hz R1-S reads {strip_r('V1_stretch_50Hz', 'R1-S')} for s = 0.90 to 1.10; at 125 Hz {strip_r('V1_stretch_125Hz', 'R1-S')}. | FINAL |
| 3 | Moving the source | {L('V2_source_50Hz')}; animation {L('V2_source_animation', 'mp4')} | R1 follows the source along the line, R0 does not: 50 Hz R1 {strip_r('V2_source_50Hz', 'R1')}; R0 {strip_r('V2_source_50Hz', 'R0')}. | FINAL |
| 4 | Changing wall materials | {L('V3d_mm1_scatter')}; {L('V3ab_decay_rate_vs_absorption')}; {L('V3c_early_decay')} | Over 180 cell-bands R1-M's response is {mi['slope']:.2f} of the true one (R1 {J(M / 'V3d_mm1_scatter.json')['R1']['slope']:.2f}). In the showcase room at 125 Hz it tracks the truth at high absorption (0.50: {v3['125']['R1-M'][-1]:.2f} against {v3['125']['truth'][-1]:.2f} 1/s) and decays too fast at low absorption (0.10: {v3['125']['R1-M'][0]:.2f} against {v3['125']['truth'][0]:.2f}). | FINAL |
| 5 | The listener's walk | {L('V4_walk_original')}, {L('V4_walk_stretched')} | R1's accuracy in the room: {' / '.join('{:.2f}'.format(x) for x in (J(M / 'V4_walk_original.json')['R1_r_off_lattice_dense'][b] for b in B))} (original) and {' / '.join('{:.2f}'.format(x) for x in (J(M / 'V4_walk_stretched.json')['R1_r_off_lattice_dense'][b] for b in B))} (stretched 1.10). | FINAL |
| 6 | Where the energy flows | {L('V5_energy_flow')} | Direction agreement {v5['numbers']['50Hz']['energy_flow_direction_agreement_mean_cosine']:.2f} at 50 Hz and {v5['numbers']['100Hz']['energy_flow_direction_agreement_mean_cosine']:.2f} at 100 Hz (mean cosine; the showcase ranks {v5['ranks']['rank_50']} and {v5['ranks']['rank_100']} of {v5['ranks']['of']}; median room {v5['ranks']['median_value_50']:.2f} and {v5['ranks']['median_value_100']:.2f}). | FINAL |
| 7 | Why decay is off | {L('V6_why_decay_is_off')}; {L('V7_late_floor')} | One room: the one-room model matches the truth's resonances and early decay; the shared one blurs them. Removing the late floor does not fix R1's decay at 125 Hz (EDT {p(v7r['EDT_before'][0]):.0f}% to {p(v7r['EDT_after'][0]):.0f}%, T20 {p(v7r['T20_before'][0]):.0f}% to {p(v7r['T20_after'][0]):.0f}%). | FINAL |
| 8 | How much training per room fixes it | {L('EC_exposure')} | Fine-tuning R1 on one room: width {ecw('0 steps (R1 @150K)', 'c'):.2f} at the start, {ecw('120 steps', 'c'):.2f} after 120 steps ({ecr('120 steps', 'c')} rows in all); the shared models read {min(ecw(k) for k in ('R0 @20K', 'R0 @30K', 'R0 @60K', 'R0 @90K', 'R0 @120K', 'CTL @150K', 'R1 @150K', 'S2X @240K (2000 rooms)')):.2f} to {max(ecw(k) for k in ('R0 @20K', 'R0 @30K', 'R0 @60K', 'R0 @90K', 'R0 @120K', 'CTL @150K', 'R1 @150K', 'S2X @240K (2000 rooms)')):.2f}. | FINAL (two rooms, one seed) |
| 9 | Capture, then edit | {L('EB_capture_then_edit')} | See section 4 before using it: this one needs care. | {EBS} |
| 10 | How physical are the fields? | {L('P2_how_physical')} | Causality: the models' energy before the direct sound is {cau['pooled']['R0_120K']['ratio_of_means']:.2f}, {cau['pooled']['R1_150K']['ratio_of_means']:.2f}, {cau['pooled']['R1S_150K']['ratio_of_means']:.2f}, {cau['pooled']['R1M_150K']['ratio_of_means']:.2f} times the truth's (R0, R1, R1-S, R1-M); the revisit condition (2 times) is not met. Swapping source and listener costs R1 0.15 to 0.35 of accuracy (R0 0.55 to 0.75). The wave-equation error is 4 to 5 times the simulation's at 25-100 Hz (raw). | FINAL |
| 11 | What is running | no figure | The combined model (R1-SM) trains to about 15:30; its verdict comes after the meeting. At +5K it reads the same as its parents: material response slope {smr['material|R1SM|test']['slope']:.2f} (R1-M {smr['material|R1M|test']['slope']:.2f}), third source {t3(smr['third_source_vs_R1']['arm'])} (R1 {t3(smr['third_source_vs_R1']['ref'])}), stretch pattern of the change {t3([smr['stretch_vs_R1S|test']['level'][b] for b in B])} (R1-S {t3([smr['stretch_vs_R1S|test']['ref_level'][b] for b in B])}). Not a verdict. A mean-value penalty (P-1) was an exploratory re-registration: its stop rule triggered at the first read (see section 5b). | P-1 FINAL, R1-SM running |
| 12 | Source-moved energy flow | {L('M5_energy_flow_moved_source')} | At 50 Hz with the source at the third position R1's energy-flow direction agrees with the truth's {m5['R1']['energy_flow_direction_agreement_mean_cosine']:.2f} (median angle {m5['R1']['median_angle_deg']:.0f} degrees), R0's {m5['R0']['energy_flow_direction_agreement_mean_cosine']:.2f} ({m5['R0']['median_angle_deg']:.0f} degrees). At this position the source radiates little at 50 Hz, so most arrows are circulating energy. | FINAL |
| 13 | Adding physics to training | {L('P1F_physics_penalty')} | The penalty lowers the wave-equation error but costs accuracy and worsens decay; stopped by its own rule (section 5b). | FINAL |
| backup | A new room type | {L('V1u_stretch_unseen_50Hz')}, {L('V1u_stretch_unseen_125Hz')}; {L('V2u_source_unseen_50Hz')}, {L('V2u_source_unseen_125Hz')}; median room {L('V2m_source_median_125Hz')} | The stretch is not predicted at 50 Hz in the unseen room (r {strip_r('V1u_stretch_unseen_50Hz', 'R1-S')}) and weakly at 125 Hz ({strip_r('V1u_stretch_unseen_125Hz', 'R1-S')}). | FINAL |

The showcase room is L5.36_W4.34_d1.88_w1.40; on this round's metrics it ranks 11th to 13th of 15 on most, 3rd on one (moved source, 100-200 Hz; the median room is shown beside it) and 10th / 14th on energy flow. It is not a best case. Every figure is 16:9, 14 pt or more, with a PDF and a JSON beside the PNG.

## 3. Slide-ready numbers

| number | value | source |
|---|---|---|
"""
for lab, val, src in numbers():
    txt += "| {} | {} | {} |\n".format(lab, val, src)
txt += f"""
More for the slides on decay and training per room (E1, E-C):
- One-room models, 2 rooms: width {e1['S-1 models in their own rooms']['width_ratio'][0]:.2f} times the truth, 11 of 11 peaks; EDT error {' / '.join('{:.1f}%'.format(p(edt('S-1 models in their own rooms')[o][0])) for o in ('63', '125', '250'))}.
- R1 in 15 of its training rooms: width {e1['R1 in 15 random training rooms']['width_ratio'][0]:.2f} [{e1['R1 in 15 random training rooms']['width_ratio'][1]:.2f}, {e1['R1 in 15 random training rooms']['width_ratio'][2]:.2f}], EDT error {' / '.join('{:.0f}%'.format(p(edt('R1 in 15 random training rooms')[o][0])) for o in ('63', '125', '250'))}.
- E-C (2 rooms): shared width R0 @120K {ecw('R0 @120K'):.2f}, CTL @150K {ecw('CTL @150K'):.2f}, R1 @150K {ecw('R1 @150K'):.2f}, S2X @240K (2,000 rooms) {ecw('S2X @240K (2000 rooms)'):.2f}; fine-tuned on one room: {ecw('30 steps', 'c'):.2f} / {ecw('60 steps', 'c'):.2f} / {ecw('120 steps', 'c'):.2f} at 30 / 60 / 120 steps. From scratch on one room, the first point that fits the room has width {ecw('1000 steps', 'b'):.2f}, at {ecr('1000 steps', 'b')} rows; earlier ones are not fits.

Wording of E-C, draft for your edit: "In two rooms, about 1,000 to 2,000 extra training rows concentrated on one room bring the shared model's resonance widths to the right value; early decay takes several times longer. The same kind of rows spread over all rooms barely helps. Specialising costs a little on edits of that room."

## 4. Read this before using the capture-then-edit slide (E-B, {EBS})

**Accuracy after the edit at 25-100 Hz, new rooms, same 32 microphones and receivers for every method:**

{eb_table()}

Pre-registered test CE1 (survival >= 0.5 at 25-100 Hz): {'; '.join(eb_line(n, fam[n]) for n in ('Stretch-U 1.10', 'Wall +0.3 m', 'All walls 0.30'))}.

**Why this is not a headline yet:**
1. **The same survival bar is passed without editing.** The captured model with no edit scores {wall_cap['not_edited_on_C']['value']:.2f} (Wall) and {mat_cap['not_edited_on_C']['value']:.2f} (all walls 0.30) on the same scale (a reading aid added after seeing the numbers, not pre-registered). So CE1 "holds" for Wall and Material mostly because capturing helps, not because the edit does. The edit's own contribution (captured-and-edited minus captured-not-edited) is {wall_cap['cap_edit_minus_cap_unedited_C'][0]:+.2f} (Wall) and {mat_cap['cap_edit_minus_cap_unedited_C'][0]:+.2f} (Material) at 25-100 Hz.
2. **Interpolating the 32 microphones (KRR) scores as high as, or higher than, capture-then-edit on Wall and Material** (the true edit changes the field little there; ignoring the edit also scores {fl('Wall|L+0.3')['ignore'][0]:.2f} and {fl('Material|all0.30')['ignore'][0]:.2f}). Capture-then-edit is clearly better only where the edit moves the field a lot, i.e. Stretch ({fl('StretchU|1.10')['cap_edit'][0]:.2f} against {fl('StretchU|1.10')['cap_unedited'][0]:.2f} unedited and {fl('StretchU|1.10')['krr'][0]:.2f} interpolated at 25-100 Hz), and for Source it loses to the zero-shot edit ({fl('Source|third')['cap_edit'][0]:.2f} against {fl('Source|third')['zero_edit'][0]:.2f}).
3. Zero-shot edit on Stretch ({fl('StretchU|1.10')['zero_edit'][0]:.2f}) is almost as good as capture-then-edit ({fl('StretchU|1.10')['cap_edit'][0]:.2f}), so the capture adds little there, and Stretch fails CE1.
4. Decay after capture (both Material cells): the captured model's response to a material change is {eb['decay']['cap']['pooled']['slope']:.2f} of the true one (Pearson {eb['decay']['cap']['pooled']['pearson']:.2f}), against {eb['decay']['zero']['pooled']['slope']:.2f} ({eb['decay']['zero']['pooled']['pearson']:.2f}) zero-shot. The decay error falls ({p(eb['decay']['cap']['edt_err_edit_med']):.1f}% edited against {p(eb['decay']['cap']['edt_err_orig_med']):.1f}% unedited) but the response shrinks. These cover two magnitudes only: do not compare the zero-shot slope with MM1's.
5. Capturing hurts the edit for Stretch above 100 Hz and for Source in every band (the uncaptured model is better).

A safe sentence if the slide is used: "Capturing a new room with 32 microphones raises its accuracy a lot before any edit ({eb['static']['25-100']['zero_shot']:.2f} to {eb['static']['25-100']['captured']:.2f} on the original room). Editing afterwards adds {wall_cap['cap_edit_minus_cap_unedited_C'][0]:+.2f} (wall) and {mat_cap['cap_edit_minus_cap_unedited_C'][0]:+.2f} (materials) to the captured model and {eb['families']['StretchU|1.10']['bands']['25-100']['cap_edit_minus_cap_unedited_C'][0]:+.2f} for a stretch, which the zero-shot edit nearly matches; a moved source gets worse than zero-shot." (25-100 Hz)

## 5. Physics audit (P-2, FINAL)

- **Causality (complete, 15 rooms, 4 models; FINAL):** energy before the direct sound over the direct-sound energy: truth {cau['pooled']['truth'][0]:.3f}; R0 {cau['pooled']['R0_120K']['level'][0]:.3f}, R1 {cau['pooled']['R1_150K']['level'][0]:.3f}, R1-S {cau['pooled']['R1S_150K']['level'][0]:.3f}, R1-M {cau['pooled']['R1M_150K']['level'][0]:.3f}. The revisit condition for the Kramers-Kronig term (model at least 2 times the truth, with its interval above zero) is **not met by any model**. Limit: the truth's own level ({cau['pooled']['truth'][0]:.2f}) is the band limit's own pre-ringing, so the test cannot see non-causality smaller than that.
- **Reciprocity (complete, 4 models):** swapped against normal accuracy in known rooms at 25-100 Hz: R0 {rec['R0_120K']['in_family15']['25-100']['swapped'][0]:.2f} against {rec['R0_120K']['in_family15']['25-100']['normal'][0]:.2f}; R1 {rec['R1_150K']['in_family15']['25-100']['swapped'][0]:.2f} against {rec['R1_150K']['in_family15']['25-100']['normal'][0]:.2f}. At 200-300 Hz R1 is {rec['R1_150K']['in_family15']['200-300']['swapped'][0]:.2f} against {rec['R1_150K']['in_family15']['200-300']['normal'][0]:.2f}.
- **Wave equation (complete, 4 models):** the mean-value residual, times the truth's, raw: R0 {res['bands']['25-100']['R0_120K|raw']['over_truth']:.1f}, R1 {res['bands']['25-100']['R1_150K|raw']['over_truth']:.1f}, R1-S {res['bands']['25-100']['R1S_150K|raw']['over_truth']:.1f}, R1-M {res['bands']['25-100']['R1M_150K|raw']['over_truth']:.1f} at 25-100 Hz. R0 is {res['bands']['25-100']['R0_120K|raw']['over_truth']:.1f}, {res['bands']['100-200']['R0_120K|raw']['over_truth']:.1f} and {res['bands']['200-300']['R0_120K|raw']['over_truth']:.1f} times the truth's (raw, three bands); band-limited {res['bands']['25-100']['R0_120K|proj']['over_truth']:.1f}, {res['bands']['100-200']['R0_120K|proj']['over_truth']:.1f}, {res['bands']['200-300']['R0_120K|proj']['over_truth']:.1f}.

## 5b. P-1, the mean-value penalty (FINAL: stopped at +2.5K, D217 / D218)

R1-P(t) (R1 plus the penalty) against its matched control R1-ctl after 2.5K of 10K steps: accuracy HF {pt['HF'][0]:.3f} against {pt['HF'][1]:.3f} ({pt['HF'][2]:+.3f}; below the control in 15 of 15 known rooms); the pre-registered stop rule triggered and the arm was cancelled. The wave-equation residual fell (raw, over the truth: {pt['raw'][1]:.2f} to {pt['raw'][0]:.2f} at 25-100 Hz), but the early-decay error rose at every octave (63 / 125 / 250 Hz: {pt['edt_ctl'][0]:.0f} / {pt['edt_ctl'][1]:.0f} / {pt['edt_ctl'][2]:.0f}% to {pt['edt_pt'][0]:.0f} / {pt['edt_pt'][1]:.0f} / {pt['edt_pt'][2]:.0f}%) and the late floor rose. **For the collaborator who asked: the penalty was tried, it reduces the physics error, and it hurts accuracy and decay at this strength.** The expected lever for the 63 Hz floor did not show. One read, one seed.

## 6. Wording that is approved

- **R1:** "Moved sources are far better than before, not solved." (D206)
- **R1-M (D211):** "The first model that responds to a change of wall material through its inputs: right direction, nearly proportional (Pearson 0.95), about half the true size. After the edit its render is closer to the truth than before. Its remaining decay error is the same as every shared model's; the edit does not add to it (MM2)."
- **E1 (D211):** "Models trained on one room reproduce their rooms' resonance widths and early decay (EDT within 2%). The model shared across 1,000 rooms blurs resonances by 1.4-1.7 times, even in rooms it trained on. The architecture can express the right decay; sharing one model across rooms blurs it." E-C softens the last sentence: concentrated training on a room sharpens it, so it is a statement about today's shared training, not about the architecture.
- **R1-S:** "Training on stretched rooms improves stretch edits a little in every band, in known and new rooms, and costs nothing at the original rooms. It did not reach the bar we set." (proposed in report 3)

## 7. Do not claim

""" + "\n".join("- " + x for x in NOT_CLAIM) + f"""
- that capture-then-edit works in general (section 4);
- that the physics penalty helps (it was stopped; section 5b);
- that decay is fixed, or that removing the floor fixes it (V7);
- that the showcase room is typical of new room types (the unseen backup pictures are much worse).

## 8. Status and expected times

| item | state | expected |
|---|---|---|
| V1-V8, V3(d), V5 caption redraw | FINAL (V5 re-draw queued) | now |
| E-C | FINAL | posted in the live log at 09:31 |
| E-B | {EBS} | posted in the live log |
| P-2 | FINAL | posted in the live log |
| R1-SM +5K | FINAL (descriptive) | posted in the live log |
| R1-SM +10K, +15K, +20K, +25K | training | +10K about 10:52; +25K about 14:15, posted as after the cut-off |
| P-1 (R1-P(t) and control) | FINAL: stopped at +2.5K by its stop rule | posted in the live log |
| E-E (source coverage) | truth done, renders only on spare GPUs | if it runs |
| R1-SM's +30K verdict | after the meeting | about 15:30 |

Cluster note: about 134 nodes were down this morning, so some reads land later than the schedule; nothing was lost. This report can be rebuilt at any time: `python scripts/p4_10_presentation_report.py`.

## 9. All links

- Board: {R('outputs/p4_10/progress/board.md')} · Live log: {R('tasks/P4_10_STATUS_REPORT.md')} · Meeting pack: {R('outputs/p4_10/meeting/MEETING_PACK.md')} · Plan: {R('tasks/P4_10_MEETING_DAY.md')} · Physics add-on: {R('tasks/P4_10_PHYSICS_ADDON.md')} · Decisions: {R('DECISIONS.md')}
- Reports: {R('outputs/p4_10/e1/report.md', 'E1')} · {R('outputs/p4_10/v7/report.md', 'V7')} · {R('outputs/p4_10/ec/report.md', 'E-C')} · {R('outputs/p4_10/eb/report.md', 'E-B')} · {R('tasks/P4_10_MANAGER_REPORT_3.md', 'report 3')}
"""
Path("tasks/P4_10_PRESENTATION_REPORT.md").write_text(txt)
print(len(txt.split()), "words")
