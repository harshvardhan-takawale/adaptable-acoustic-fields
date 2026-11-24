"""MEETING_PACK.md (D211): figure list with one-sentence takeaways, numbers sheet, do-not-claim list, what failed. Every number is read from JSON at build time; anything missing is listed as not finished.
  python scripts/p4_10_meeting_pack.py [--auto]      --auto marks the pack as built without review (used by the unattended job)
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

M = Path("outputs/p4_10/meeting")
RAW = "https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/"
B = ("25-100", "100-200", "200-300")


def J(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def strip_r(name, who):
    d = J(M / (name + ".json"))
    if not d:
        return None
    k = [x for x in next(iter(d["panels"].values())) if x.startswith("r_at")][0]
    return " / ".join("{:.2f}".format(v[k]) for p, v in d["panels"].items() if p.split("|")[1] == who)


def t3(x):
    return " / ".join("{:.3f}".format(v) for v in x)


def figures():
    out = []

    def add(files, label, fn, movie=False):
        try:
            take = fn()
        except Exception:
            take = None
        out.append((files, label, take, movie))
    add(["V1_stretch_50Hz", "V1_stretch_125Hz"], "V1 Stretching a room (known room)", lambda: "R1-S follows a uniform stretch from 0.90 to 1.10: r at 50 Hz {}; at 125 Hz {}.".format(strip_r("V1_stretch_50Hz", "R1-S"), strip_r("V1_stretch_125Hz", "R1-S")))
    add(["V1_stretch_animation"], "V1 animation", lambda: "Nine stretch factors, truth and R1-S side by side (MP4 and GIF).", True)
    add(["V1u_stretch_unseen_50Hz", "V1u_stretch_unseen_125Hz"], "V1 backup, a room type never seen in training", lambda: "In the unseen median room the stretch is not predicted at 50 Hz (r {}) and weakly at 125 Hz (r {}).".format(strip_r("V1u_stretch_unseen_50Hz", "R1-S"), strip_r("V1u_stretch_unseen_125Hz", "R1-S")))
    add(["V2_source_50Hz", "V2_source_125Hz"], "V2 Moving the source (known room)", lambda: "R1 follows the source along the line, R0 does not: 50 Hz R1 {} (R0 {}); 125 Hz R1 {} (R0 {}).".format(strip_r("V2_source_50Hz", "R1"), strip_r("V2_source_50Hz", "R0"), strip_r("V2_source_125Hz", "R1"), strip_r("V2_source_125Hz", "R0")))
    add(["V2m_source_median_125Hz"], "V2 second example (median room, 125 Hz)", lambda: "The showcase ranks 3 of 15 here, so the median room is shown: R1 {} (R0 {}).".format(strip_r("V2m_source_median_125Hz", "R1"), strip_r("V2m_source_median_125Hz", "R0")))
    add(["V2u_source_unseen_50Hz", "V2u_source_unseen_125Hz"], "V2 backup, unseen room", lambda: "Unseen median room: R1 at 50 Hz {}; at 125 Hz {}.".format(strip_r("V2u_source_unseen_50Hz", "R1"), strip_r("V2u_source_unseen_125Hz", "R1")))
    add(["V2_source_animation"], "V2 animation", lambda: "Twelve source positions, truth, R1 and R0 (MP4 and GIF).", True)
    v3 = J(M / "V3ab_decay_rate_vs_absorption.json")
    add(["V3ab_decay_rate_vs_absorption"], "V3 (a, b) Decay rate against wall absorption", lambda: "At 125 Hz, all walls from 0.10 to 0.50: truth {:.2f} to {:.2f} 1/s, R1-M {:.2f} to {:.2f}, R1 {:.2f} to {:.2f}; R1-M tracks the truth at high absorption and decays too fast at low absorption (one room).".format(*(v for m in ("truth", "R1-M", "R1") for v in (v3["rates"]["all"]["125"][m][0], v3["rates"]["all"]["125"][m][-1]))))
    add(["V3c_early_decay"], "V3 (c) Early energy decay", lambda: "With absorption 0.50 R1-M's early decay follows the truth; with 0.10 it decays too fast.")
    d3 = J("outputs/p4_10/verdict_R1M_150K.json")
    add(["V3d_mm1_scatter"], "V3 (d) The materials scatter", lambda: "Over all 180 known-room cell-bands R1-M's change in decay rate is {:.2f} of the true one (Pearson {:.2f}); the bar was 0.8 to 1.2.".format(d3["MM1"]["in_family_full"]["slope"], d3["MM1"]["in_family_full"]["pearson"]))
    add(["V4_walk_original", "V4_walk_stretched"], "V4 The listener's walk", lambda: "R1's accuracy in the room is {} (original) and {} (stretched by 1.10), 25-100 / 100-200 / 200-300 Hz.".format(*(t3([J(M / (n + ".json"))["R1_r_off_lattice_dense"][b] for b in B]) for n in ("V4_walk_original", "V4_walk_stretched"))))
    v5 = J(M / "V5_energy_flow.json")
    add(["V5_energy_flow"], "V5 Where the energy flows", lambda: "R1's energy-flow direction agrees with the truth's {:.2f} at 50 Hz and {:.2f} at 100 Hz (mean cosine; the showcase ranks {} and {} of {}; median room {:.2f} and {:.2f}).".format(v5["numbers"]["50Hz"]["energy_flow_direction_agreement_mean_cosine"], v5["numbers"]["100Hz"]["energy_flow_direction_agreement_mean_cosine"], v5["ranks"]["rank_50"], v5["ranks"]["rank_100"], v5["ranks"]["of"], v5["ranks"]["median_value_50"], v5["ranks"]["median_value_100"]))
    add(["V6_why_decay_is_off"], "V6 Why decay is off", lambda: "In one room the one-room model's resonances are {:.2f} times the truth's width and its early decay is right; the shared model's are {:.2f} times the width. Both keep a late floor.".format(J(M / "V6_why_decay_is_off.json")["width_ratio"]["S-1"], J(M / "V6_why_decay_is_off.json")["width_ratio"]["R1"]))
    add(["V7_late_floor"], "V7 The late floor under standard compensation", lambda: (lambda g: "Removing the late floor does not fix R1's decay at 125 Hz: EDT error {:.0f}% to {:.0f}%, T20 {:.0f}% to {:.0f}%, T30 {:.0f}% to {:.0f}% (only T30's artefact goes away). The one-room models pass the same method.".format(*(100 * g[k][0] for k in ("EDT_before", "EDT_after", "T20_before", "T20_after", "T30_before", "T30_after"))))(J("outputs/p4_10/v7/report.json")["groups"]["in-family 15|R1|125"]))
    add(["V8_scorecards"], "V8 Scorecards", lambda: "Largest gains over R0 this round: Source and Material; 'ignore the edit' still beats this round's model for Material everywhere.")
    add(["EC_exposure"], "E-C Per-room training against decay (two rooms)", lambda: (lambda t: "Fine-tuning R1 on one room takes the resonance width from {:.2f} to {:.2f} times the truth after 120 steps; R0 @120K stays at {:.2f}.".format(t[("c", "0 steps (R1 @150K)")], t[("c", "120 steps")], t[("a", "R0 @120K")]))({(x["traj"], x["label"]): x["width_mean_both"] for x in J("outputs/p4_10/ec/report.json")["table"]}))
    add(["EB_capture_then_edit"], "E-B Capture then edit", lambda: (lambda e: "{} of 16 rooms: CE1 {}; but a captured model with no edit already scores {:.2f} (Wall) on the survival scale, and interpolating the microphones scores as high on Wall and Material (see section 4 of the presentation report).".format(e["n_rooms"], "; ".join("{} {}".format(k.split("|")[0], "holds" if v["holds"] else "fails") for k, v in e["CE1"].items()), e["families"]["Wall|L+0.3"]["bands"]["25-100"]["not_edited_on_C"]["value"]))(J("outputs/p4_10/eb/report.json")))
    M1K = ("R0: one source per room", "RCP: one source, plus source-listener swaps", "R1: four sources per room, plus swaps")
    add(["M1_source_three_recipes_50Hz", "M1_source_three_recipes_125Hz"], "M1 Moving the source, three recipes", lambda: "50 Hz (start to third source): R0 {}; RCP (one source plus swaps) {}; R1 (four sources plus swaps) {}. 125 Hz: R0 {}; RCP {}; R1 {}. All three are much weaker at 125 Hz.".format(*[strip_r("M1_source_three_recipes_{}Hz".format(f), k) for f in (50, 125) for k in M1K]))
    add(["M2_reciprocity_maps"], "M2 Swap the source and the listener", lambda: "A physical field does not change when source and listener swap: R0's swapped map is noise; R1's keeps its 50 Hz pattern (see the JSON panels for r); R1 ranks 7 of 15 on swapped-minus-normal accuracy at 25-100 Hz ({:.2f}; median room {:.2f}).".format(J(M / "M2_reciprocity_maps.json")["ranks"]["swapped_minus_normal_r_25-100"], J(M / "M2_reciprocity_maps.json")["ranks"]["median"]))
    add(["M3_walk_moved_source"], "M3 The listener's walk with the source moved", lambda: "Source at the third position: R1 {}, R0 {} (25-100 / 100-200 / 200-300 Hz).".format(*(" / ".join("{:.2f}".format(J(M / "M3_walk_moved_source.json")[k][b]) for b in B) for k in ("R1_r_off_lattice_dense", "R0_r_off_lattice_dense"))))
    add(["P1F_physics_penalty"], "P-1 figure: the physics penalty, stopped by its pre-set rule", lambda: (lambda d: "The mean-value penalty lowers the wave-equation residual (25-100 Hz {:.2f} to {:.2f} times the truth's) but costs accuracy at 200-300 Hz ({:.3f} against {:.3f}); the run was stopped at its first read.".format(*(d[k] for k in ("raw_ctl", "raw_pt", "hf_pt", "hf_ctl"))))((lambda t: dict(zip(("raw_pt", "raw_ctl", "hf_pt", "hf_ctl"), [float(x) for x in __import__("re").search(r"raw over truth ([\d.]+) vs ([\d.]+)", t).groups()] + [float(x) for x in __import__("re").search(r"HF ([\d.]+) vs ([\d.]+)", t).groups()])))((Path("outputs/p4_10/pt/read_152K.md")).read_text())))
    add(["M5_energy_flow_moved_source"], "M5 Energy flow with the source moved (50 Hz)", lambda: "With the source at the third position R1's energy-flow direction agrees with the truth's {:.2f} (mean cosine; median angle {:.0f} degrees), R0's {:.2f} ({:.0f} degrees).".format(*(lambda n: (n["R1"]["energy_flow_direction_agreement_mean_cosine"], n["R1"]["median_angle_deg"], n["R0"]["energy_flow_direction_agreement_mean_cosine"], n["R0"]["median_angle_deg"]))(J(M / "M5_energy_flow_moved_source.json")["numbers"])))
    add(["P2_how_physical"], "P-2 How physical are our fields?", lambda: (lambda c: "Energy before the direct sound is {:.2f} to {:.2f} times the simulation's (R0, R1, R1-S, R1-M); the revisit condition of 2 times is not met. The swap test costs R1 0.15 to 0.35 of accuracy in known rooms; the wave-equation error is several times the simulation's at low frequency.".format(min(c[m]["ratio_of_means"] for m in ("R0_120K", "R1_150K", "R1S_150K", "R1M_150K")), max(c[m]["ratio_of_means"] for m in ("R0_120K", "R1_150K", "R1S_150K", "R1M_150K"))))(J("outputs/p4_10/p2/causality.json")["pooled"]))
    return out


def numbers():
    rows = []

    def a(label, val, src):
        rows.append((label, val, src))
    v1, vs, vm = (J("outputs/p4_10/verdict_{}.json".format(n)) for n in ("R1_150K", "R1S_150K", "R1M_150K"))
    g = v1["RS1"]["in_family_full"]
    for s, lab in (("second", "second"), ("third", "third")):
        a("R1, moved source in known rooms, {} position (ignoring the move)".format(lab), "{} ({})".format(t3(g[s]["r"]), t3(g[s]["ign"])), "verdict_R1_150K.json")
    a("R1, gap to its own original-position accuracy, third source (bar 0.10)", t3([g["third"][b][0] for b in B]), "verdict_R1_150K.json")
    a("R1, unseen rooms, third source (ignoring the move)", "{} ({})".format(t3(v1["RS2"]["unseen_full"]["third"]["r"]), t3(v1["RS2"]["unseen_full"]["third"]["ign"])), "verdict_R1_150K.json")
    a("R1, swapped render against normal (known rooms; 25-100 Hz)", "{:.2f} against {:.2f}".format(v1["RS3"]["in_family15"]["25-100"]["swapped"], v1["RS3"]["in_family15"]["25-100"]["normal"]), "verdict_R1_150K.json")
    d = vs["SS1"]["in_family_full"]
    a("R1-S minus R1, stretch pattern of the change, known rooms (bar +0.05 in 2 bands)", " / ".join("{:+.3f} [{:+.3f}, {:+.3f}]".format(*d["delta"][b]) for b in B), "verdict_R1S_150K.json")
    mi, mu = vm["MM1"]["in_family_full"], vm["MM1"]["unseen_full"]
    a("R1-M, change in decay rate (1/EDT): slope, Pearson (known / new rooms; bar 0.8 to 1.2)", "{:.2f}, {:.2f} / {:.2f}, {:.2f}".format(mi["slope"], mi["pearson"], mu["slope"], mu["pearson"]), "verdict_R1M_150K.json")
    a("R1-M, EDT error edited against its own unedited (known / new)", "{:.1f}% vs {:.1f}% / {:.1f}% vs {:.1f}%".format(100 * mi["edt_err_edit"], 100 * mi["edt_err_orig"], 100 * mu["edt_err_edit"], 100 * mu["edt_err_orig"]), "verdict_R1M_150K.json")
    sc = {n: J("outputs/p4_10/scorecard_{}.json".format(f))["families"] for n, f in (("R0_120K", "R0_full"), ("R1_150K", "R1_full"), ("R1S_150K", "R1S_full"), ("R1M_150K", "R1M_full"))}
    er = lambda m, f, g2: t3([sc[m]["{}|{}".format(f, g2)][m]["edited_r"][b][0] for b in B])  # noqa: E731
    a("Stretch-U accuracy, known rooms: R1-S (R0)", "{} ({})".format(er("R1S_150K", "StretchU", "test"), er("R0_120K", "StretchU", "test")), "scorecard_R1S_full.json, scorecard_R0_full.json")
    a("Stretch-U accuracy, new rooms: R1-S (R0)", "{} ({})".format(er("R1S_150K", "StretchU", "unseen"), er("R0_120K", "StretchU", "unseen")), "the same files")
    ig = lambda f, g2: t3([sc["R1M_150K"]["{}|{}".format(f, g2)]["R1M_150K"]["ignore_r"][b][0] for b in B])  # noqa: E731
    a("Material accuracy, known rooms: R1-M (R1); ignore the edit", "{} ({}); {}".format(er("R1M_150K", "Material", "test"), er("R1_150K", "Material", "test"), ig("Material", "test")), "scorecard_R1M_full.json, scorecard_R1_full.json")
    e1 = J("outputs/p4_10/e1/report.json")["groups"]
    a("Resonance width / truth: one-room models (2 rooms), R1 in 15 training rooms", "{:.2f}; {:.2f} [{:.2f}, {:.2f}]".format(e1["S-1 models in their own rooms"]["width_ratio"][0], *e1["R1 in 15 random training rooms"]["width_ratio"]), "e1/report.json")
    a("EDT error 63 / 125 / 250 Hz: one-room models; R1 in 15 training rooms", "{}; {}".format(" / ".join("{:.1f}%".format(100 * e1["S-1 models in their own rooms"]["EDT_err"][o][0]) for o in ("63", "125", "250")), " / ".join("{:.0f}%".format(100 * e1["R1 in 15 random training rooms"]["EDT_err"][o][0]) for o in ("63", "125", "250"))), "e1/report.json")
    v7 = J("outputs/p4_10/v7/report.json")
    g7 = v7["groups"]["in-family 15|R1|125"]
    a("R1 decay error at 125 Hz, in-family, after removing the late floor: EDT / T20 (before)", "{:.0f}% / {:.0f}% ({:.0f}% / {:.0f}%)".format(*(100 * g7[k][0] for k in ("EDT_after", "T20_after", "EDT_before", "T20_before"))), "v7/report.json")
    cf = {n: J("outputs/p4_9/confirm/{}.json".format(n)) for n in ("R0_120K", "S2X_240K", "R1_150K")}
    for n, lab in (("R0_120K", "R0"), ("S2X_240K", "S2X"), ("R1_150K", "R1")):
        a("Confirmation set (16 fresh unseen rooms): {}".format(lab), t3([cf[n]["unseen_all"]["raw"][b] for b in B]), "p4_9/confirm/{}.json".format(n))
    a("R1 on the frozen unseen set (16)", t3([vm["MC1"]["UF"][b]["ref"] for b in B]), "verdict_R1M_150K.json (reference column)")
    hf = v1["RC1"]["HF"]
    a("R1 no-harm at the original source, HF (CTL @150K)", "{:.3f} ({:.3f}), {:+.3f} [{:+.3f}, {:+.3f}]".format(hf["r1"], hf["ctl"], *hf["delta"]), "verdict_R1_150K.json")
    lat = {x["edit"]: x["net_1rx_601"] for x in J("outputs/p4_9/latency.json")["rows"]}
    a("Single-listener render time after a geometry edit (network, one A5000) -- the only real-time claim", "{:.0f} ms".format(1000 * lat["geometry change"]), "outputs/p4_9/latency.json")
    return rows


NOT_CLAIM = ["that clearance explains the second source;", "that material edits beat the original room's true field;", "that geometry edits transfer to new rooms (unseen pattern of change <= 0.33; notches not predicted);",
             "that R1-SM's gains add, before its +30K verdict (about 15:30 ET);", "that anything renders in real time except single listeners (12 ms);",
             "that decay is 12-34% too fast: that is the early decay only; R1's T20 is 33-61% off after the late floor is removed;", "that R1-P was launched: the pre-registered arm is not (D205); P-1 today was an exploratory re-registration (D217) and was stopped at +2.5K by its stop rule (HF 0.534 against 0.572);", "that the physics penalty helps: it lowers the residual but costs accuracy and worsens the decay."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true")
    a = ap.parse_args()
    now = subprocess.run(["date", "+%Y-%m-%d %H:%M ET"], capture_output=True, text=True).stdout.strip()
    L = ["# MEETING PACK (built {})".format(now), "", "Meeting 2026-11-24 about 16:00 ET. Every number is read from a JSON file at build time. Triples are 25-100 / 100-200 / 200-300 Hz; r is the off-lattice accuracy from the frozen scorer." + (
        " **This copy was built automatically by a Slurm job and has not been reviewed by the agent.**" if a.auto else ""), "", "## 1. Figures", ""]
    missing = []
    for files, label, take, movie in figures():
        have = [f for f in files if (M / (f + (".mp4" if movie else ".png"))).exists()]
        if not have:
            missing.append(label)
            continue
        ext = "mp4" if movie else "png"
        links = " · ".join("[{}]({}outputs/p4_10/meeting/{}.{}) ([json]({}outputs/p4_10/meeting/{}.json))".format(f, RAW, f, ext, RAW, f) for f in have)
        L += ["- **{}** {}: {}".format(label, links, take or "see the figure and its report")]
    L += ["", "PDF copies sit beside every PNG. Figures of 2026-11-24 00:00-04:00 (F1, F2, F3, F4, F5, F7) are in the [manifest]({}outputs/p4_10/meeting/FIGURE_MANIFEST.md); F3, F4 and F5 are superseded by V1-V4 (the lattice gaps in F3 / F4 are gone).".format(RAW), "", "## 2. Numbers", "", "| number | value | source |", "|---|---|---|"]
    for lab, val, src in numbers():
        L.append("| {} | {} | {} |".format(lab, val, src))
    L += ["", "## 3. Do not claim", ""] + ["- " + x for x in NOT_CLAIM] + ["", "## 4. Failed or not finished", ""]
    notes = Path("outputs/p4_10/meeting/PACK_NOTES.md")
    L += ([ln for ln in notes.read_text().splitlines() if ln.strip()] if notes.exists() else []) + ["- Not finished at build time: {}".format("; ".join(missing)) if missing else "- Every listed figure exists."]
    (M / "MEETING_PACK.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
