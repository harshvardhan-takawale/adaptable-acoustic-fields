# V3 length-morphing audio demo (C2_latent_jitter + B6)

Source: 1.0-sec, fs=4096, x(t) = impulse + 0.3·sin(2π·80·t) + 0.2·sin(2π·120·t) + 0.1·sin(2π·180·t).

For each unseen L the predicted RIR at the centre receiver was produced by inverse-rfft of the saved `H_pred_all.pt`, then convolved with the source and peak-normalised. Files:

- `morph_L3.25.wav` — peak/median = 33.83
- `morph_L4.25.wav` — peak/median = 39.78
- `morph_L5.75.wav` — peak/median = 40.22

Quality caveat: qualitative, full-band LSD ~4-5 dB; demo shows smooth latent morphing, not faithful reconstruction.
