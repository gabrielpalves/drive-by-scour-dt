"""Monte-Carlo the NEW A00 track/crack sampling logic (Python mirror) and check
it hits the deep-research targets."""
import numpy as np
rng = np.random.default_rng(1)

L_bridge, num_spans, app, after = 60.0, 3, 30.0, 30.0
track_win = app + L_bridge + after                    # 120 m
hang_rate, ball_rate = 3.0, 1.2
hang_foul_mult, ball_trans_mult, ball_trans_margin = 3.0, 3.0, 20.0
hang_p_trans, hang_trans_margin = 0.6, 15.0
patch_len = (5.0, 20.0)
abut = [app, app + L_bridge]

N = 20000
n_p, n_g, in_foul_hits, foul_len, near_ab_hits = [], [], 0, [], 0
tot_g = 0
for _ in range(N):
    np_ = rng.poisson(ball_rate * track_win / 100)
    P = []
    for _ in range(np_):
        plen = patch_len[0] + (patch_len[1]-patch_len[0])*rng.random()
        for _try in range(50):
            x0 = (track_win - plen)*rng.random()
            near = any(abs((x0+plen/2)-a) <= ball_trans_margin for a in abut)
            w = ball_trans_mult if near else 1.0
            if rng.random() <= w/ball_trans_mult: break
        P.append((x0, x0+plen))
        foul_len.append(plen)
        if any(abs((x0+plen/2)-a) <= ball_trans_margin for a in abut): near_ab_hits += 1
    ng_ = rng.poisson(hang_rate * track_win / 100)
    for _ in range(ng_):
        for _try in range(50):
            if rng.random() < hang_p_trans:
                tx = app + (L_bridge if rng.random() < 0.5 else 0.0)
                gx = tx - hang_trans_margin + 2*hang_trans_margin*rng.random()
            else:
                gx = rng.random()*track_win
            gx = max(gx, 0.0)
            inf = any(a <= gx <= b for a, b in P)
            w = hang_foul_mult if inf else 1/hang_foul_mult
            if rng.random() <= w/hang_foul_mult: break
        tot_g += 1
        if any(a <= gx <= b for a, b in P): in_foul_hits += 1
    n_p.append(np_); n_g.append(ng_)

n_p, n_g = np.array(n_p), np.array(n_g)
print(f"window = {track_win:.0f} m  (rates are per 100 m, scaled by {track_win/100:.2f})")
print(f"\nBALLAST patches/window: mean {n_p.mean():.2f} (target {ball_rate*track_win/100:.2f})"
      f"  P(0) = {(n_p==0).mean():.3f}  [report P(0)@100m = e^-1.2 = 0.30]")
print(f"  mean patch length {np.mean(foul_len):.1f} m (target ~12.5)")
print(f"  fouled fraction of window ~ {n_p.mean()*np.mean(foul_len)/track_win*100:.1f}% "
      f"[report: 10-20% of route length]")
print(f"  patches centred within {ball_trans_margin:.0f} m of an abutment: "
      f"{near_ab_hits/max(n_p.sum(),1)*100:.0f}%")
print(f"\nHANGING groups/window: mean {n_g.mean():.2f} (target {hang_rate*track_win/100:.2f})"
      f"  P(0) = {(n_g==0).mean():.3f}  [report lambda 2-3 @100m -> P(0)=0.05-0.14]")
mean_sleepers = n_g.mean()*3.0
print(f"  => impactfully unsupported sleepers ~ {mean_sleepers:.1f} of "
      f"{track_win/0.6:.0f} = {mean_sleepers/(track_win/0.6)*100:.1f}%  "
      f"[report target 5-10%]")
print(f"  groups landing INSIDE a fouled patch: {in_foul_hits/max(tot_g,1)*100:.0f}% "
      f"(fouled = ~{n_p.mean()*np.mean(foul_len)/track_win*100:.0f}% of length -> strong enrichment)")

# ---- crack placement ----
hog_ratio, hog_margin = 4.0, 0.175
supp = np.linspace(0, L_bridge, num_spans+1)
ints = supp[1:-1]; span = L_bridge/num_spans
locs, ishog = [], []
for _ in range(N):
    if len(ints) and rng.random() < hog_ratio/(hog_ratio+1):
        s = ints[rng.integers(len(ints))]; h = True
    else:
        k = rng.integers(num_spans); s = (supp[k]+supp[k+1])/2; h = False
    c = s + (2*rng.random()-1)*hog_margin*span
    locs.append(min(max(c, 0.10*L_bridge), 0.90*L_bridge)); ishog.append(h)
locs = np.array(locs); ishog = np.array(ishog)
print(f"\nCRACK: hogging share = {ishog.mean()*100:.0f}% (target {hog_ratio/(hog_ratio+1)*100:.0f}%"
      f" = {hog_ratio:.0f}:1)")
print(f"  internal supports at {np.round(ints,1)}, mid-spans at "
      f"{np.round([(supp[k]+supp[k+1])/2 for k in range(num_spans)],1)}")
d_int = np.min(np.abs(locs[:,None]-ints[None,:]), axis=1)
print(f"  {(d_int <= hog_margin*span).mean()*100:.0f}% of cracks land within "
      f"+-{hog_margin*span:.1f} m ({hog_margin*100:.1f}% of span) of a PIER")
