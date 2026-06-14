"""Temporary regression + feature harness (structural-only: fast, no dynamic
solve). Verifies the damage refactor is backward compatible and that the new
toggles behave. Safe to delete."""
import os, sys, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
from a01_train import a01_train
from a02_track import a02_track
from a03_bridge import a03_bridge
from a04_options import a04_options
from b43_model_geometry import b43_model_geometry
from b07_options_processing import b07_options_processing
from b01_elements_and_coordinates import b01_elements_and_coordinates
from b02_boundary_conditions import b02_boundary_conditions
from b03_beam_matrices import b03_beam_matrices
from b09_beam_frq import b09_beam_frq
from damage_config import make_damage, configure_bridge, apply_cracks, EmptyObj


def legacy_damage(scour):
    d = EmptyObj(); d.desvio = 0.0; d.DOFStiff_ROT_value = 0
    d.DOF_ChangeRate_value = scour
    return d


def build(Damage, length=40.0, num_spans=2, support_locs=None):
    Beam = EmptyObj(); Beam.Prop = EmptyObj()
    Beam.Prop.E_mod = 1; Beam.Prop.n_mod = 100
    Beam = configure_bridge(Beam, length=length, num_spans=num_spans,
                            support_locs=support_locs)
    Train = a01_train(80 / 3.6, np.zeros((5, 3)))
    Track = a02_track()
    Beam = a03_bridge(Beam)
    Calc, Beam, Track = a04_options(Beam, Track)
    Calc, Train, Beam = b43_model_geometry(Calc, Train, Track, Beam)
    Calc, Train, Track, Beam = b07_options_processing(Calc, Train, Track, Beam)
    Beam = b01_elements_and_coordinates(Beam, Calc)
    Beam.Prop.E_n[Beam.Prop.n_mod] = Beam.Prop.E * Beam.Prop.E_mod
    Beam = apply_cracks(Beam, Damage)   # mimic b00's crack step (before b02)
    Beam, Damage = b02_boundary_conditions(Beam, Damage)
    Beam = b03_beam_matrices(Beam)
    Beam = b09_beam_frq(Beam, Calc)
    return Beam


def w6(B):
    return np.round(np.asarray(B.Modal.w).ravel()[:6], 4)


# Reference values captured BEFORE the refactor (legacy code path):
REF_HEALTHY = np.array([824.1213, 1002.1424, 2223.0328, 2840.228, 4159.3323, 5352.7448])
REF_SCOUR30 = np.array([823.7625, 918.7162, 2084.4556, 2839.9953, 4104.762, 5352.5863])

print("=== Backward-compatibility (must match pre-refactor reference) ===")
h = w6(build(legacy_damage(0.0)))
s = w6(build(legacy_damage(0.30)))
print(f"legacy healthy  : {h}  match={np.allclose(h, REF_HEALTHY)}")
print(f"legacy scour=0.3: {s}  match={np.allclose(s, REF_SCOUR30)}")

print("\n=== New config path reproduces legacy ===")
hc = w6(build(make_damage(num_supports=3)))
sc = w6(build(make_damage(num_supports=3, scour_rates=[0.0, 0.30, 0.0])))
print(f"make_damage healthy  : {hc}  match_legacy={np.allclose(hc, REF_HEALTHY)}")
print(f"make_damage scour=0.3: {sc}  match_legacy={np.allclose(sc, REF_SCOUR30)}")

print("\n=== Feature toggles (should change frequencies sensibly) ===")
br = w6(build(make_damage(num_supports=3, bearing_rot_stiff=[1e9, 0.0, 1e9])))
print(f"bearing 1e9 @abutments  : {br}  (stiffer end rotation -> shifts modes)")
ck = w6(build(make_damage(num_supports=3, crack_locs=[30.0], crack_intensity=0.22)))
print(f"crack @30m, 22% EI loss : {ck}  (Fernandes-style mid-2nd-span)")
ms = w6(build(make_damage(num_supports=5, scour_rates=[0, 0.3, 0, 0.2, 0]),
              length=100.0, num_spans=4))
print(f"4-span multi-scour 100m : {ms}  (n_supports=5)")
mh = w6(build(make_damage(num_supports=5), length=100.0, num_spans=4))
print(f"4-span healthy 100m     : {mh}")
cs = w6(build(make_damage(num_supports=4),
              length=100.0, support_locs=[0.0, 0.2, 0.25, 1.0]))
print(f"custom close piers 100m : {cs}  (supports at [0,20,25,100] m)")
