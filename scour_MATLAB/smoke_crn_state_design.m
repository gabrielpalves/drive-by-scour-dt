function smoke_crn_state_design()
%SMOKE_CRN_STATE_DESIGN No-solver acceptance for generation-rules-v8 CRN.
%
% Verifies all four Paper-1 state catalogues, the explicit F40-S/F40-M
% matched subset, complete L99-S/L99-M pairing, stable semantic identities,
% collision-free named streams, and namespace isolation.

f40s = local_fixture('F40-S');
f40m = local_fixture('F40-M');
l99s = local_fixture('L99-S');
l99m = local_fixture('L99-M');

assert(f40s.n_states == 305);
assert(f40m.n_states == 425);
assert(l99s.n_states == 475 && l99m.n_states == 475);
local_family_counts(f40s,[5 300 0 0 0]);
local_family_counts(f40m,[50 25 50 50 250]);
local_family_counts(l99s,[50 75 50 50 250]);
local_family_counts(l99m,[50 75 50 50 250]);

% Dense design is severity-major and holds five independent semantic states
% at each integer percentage from 0 through 60.
assert(isequal(f40s.DamageStates(:,2), ...
    repelem((0:60)'/100,5)));
assert(all(f40s.DamageStates(:,[1 3]) == 0,'all'));
assert(isequal(f40s.AnchorLevel(6:end),repelem((1:60)',5)));

% F40-M shares exactly 5 healthy states and 5 replicas at each of the five
% controlled 12/24/36/48/60-percent anchors: 30 matched states in total.
[shared_uid,index_s,index_m] = intersect( ...
    f40s.StateUID,f40m.StateUID,'stable');
assert(numel(shared_uid) == 30);
assert(sum(strcmp(f40s.StateFamily(index_s),'target_healthy')) == 5);
assert(isequal(unique(f40s.AnchorLevel(index_s(index_s > 5)))', ...
    [12 24 36 48 60]));
assert(isequal(f40s.DamageStates(index_s,:),f40m.DamageStates(index_m,:)));
assert(isequal(f40s.StateSeedID(index_s),f40m.StateSeedID(index_m)));
assert(isequal(f40s.StateNamedStreamSeedID(index_s,:), ...
    f40m.StateNamedStreamSeedID(index_m,:)));
assert(isequal(f40s.PassageNamedStreamSeedID(index_s,:,:), ...
    f40m.PassageNamedStreamSeedID(index_m,:,:)));
assert(~any(f40m.CrackOn(index_m)));
assert(all(f40m.BearingFixity(index_m,:) == 0,'all'));

% L99-S and L99-M are the same complete latent design.  Only the registered
% bearing/crack activation changes; sample count, rows, UIDs, and EOV seeds do
% not, so the paired bridge-damage comparison is not composition-confounded.
assert(isequal(l99s.StateUID,l99m.StateUID));
assert(isequal(l99s.StateSeedID,l99m.StateSeedID));
assert(isequal(l99s.DamageStates,l99m.DamageStates));
assert(isequal(l99s.LatentBearingFixity,l99m.LatentBearingFixity));
assert(isequal(l99s.LatentCrackOn,l99m.LatentCrackOn));
assert(isequal(l99s.StateNamedStreamSeedID, ...
    l99m.StateNamedStreamSeedID));
assert(isequal(l99s.PassageNamedStreamSeedID, ...
    l99m.PassageNamedStreamSeedID));
assert(all(l99s.BearingFixity(:) == 0));
assert(isequal(l99m.BearingFixity,l99m.LatentBearingFixity));
assert(~any(l99s.CrackOn));
assert(isequal(l99m.CrackOn,l99m.LatentCrackOn));

% Independent hashlib.sha256 oracle (ASCII bytes, first eight hex digits).
% These constants prevent MATLAB's producer and test from agreeing on a wrong
% semantic byte contract.
expected_uid = [ ...
    'ttbi-state-v2|Lmm=040000|spans=2|scour=02|' ...
    'family=target_healthy|target=00|level=0000|rep=001'];
assert(strcmp(f40s.StateUID{1},expected_uid));
assert(f40s.StateSeedID(1) == uint32(1955233256));
assert(isequal(f40s.StateNamedStreamSeedID(1,:),uint32([ ...
    1471267274 2445595845 4221784991 2506555902 166414681])));
assert(isequal(reshape(f40s.PassageNamedStreamSeedID(1,1,:),1,[]), ...
    uint32([1508343681 3254353867])));

% A hostile insertion in one RNG namespace cannot perturb another namespace.
joint_index = find(strcmp(l99m.StateFamily,'joint'),1);
state_seed_row = l99m.StateNamedStreamSeedID(joint_index,:);
pass_seed_row = reshape( ...
    l99m.PassageNamedStreamSeedID(joint_index,1,:),1,[]);
base = local_namespace_draws(state_seed_row,pass_seed_row,0);
mutated = local_namespace_draws(state_seed_row,pass_seed_row,1000);
assert(isequal(base.operations,mutated.operations));
assert(isequal(base.profile,mutated.profile));
assert(isequal(base.track,mutated.track));
assert(isequal(base.oor,mutated.oor));
assert(~isequal(base.crack_tail,mutated.crack_tail));

fprintf(['[PASS] generation-rules-v8 CRN: F40-S=305, F40-M=425 ' ...
    '(30 matched), L99-S/M=475 (complete pairing), named streams isolated.\n']);
end

function design = local_fixture(stage)
if strcmp(stage,'F40-S')
    counts = [0 5 60 5 0];
else
    counts = [250 50 5 5 50];
end
campaign = ttbi.campaign_setup(struct( ...
    'STAGE',stage, ...
    'n_states_multi',counts(1), ...
    'Npass',50, ...
    'n_healthy_states',counts(2), ...
    'n_anchor_levels',counts(3), ...
    'n_anchor_reps',counts(4), ...
    'n_nuisance_states',counts(5)));
expected_decision_id = 'paper1-rail-domain-clearance-c06-v1';
assert(campaign.rail_end_clearance_m == 6);
assert(strcmp(campaign.rail_end_clearance_decision_id,expected_decision_id));
[profile_config,~,~] = ttbi.build_profile_config( ...
    campaign,[],uint32(1),uint32(2),uint32(3),1);
assert(profile_config.rail_end_clearance_m == 6);
assert(strcmp(profile_config.rail_end_clearance_decision_id, ...
    expected_decision_id));
[Calc,~,~] = A04_Options( ...
    A03_Bridge(struct()),A02_Track(),profile_config);
assert(Calc.Profile.rail_end_clearance_m == 6);
assert(strcmp(Calc.Profile.rail_end_clearance_decision_id, ...
    expected_decision_id));
design = ttbi.build_state_design(campaign);
assert(strcmp(design.random_stream_schedule_version, ...
    'uid-named-substreams-v2'));
end

function local_family_counts(design,expected)
names = {'target_healthy','scour_only','bearing_only','nuisance_only','joint'};
actual = zeros(1,numel(names));
for k = 1:numel(names)
    actual(k) = sum(strcmp(design.StateFamily,names{k}));
end
assert(isequal(actual,expected), ...
    'Family counts %s do not equal expected %s.', ...
    mat2str(actual),mat2str(expected));
end

function draws = local_namespace_draws(state_seeds,passage_seeds,extra_crack)
rng(double(state_seeds(1)),'twister');
draws.operations = [rand(1,8),randn(1,8)];
rng(double(state_seeds(2)),'twister');
rand(1,extra_crack);
draws.crack_tail = rand(1,8);
rng(double(state_seeds(3)),'twister');
draws.profile = rand(1,8);
rng(double(state_seeds(4)),'twister');
draws.track = rand(1,8);
rng(double(passage_seeds(2)),'twister');
draws.oor = rand(1,8);
end
