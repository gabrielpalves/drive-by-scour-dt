function smoke_crn_state_design()
%SMOKE_CRN_STATE_DESIGN Focused no-physics test of generation-rules-v6 CRN.
% Verifies the fixed per-geometry state universe, master LHS pairing, stable
% UID/root seeds/latent variables, collision-free named streams, and namespace
% isolation. This intentionally does not launch the TTBI solver.

off60 = local_fixture(60, 3, [2 3], false, false);
on60  = local_fixture(60, 3, [2 3], true,  true);
on99  = local_fixture(99.6, 4, [2 3 4], true, true);

assert(size(off60.DamageStates, 1) == 450);
assert(size(on99.DamageStates, 1) == 475);
assert(isequal(off60.StateUID, on60.StateUID));
assert(isequal(off60.StateSeedID, on60.StateSeedID));
assert(isequal(off60.DamageStates, on60.DamageStates));
assert(isequal(off60.LatentBearingFixity, on60.LatentBearingFixity));
assert(isequal(off60.LatentCrackOn, on60.LatentCrackOn));
assert(isequal(off60.StateNamedStreamSeedID, ...
    on60.StateNamedStreamSeedID));
assert(isequal(off60.PassageNamedStreamSeedID, ...
    on60.PassageNamedStreamSeedID));
assert(all(off60.BearingFixity(:) == 0));
assert(isequal(on60.BearingFixity, on60.LatentBearingFixity));
assert(~any(off60.CrackOn));
assert(isequal(on60.CrackOn, on60.LatentCrackOn));

% Independent hashlib.sha256 oracle (UTF-8/ASCII keys, first eight hex
% digits). These constants prevent a shared MATLAB implementation error from
% making both A00 and this fixture agree on the wrong cross-language bytes.
assert(strcmp(off60.StateUID{1}, ...
    ['ttbi-state-v1|Lmm=060000|spans=3|scour=0203|' ...
     'family=target_healthy|target=00|level=0000|rep=001']));
assert(isequal(off60.StateSeedID([1 end])', ...
    uint32([1818075665 2898326234])));
assert(isequal(off60.StateNamedStreamSeedID(1, :), ...
    uint32([1571465212 3550721733 412983905 2757240308 3647310888])));
assert(isequal(reshape(off60.PassageNamedStreamSeedID(1, 1, :), 1, []), ...
    uint32([2623793449 2345927504])));

joint_ = strcmp(off60.StateFamily, 'joint');
assert(sum(joint_) == 250);
assert(isequal(off60.StateUID(joint_), on60.StateUID(joint_)));

% A hostile insertion in one namespace cannot perturb any other namespace.
seed_row_ = off60.StateNamedStreamSeedID(find(joint_, 1), :);
pass_seed_ = reshape(off60.PassageNamedStreamSeedID( ...
    find(joint_, 1), 1, :), 1, []);
base_ = local_namespace_draws(seed_row_, pass_seed_, 0);
mutated_ = local_namespace_draws(seed_row_, pass_seed_, 1000);
assert(isequal(base_.operations, mutated_.operations));
assert(isequal(base_.profile, mutated_.profile));
assert(isequal(base_.track, mutated_.track));
assert(isequal(base_.oor, mutated_.oor));
assert(~isequal(base_.crack_tail, mutated_.crack_tail));

fprintf(['[PASS] strong CRN state design: fixed L60=450/L99.6=475, ' ...
    '250 paired joint UIDs, collision-free named streams.\n']);
end

function out = local_fixture(L_bridge, num_spans, scour_supports, ...
        bearing_on, crack_on)
n_healthy = 50; n_levels = 5; n_reps = 5; n_nuisance = 50;
n_joint = 250; Npass = 50; damage_seed = 1; dano_max = 0.60;
fixity_max = 0.95; n_supp = num_spans + 1; n_tgt = numel(scour_supports);
schedule = 'uid-named-substreams-v2';
state_names = {'operations','crack','profile-state','track','profile-phase'};
passage_names = {'profile-passage','oor-passage'};

n_anchor = n_healthy + n_tgt * n_levels * n_reps + ...
    2 * n_levels * n_reps + n_nuisance;
n_total = n_anchor + n_joint;
damage = zeros(n_total, n_supp);
latent_bearing = zeros(n_total, 2);
family = cell(n_total, 1);
uids = cell(n_total, 1);
cursor = 0;

for rep = 1:n_healthy
    cursor = cursor + 1;
    family{cursor} = 'target_healthy';
    uids{cursor} = local_uid(L_bridge, num_spans, scour_supports, ...
        'target_healthy', 0, 0, rep);
end

levels_s = linspace(dano_max / n_levels, dano_max, n_levels)';
for target = scour_supports
    for rep = 1:n_reps
        for level = 1:n_levels
            cursor = cursor + 1;
            damage(cursor, target) = levels_s(level);
            family{cursor} = 'scour_only';
            uids{cursor} = local_uid(L_bridge, num_spans, ...
                scour_supports, 'scour_only', target, level, rep);
        end
    end
end

levels_b = linspace(fixity_max / n_levels, fixity_max, n_levels)';
for target = 1:2
    for rep = 1:n_reps
        for level = 1:n_levels
            cursor = cursor + 1;
            latent_bearing(cursor, target) = levels_b(level);
            family{cursor} = 'bearing_only';
            uids{cursor} = local_uid(L_bridge, num_spans, ...
                scour_supports, 'bearing_only', target, level, rep);
        end
    end
end

for rep = 1:n_nuisance
    cursor = cursor + 1;
    family{cursor} = 'nuisance_only';
    uids{cursor} = local_uid(L_bridge, num_spans, scour_supports, ...
        'nuisance_only', 0, 0, rep);
end
assert(cursor == n_anchor);

rng(damage_seed, 'twister');
lhs = lhsdesign(n_joint, n_tgt + 2);
joint_rows = cursor + (1:n_joint);
damage(joint_rows, scour_supports) = lhs(:, 1:n_tgt) * dano_max;
latent_bearing(joint_rows, :) = lhs(:, n_tgt+1:n_tgt+2) * fixity_max;
for row = 1:n_joint
    cursor = cursor + 1;
    family{cursor} = 'joint';
    uids{cursor} = local_uid(L_bridge, num_spans, scour_supports, ...
        'joint', 0, row, 1);
end
assert(cursor == n_total);

out.DamageStates = damage;
out.LatentBearingFixity = latent_bearing;
out.StateFamily = family;
out.StateUID = uids;
out.StateSeedID = local_root_seeds(uids, damage_seed);
[out.StateNamedStreamSeedID, out.PassageNamedStreamSeedID] = ...
    local_named_seeds(out.StateSeedID, uids, Npass, schedule, ...
        state_names, passage_names);
out.LatentCrackOn = false(numel(uids), 1);
joint_indices = find(strcmp(family, 'joint'));
for row = reshape(joint_indices, 1, [])
    out.LatentCrackOn(row) = local_uniform(uids{row}, damage_seed) <= 0.25;
end
out.LatentCrackOn(strcmp(family, 'nuisance_only')) = true;
out.BearingFixity = double(bearing_on) .* out.LatentBearingFixity;
out.CrackOn = logical(crack_on) & out.LatentCrackOn;
end

function draws = local_namespace_draws(state_seeds, passage_seeds, extra_crack)
rng(double(state_seeds(1)), 'twister');
draws.operations = [rand(1, 8), randn(1, 8)];
rng(double(state_seeds(2)), 'twister');
rand(1, extra_crack);
draws.crack_tail = rand(1, 8);
rng(double(state_seeds(3)), 'twister');
draws.profile = rand(1, 8);
rng(double(state_seeds(4)), 'twister');
draws.track = rand(1, 8);
rng(double(passage_seeds(2)), 'twister');
draws.oor = rand(1, 8);
end

function uid = local_uid(L_bridge, num_spans, scour_supports, ...
        family, target, level, replica)
uid = sprintf(['ttbi-state-v1|Lmm=%06d|spans=%d|scour=%s|' ...
    'family=%s|target=%02d|level=%04d|rep=%03d'], ...
    round(1000 * L_bridge), num_spans, sprintf('%02d', scour_supports), ...
    family, target, level, replica);
end

function ids = local_root_seeds(uids, damage_seed)
ids = zeros(numel(uids), 1, 'uint32');
for row = 1:numel(uids)
    ids(row) = local_seed32(sprintf( ...
        'ttbi-state-seed-v1|damage_seed=%.0f|%s', damage_seed, uids{row}));
end
assert(~any(ids == 0) && numel(unique(ids)) == numel(ids));
end

function [state_seeds, passage_seeds] = local_named_seeds( ...
        roots, uids, Npass, schedule, state_names, passage_names)
state_seeds = zeros(numel(uids), numel(state_names), 'uint32');
passage_seeds = zeros(numel(uids), Npass, numel(passage_names), 'uint32');
for row = 1:numel(uids)
    for stream = 1:numel(state_names)
        state_seeds(row, stream) = local_seed32(sprintf( ...
            '%s|root=%u|uid=%s|stream=%s', schedule, roots(row), ...
            uids{row}, state_names{stream}));
    end
    for pass = 1:Npass
        for stream = 1:numel(passage_names)
            passage_seeds(row, pass, stream) = local_seed32(sprintf( ...
                '%s|root=%u|uid=%s|stream=%s|pass=%05d', schedule, ...
                roots(row), uids{row}, passage_names{stream}, pass));
        end
    end
end
all_ids = [roots(:); state_seeds(:); passage_seeds(:)];
assert(~any(all_ids == 0) && numel(unique(all_ids)) == numel(all_ids));
end

function u = local_uniform(uid, damage_seed)
h = local_sha256(sprintf('latent-crack-v1|damage_seed=%.0f|%s', ...
    damage_seed, uid));
u = hex2dec(h(1:13)) / 16^13;
end

function seed = local_seed32(key)
h = local_sha256(key);
seed = uint32(hex2dec(h(1:8)));
end

function h = local_sha256(text)
md = java.security.MessageDigest.getInstance('SHA-256');
raw = md.digest(reshape(uint8(text), [], 1));
h = lower(sprintf('%02x', typecast(int8(raw), 'uint8')));
end
