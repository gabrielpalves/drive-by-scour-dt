function smoke_bridge_mesh_alignment()
%SMOKE_BRIDGE_MESH_ALIGNMENT Verify production supports cannot be snapped.

cases = struct( ...
    'length_m', {60.0, 99.6, 40.0}, ...
    'num_spans', {3, 4, 2}, ...
    'expected_elements_per_bay', {3, 2, 3});

for c = 1:numel(cases)
    Track = A02_Track();
    Beam_seed = struct('Prop', struct( ...
        'L', cases(c).length_m, 'num_spans', cases(c).num_spans));
    Beam = A03_Bridge(Beam_seed);
    [~, Beam, Track] = A04_Options(Beam, Track);
    assert(Beam.Mesh.Ele.num_per_spacing == ...
        cases(c).expected_elements_per_bay);

    h = Track.Sleeper.spacing / Beam.Mesh.Ele.num_per_spacing;
    Beam.Mesh.Ele.num = round(Beam.Prop.L/h);
    Beam.Prop.A = 1;
    Beam = B01_ElementsAndCoordinates(Beam);
    Beam.BC.loc = linspace(0, Beam.Prop.L, Beam.Prop.num_spans+1);
    Beam.BC.vert_stiff = ones(size(Beam.BC.loc));
    Beam.BC.rot_stiff = [1, zeros(1, Beam.Prop.num_spans-1), 1];
    Damage = struct('scour_rates', zeros(size(Beam.BC.loc)), ...
        'bearing_left', 0, 'bearing_right', 0);
    [Beam, ~] = B02_BoundaryConditions(Beam, Damage);
    assert(max(abs(Beam.BC.loc_offset)) <= Beam.BC.loc_tolerance);
end

% Recreate the old L60 mesh directly: the boundary-condition layer must now
% reject its 0.10 m support shift instead of accepting nearest nodes.
Beam = A03_Bridge(struct('Prop', struct('L', 60, 'num_spans', 3)));
Beam.Prop.A = 1;
Beam.Mesh.Ele.num = 200;
Beam = B01_ElementsAndCoordinates(Beam);
Beam.BC.loc = [0, 20, 40, 60];
Beam.BC.vert_stiff = ones(1, 4);
Beam.BC.rot_stiff = [1, 0, 0, 1];
Damage = struct('scour_rates', zeros(1, 4), ...
    'bearing_left', 0, 'bearing_right', 0);
local_assert_error(@() B02_BoundaryConditions(Beam, Damage), ...
    'B02:SupportNotOnNode');

local_assert_error(@() bridge_mesh_elements_per_sleeper( ...
    sqrt(2), 3, 0.6, 2), 'bridge_mesh:NoSupportAlignedDensity');
local_assert_error(@() bridge_mesh_elements_per_sleeper( ...
    60, 2.5, 0.6, 2), 'bridge_mesh:InvalidPositiveInteger');

fprintf(['[PASS] support-aligned bridge meshes: L60/3 and L40/2 use ' ...
    'h=0.20 m; L99.6/4 retains h=0.30 m; snapping rejected.\n']);
end

function local_assert_error(action, expected_id)
did_error = false;
try
    action();
catch ME
    did_error = true;
    assert(strcmp(ME.identifier, expected_id), ...
        'Expected error %s, received %s.', expected_id, ME.identifier);
end
assert(did_error, 'Expected error %s was not raised.', expected_id);
end
