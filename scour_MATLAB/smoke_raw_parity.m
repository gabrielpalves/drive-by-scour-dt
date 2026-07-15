% smoke_raw_parity.m — verify the Python load-time interp+crop mirrors MATLAB.
%
% "Option B" (2026-07-14): D01 now saves the RAW time-domain signal + the
% space-transform/crop parameters; Python rebuilds the space window at load time
% (core/dataset._raw_to_space_crop). This smoke computes the MATLAB reference for
% a few channels of a generated file and writes it next to the dataset; the
% companion checker (scratchpad/check_raw_parity.py) compares Python's result.
%
% RUN: from scour_MATLAB, after ONE state has been generated:
%   smoke_raw_parity('Results/<case_name>')
% Then:  python check_raw_parity.py "<path to the same folder>"
% PASS  = max|MATLAB - Python| == 0 (or < 1e-12).

function smoke_raw_parity(run_folder)

if nargin < 1 || isempty(run_folder)
    error('Usage: smoke_raw_parity(''Results/<case_name>'')');
end
f = fullfile(run_folder, '0001.mat');
assert(exist(f, 'file') == 2, 'No 0001.mat in %s — generate one state first.', f);
S = load(f); data = S.data;

assert(isfield(data, 'DimSpace'), ...
    'This file is LEGACY (pre-interpolated) — no DimSpace. Nothing to check.');

npass = min(3, numel(data.Velocidade));
ref = struct('AcelPrimVag', {{}}, 'PitchPrimVag', {{}}, 'AcelRodaPrimVag', {{}});

for p = 1:npass
    DimAcel  = data.DimAcel(1, p);
    DimSpace = data.DimSpace(1, p);
    cs = data.crop_start(1, p);
    ce = data.crop_end(1, p);

    % --- the LEGACY transform, verbatim ---
    xi = (1:DimSpace);
    xx = linspace(1, DimSpace, DimAcel);

    groups = {'AcelPrimVag', 'PitchPrimVag', 'AcelRodaPrimVag'};
    for g = 1:numel(groups)
        raw = data.(groups{g}){1, p};
        out = zeros(size(raw, 1), ce - cs + 1);
        for r = 1:size(raw, 1)
            y_space = interp1(xx, raw(r, :), xi);
            out(r, :) = y_space(cs:ce);
        end
        ref.(groups{g}){p} = out;
    end
    fprintf('passage %d: DimAcel=%d DimSpace=%d crop=[%d %d] -> %d samples\n', ...
        p, DimAcel, DimSpace, cs, ce, ce - cs + 1);
end

out_f = fullfile(run_folder, 'matlab_ref_parity.mat');
save(out_f, 'ref', 'npass');
fprintf('\nMATLAB reference written -> %s\n', out_f);
fprintf('Now run:  python check_raw_parity.py "%s"\n', run_folder);

end
