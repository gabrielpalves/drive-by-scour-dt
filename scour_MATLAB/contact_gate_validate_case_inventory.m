function contact_gate_validate_case_inventory( ...
        cases_dir, expected_cases, require_complete)
%CONTACT_GATE_VALIDATE_CASE_INVENTORY Enforce exact case MAT/JSON pairs.

common = contact_closure_common();
entries = dir(cases_dir);
entries = entries(~ismember({entries.name}, {'.', '..'}));
mat_seen = false(expected_cases, 1);
json_seen = false(expected_cases, 1);
for k = 1:numel(entries)
    entry = entries(k);
    tokens = regexp(entry.name, ...
        '^(\d{4})_case\.(mat|json)$', 'tokens', 'once');
    if entry.isdir || isempty(tokens) || ...
            ~common.regular_nonsymlink(fullfile(cases_dir, entry.name))
        error('contact_closure_gate:CaseInventory', ...
            'Foreign/subdirectory/tmp/symlink case entry: %s', entry.name);
    end
    ordinal = str2double(tokens{1});
    if ordinal < 1 || ordinal > expected_cases || ...
            ~strcmp(tokens{1}, sprintf('%04d', ordinal))
        error('contact_closure_gate:CaseInventory', ...
            'Case artifact ordinal is outside the frozen inventory: %s', ...
            entry.name);
    end
    if strcmp(tokens{2}, 'mat')
        if mat_seen(ordinal)
            error('contact_closure_gate:CaseInventory', ...
                'Duplicate case MAT ordinal %d.', ordinal);
        end
        mat_seen(ordinal) = true;
    else
        if json_seen(ordinal)
            error('contact_closure_gate:CaseInventory', ...
                'Duplicate case JSON ordinal %d.', ordinal);
        end
        json_seen(ordinal) = true;
    end
end
if any(json_seen & ~mat_seen)
    error('contact_closure_gate:CaseInventory', ...
        'A case JSON exists without its immutable MAT source of truth.');
end
if require_complete && ...
        (~all(mat_seen) || ~all(json_seen) || numel(entries) ~= 2 * expected_cases)
    error('contact_closure_gate:CaseInventory', ...
        'Final case inventory is not exactly %d MAT/JSON pairs.', ...
        expected_cases);
end
end
