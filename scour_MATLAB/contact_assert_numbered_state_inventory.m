function names = contact_assert_numbered_state_inventory( ...
        dataset_dir, n_states)
%CONTACT_ASSERT_NUMBERED_STATE_INVENTORY Require exactly 0001..NNNN MAT files.

listed = dir(dataset_dir);
listed = listed(~[listed.isdir]);
numbered_mask = ~cellfun(@isempty, ...
    regexpi({listed.name}, '^\d{4}\.mat$', 'once'));
names = sort({listed(numbered_mask).name})';
expected = arrayfun(@(k) sprintf('%04d.mat', k), (1:n_states)', ...
    'UniformOutput', false);
if ~isequal(names, expected)
    error('contact_closure_gate:BadInventory', ...
        ['Dataset numbered-state inventory must be exactly ' ...
         '0001.mat..%04d.mat with canonical case.'], n_states);
end
end
