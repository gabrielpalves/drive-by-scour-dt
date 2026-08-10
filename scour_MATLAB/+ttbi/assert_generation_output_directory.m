function assert_generation_output_directory(path, expected)
%ASSERT_GENERATION_OUTPUT_DIRECTORY Keep generation inside one real directory.
%
% EXPECTED is captured once, immediately after A00 creates or opens the run
% folder. Re-reading both the canonical path and filesystem identity rejects
% a symlink/junction/reparse component and a persistent directory replacement
% before a later read, write, move, or credential revocation can cross it.

required = sort({'canonical_path'; 'file_key'});
if ~isstruct(expected) || ~isscalar(expected) || ...
        ~isequal(sort(fieldnames(expected)), required)
    error('ttbi:GenerationOutputObservation', ...
        'Expected generation-output observation is malformed.');
end

current = ttbi.directory_observation(path);
if ~isequal(current, expected)
    error('ttbi:GenerationOutputChanged', ...
        ['Generation output directory identity changed after it was ' ...
         'authenticated: %s'], path);
end
end
