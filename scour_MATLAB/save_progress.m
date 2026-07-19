function save_progress(data, DC, run_path, gen_schema, gen_fingerprint)
    % Atomic per-state save (audit R4) + top-level provenance (audit R5).
    % Writes to a temp file in the SAME folder then renames over the final name,
    % so a crash mid-write can never leave a partial NNNN.mat that resume treats
    % as done. gen_schema / gen_fingerprint are stored BOTH inside `data` and as
    % small TOP-LEVEL variables so the resume guard can validate a file's
    % provenance with a cheap `load(file,'file_gen_schema','file_gen_fingerprint')`
    % instead of loading the large signal arrays.
    if nargin >= 4, file_gen_schema      = gen_schema;      else, file_gen_schema = ''; end %#ok<NASGU>
    if nargin >= 5, file_gen_fingerprint = gen_fingerprint; else, file_gen_fingerprint = ''; end %#ok<NASGU>
    filename   = sprintf('%04d.mat', DC);
    final_path = fullfile(run_path, filename);
    tmp_path   = fullfile(run_path, sprintf('.%04d.mat.tmp', DC));
    if exist(tmp_path, 'file'), delete(tmp_path); end
    save(tmp_path, 'data', 'file_gen_schema', 'file_gen_fingerprint');
    if exist(final_path, 'file'), delete(final_path); end
    movefile(tmp_path, final_path, 'f');
end
