function seed_stale_generation_credentials(folder)
%SEED_STALE_GENERATION_CREDENTIALS Install all obsolete smoke credentials.

credential_names = ttbi.generation_publication_credential_names();
stale_bytes = unicode2native(sprintf('stale-publication-credential\n'), ...
    'UTF-8');
for name_index = 1:numel(credential_names)
    path = fullfile(folder, credential_names{name_index});
    file_id = fopen(path, 'wb');
    if file_id < 0
        error('ttbi:SmokeCredentialOpen', ...
            'Could not create stale publication credential: %s', path);
    end
    written = fwrite(file_id, stale_bytes, 'uint8');
    close_status = fclose(file_id);
    if written ~= numel(stale_bytes) || close_status ~= 0
        error('ttbi:SmokeCredentialWrite', ...
            'Could not flush stale publication credential: %s', path);
    end
end
end
