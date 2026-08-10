function write_qualification_host_receipt(path, receipt)
%WRITE_QUALIFICATION_HOST_RECEIPT Write the qualification host receipt as canonical JSON.
    encoded_ = [jsonencode(receipt), newline];
    expected_bytes_ = reshape( ...
        unicode2native(encoded_, 'UTF-8'), 1, []);
    if isfile(path)
        fid_ = fopen(path, 'rb');
        if fid_ < 0
            error('A00:QualificationHostReceiptRead', ...
                'Cannot read existing host receipt: %s', path);
        end
        observed_bytes_ = reshape(fread(fid_, Inf, '*uint8'), 1, []);
        fclose(fid_);
        if ~isequal(observed_bytes_, expected_bytes_)
            error('A00:QualificationHostReceiptCollision', ...
                ['Existing qualification_host_receipt.json differs from this ' ...
                 'host/run identity. Use a fresh output folder.']);
        end
        return
    end
    temp_path_ = [path '.tmp'];
    if isfile(temp_path_)
        error('A00:QualificationHostReceiptTemp', ...
            'Stale host-receipt temporary file exists: %s', temp_path_);
    end
    fid_ = fopen(temp_path_, 'wb');
    if fid_ < 0
        error('A00:QualificationHostReceiptWrite', ...
            'Cannot create host receipt temporary file: %s', temp_path_);
    end
    wrote_ = fwrite(fid_, expected_bytes_, 'uint8');
    close_status_ = fclose(fid_);
    if wrote_ ~= numel(expected_bytes_) || close_status_ ~= 0
        error('A00:QualificationHostReceiptWrite', ...
            'Could not persist complete host receipt bytes: %s', temp_path_);
    end
    [moved_, move_message_] = movefile(temp_path_, path, 'f');
    if ~moved_
        error('A00:QualificationHostReceiptWrite', ...
            'Could not atomically install host receipt: %s', move_message_);
    end
    if ~strcmp(ttbi.file_sha256(path), ...
            ttbi.sha256_bytes(expected_bytes_))
        error('A00:QualificationHostReceiptWrite', ...
            'Persisted host receipt bytes failed verification.');
    end
end
