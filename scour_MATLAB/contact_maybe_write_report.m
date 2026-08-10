function contact_maybe_write_report(report, output_dir, overwrite)
%CONTACT_MAYBE_WRITE_REPORT Publish optional report files by checked renames.

common = contact_closure_common();
if strlength(string(output_dir)) == 0
    return
end
output_dir = common.absolute_path(char(output_dir));
dataset_dir = common.absolute_path(report.dataset_dir);
if common.is_same_or_child(output_dir, dataset_dir)
    error('contact_closure:UnsafeOutput', ...
        'OutputDir must not be the source dataset directory or its child.');
end
if ~isfolder(output_dir)
    [ok, msg] = mkdir(output_dir);
    if ~ok
        error('contact_closure:OutputCreateFailed', ...
            'Could not create OutputDir: %s', msg);
    end
end
safe_stage = regexprep(report.stage, '[^A-Za-z0-9_-]', '_');
stem = sprintf('%s_state%04d_pass%02d_contact_closure', ...
    safe_stage, report.state_index, report.passage_index);
mat_path = fullfile(output_dir, [stem, '.mat']);
md_path = fullfile(output_dir, [stem, '.md']);
if ~overwrite && (isfile(mat_path) || isfile(md_path))
    error('contact_closure:OutputExists', ...
        'Report already exists; pass Overwrite=true to replace it: %s', stem);
end

tmp_mat = [mat_path, '.tmp'];
tmp_md = [md_path, '.tmp'];
if isfile(tmp_mat) || isfile(tmp_md)
    error('contact_closure:OutputTempExists', ...
        'Stale report temporary file exists for: %s', stem);
end
save(tmp_mat, 'report');
contact_write_markdown(tmp_md, report);
[mat_moved, mat_message] = movefile(tmp_mat, mat_path, 'f');
if ~mat_moved
    error('contact_closure:OutputPublish', ...
        'Could not publish the MAT report: %s', mat_message);
end
[md_moved, md_message] = movefile(tmp_md, md_path, 'f');
if ~md_moved
    if isfile(mat_path)
        delete(mat_path);
    end
    error('contact_closure:OutputPublish', ...
        'Could not publish the Markdown report: %s', md_message);
end
fprintf('Contact-closure report: %s\n', md_path);
end
