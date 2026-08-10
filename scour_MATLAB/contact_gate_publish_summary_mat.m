function contact_gate_publish_summary_mat( ...
        output_dir, summary, selection_descriptor, selection_sha)
%CONTACT_GATE_PUBLISH_SUMMARY_MAT Publish the immutable summary source.

summary_path = fullfile(output_dir, 'gate_summary.mat');
publication = struct('summary', summary, ...
    'selection_descriptor', selection_descriptor, ...
    'selection_sha256', selection_sha);
contact_gate_save_atomic(summary_path, 'publication', publication);
end
