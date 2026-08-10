function assert_resume_folder_rejected(folder, label, context)
%ASSERT_RESUME_FOLDER_REJECTED Require one file-inventory mutation to fail.

rejected = false;
folder_context = context;
folder_context.run_folder = folder;
folder_context.run_folder_observation = ...
    ttbi.directory_observation(folder);
try
    ttbi.validate_resume_states(folder, folder_context);
catch
    rejected = true;
end
assert(rejected, 'Resume inventory accepted mutation: %s', label);
end
