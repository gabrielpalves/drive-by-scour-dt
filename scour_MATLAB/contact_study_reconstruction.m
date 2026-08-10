function R = contact_study_reconstruction()
%CONTACT_STUDY_RECONSTRUCTION Saved-state capture, validation and rebuild.
%
% Each responsibility is a one-function module. Stable handle fields preserve
% the public API used by the closure-study orchestrator and metrics modules.

R = struct();
R.load_dataset_snapshots = @contact_load_study_dataset_snapshots;
R.validate_case = @contact_validate_case;
R.validate_r11_descriptor = @contact_validate_r11_descriptor;
R.verify_dataset_integrity = @contact_verify_dataset_integrity;
R.select_passage = @contact_select_passage;
R.damage_descriptor = @contact_damage_descriptor;
R.profile_descriptor = @contact_profile_descriptor;
R.indexed_value = @contact_indexed_value;
R.case_text = @contact_case_text;
R.state_text = @contact_state_text;
end
