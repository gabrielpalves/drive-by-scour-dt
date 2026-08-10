function case_info = contact_case_info_from_snapshot(case_blob)
%CONTACT_CASE_INFO_FROM_SNAPSHOT Extract one scalar saved case descriptor.

if ~isfield(case_blob, 'case_info') || ~isstruct(case_blob.case_info) || ...
        ~isscalar(case_blob.case_info)
    error('contact_closure:BadCaseInfo', ...
        'case_info.mat does not contain one scalar case_info struct.');
end
case_info = case_blob.case_info;
end
