function file = java_file(path)
%JAVA_FILE java.io.File for PATH, resolved against MATLAB's current folder.
%
% MATLAB resolves a relative path against pwd. The JVM resolves it against
% user.dir, which is pinned to MATLAB's current folder when the JVM starts
% (at MATLAB startup in the desktop; at the first Java call under -batch) and
% is never updated by a later cd() (measured on R2023b and R2025b). A
% relative path handed straight to java.io.File therefore names an entry
% below the folder MATLAB was launched from, not the folder A00 verified with
% pwd: on a lab PC that opened MATLAB from the Start menu,
% ttbi.directory_observation('Results') looked for Documents\MATLAB\Results
% and threw NoSuchFileException while the MATLAB side (isfolder, mkdir) had
% just created scour_MATLAB\Results. Every Java path query in +ttbi builds
% its File here so both views name one filesystem entry.

if isstring(path) && isscalar(path)
    path = char(path);
end
if ~(ischar(path) && isrow(path) && ~isempty(path))
    error('ttbi:JavaFilePath', ...
        'Java file resolution requires one nonempty character-row path.');
end

file = javaObject('java.io.File', path);
if file.isAbsolute()
    return
end
% A drive-relative (C:name) or root-relative (\name) spelling is neither
% absolute for Java nor pwd-relative for MATLAB; refuse the ambiguity.
if ~isempty(regexp(path, '^([A-Za-z]:|[\\/])', 'once'))
    error('ttbi:JavaFileAmbiguousPath', ...
        ['Drive-relative or root-relative paths are ambiguous between the ' ...
         'MATLAB and JVM working directories: %s'], path);
end
file = javaObject('java.io.File', fullfile(pwd, path));
end
