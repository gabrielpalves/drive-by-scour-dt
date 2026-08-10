function tf = regular_nonsymlink_directory(path)
%REGULAR_NONSYMLINK_DIRECTORY Accept one real directory with no path aliases.

tf = false;
try
    ttbi.directory_observation(path);
    tf = true;
catch
    % This predicate is used at trust boundaries. Any unavailable or
    % ambiguous identity query rejects the directory.
end
end
