function assert_generator_source_unchanged(provenance)
%ASSERT_GENERATOR_SOURCE_UNCHANGED Re-read the reviewed source-root boundary.
%
% The start-time root, canonical digest lines, and file count must all match.
% Checking all three prevents a long campaign from being certified after a
% reviewed source or asset changes during execution.

ttbi.build_generator_source_attestation(provenance);
end
