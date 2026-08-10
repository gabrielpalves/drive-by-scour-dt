function [positions, lattice, failed] = sample_pad_failures( ...
        track_window, pad_spacing, failure_probability)
%SAMPLE_PAD_FAILURES Independent Bernoulli failure at each sleeper position.
%
% The caller owns the RNG stream. This helper consumes exactly one uniform
% draw for every point on the inclusive bridge-local sleeper lattice and
% selects positions without replacement.

validateattributes(track_window, {'numeric'}, ...
    {'real','finite','scalar','positive'});
validateattributes(pad_spacing, {'numeric'}, ...
    {'real','finite','scalar','positive'});
validateattributes(failure_probability, {'numeric'}, ...
    {'real','finite','scalar','>=',0,'<=',1});

lattice = 0:pad_spacing:track_window;
failed = rand(size(lattice)) < failure_probability;
positions = lattice(failed);
end
