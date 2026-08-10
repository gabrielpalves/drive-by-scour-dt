function Ref = vv_euler_bernoulli_reference(E,I,rho,A,element_lengths)
%VV_EULER_BERNOULLI_REFERENCE Independent exact Hermite-beam oracle.
%
% Ref = vv_euler_bernoulli_reference(E,I,rho,A,element_lengths) derives
% two-node Euler--Bernoulli element matrices by exact integration of cubic
% Hermite polynomials on xi in [0,1].  It then assembles the global matrices
% through one Boolean local-to-global transformation.  The implementation is
% intentionally test-only and does not call or copy B03_BeamMatrices.
%
% Inputs use SI units: E [N/m^2], I [m^4], rho [kg/m^3], A [m^2], and
% element_lengths [m].  Each material/section input may be scalar or have one
% value per element.  Generalized DOFs alternate transverse displacement [m]
% and rotation [rad].

lengths = local_positive_row(element_lengths,'element_lengths');
n_elements = numel(lengths);
E = local_expand_positive(E,n_elements,'E');
I = local_expand_positive(I,n_elements,'I');
rho = local_expand_positive(rho,n_elements,'rho');
A = local_expand_positive(A,n_elements,'A');

n_dof = 2*(n_elements + 1);
block_mass = zeros(4*n_elements);
block_stiffness = zeros(4*n_elements);
transform = zeros(4*n_elements,n_dof);
element = repmat(struct( ...
    'M',zeros(4),'K',zeros(4),'dof',zeros(1,4), ...
    'shape_coefficients_xi',zeros(4), ...
    'curvature_coefficients_xi',zeros(4,2)),1,n_elements);

for item = 1:n_elements
    L = lengths(item);
    % Coefficients are ascending powers of xi.  Rotational shape functions
    % carry L because the rotational generalized coordinates are radians.
    shape = [1, 0, -3,  2; ...
             0, L, -2*L, L; ...
             0, 0,  3, -2; ...
             0, 0, -L,  L];
    curvature = zeros(4,2);
    for shape_index = 1:4
        curvature(shape_index,:) = ...
            local_derivative(local_derivative(shape(shape_index,:)))/L^2;
    end

    element_mass = zeros(4);
    element_stiffness = zeros(4);
    for row = 1:4
        for column = 1:4
            element_mass(row,column) = rho(item)*A(item)*L * ...
                local_integral_product(shape(row,:),shape(column,:));
            element_stiffness(row,column) = E(item)*I(item)*L * ...
                local_integral_product( ...
                curvature(row,:),curvature(column,:));
        end
    end

    block_rows = (4*item-3):(4*item);
    global_dof = (2*item-1):(2*item+2);
    block_mass(block_rows,block_rows) = element_mass;
    block_stiffness(block_rows,block_rows) = element_stiffness;
    transform(block_rows,global_dof) = eye(4);

    element(item).M = element_mass;
    element(item).K = element_stiffness;
    element(item).dof = global_dof;
    element(item).shape_coefficients_xi = shape;
    element(item).curvature_coefficients_xi = curvature;
end

node_x = [0,cumsum(lengths)];
translation_mode = zeros(n_dof,1);
translation_mode(1:2:end) = 1;
rotation_mode = zeros(n_dof,1);
rotation_mode(1:2:end) = node_x;
rotation_mode(2:2:end) = 1;

Ref = struct();
Ref.M = transform.'*block_mass*transform;
Ref.K = transform.'*block_stiffness*transform;
Ref.element = element;
Ref.transform = transform;
Ref.node_x_m = node_x;
Ref.translation_mode = translation_mode;
Ref.rotation_mode = rotation_mode;
Ref.total_mass_kg = sum(rho.*A.*lengths);
Ref.units = struct( ...
    'E','N/m^2','I','m^4','rho','kg/m^3','A','m^2', ...
    'length','m','vertical_dof','m','rotation_dof','rad');
end

function values = local_expand_positive(values,n_elements,label)
values = local_positive_row(values,label);
if isscalar(values)
    values = repmat(values,1,n_elements);
elseif numel(values) ~= n_elements
    error('vv_beam_reference:PropertyCountMismatch', ...
        '%s must be scalar or have one value per element.',label);
end
end

function values = local_positive_row(values,label)
if ~isnumeric(values) || ~isreal(values) || ~isvector(values) || ...
        isempty(values) || any(~isfinite(values(:))) || any(values(:) <= 0)
    error('vv_beam_reference:InvalidInput', ...
        '%s must be a nonempty finite real positive vector.',label);
end
values = double(values(:).');
end

function derivative = local_derivative(polynomial)
% Differentiate ascending-power polynomial coefficients.
if isscalar(polynomial)
    derivative = 0;
else
    derivative = (1:numel(polynomial)-1).*polynomial(2:end);
end
end

function value = local_integral_product(left,right)
% Exact integral from zero to one of two ascending-power polynomials.
product = conv(left,right);
value = sum(product./(1:numel(product)));
end
