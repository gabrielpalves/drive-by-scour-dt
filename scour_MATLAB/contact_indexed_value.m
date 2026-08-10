function value = contact_indexed_value(container, index)
%CONTACT_INDEXED_VALUE Read one passage descriptor from supported containers.

if iscell(container)
    value = container{index};
elseif isstruct(container) && numel(container) >= index
    value = container(index);
elseif isempty(container)
    value = [];
else
    error('contact_closure:BadDescriptor', ...
        'Passage descriptor has an unsupported container type/size.');
end
end
