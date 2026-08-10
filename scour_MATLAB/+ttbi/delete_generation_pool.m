function delete_generation_pool(pool)
%DELETE_GENERATION_POOL Tear down one temporary generation ProcessPool.

if ~isempty(pool) && isvalid(pool)
    delete(pool);
end
end
