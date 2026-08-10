function seed = seed32(key)
%SEED32 32-bit seed from a namespaced key (SHA-256 prefix).
    h = ttbi.sha256(key);
    seed = uint32(hex2dec(h(1:8)));
end
