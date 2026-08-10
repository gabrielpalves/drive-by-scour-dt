function value = contact_utc_now()
%CONTACT_UTC_NOW Return current UTC time in the canonical receipt format.

value = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
end
