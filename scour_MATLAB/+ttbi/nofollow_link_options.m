function options = nofollow_link_options()
%NOFOLLOW_LINK_OPTIONS Java NIO option array for link-safe metadata queries.

options = javaArray('java.nio.file.LinkOption', 1);
options(1) = javaMethod( ...
    'valueOf', 'java.nio.file.LinkOption', 'NOFOLLOW_LINKS');
end
