function create_hardlink(alias_path, target_path)
%CREATE_HARDLINK Create a test alias through Java NIO without a shell.

alias_nio = javaObject('java.io.File', alias_path).toPath();
target_nio = javaObject('java.io.File', target_path).toPath();
javaMethod('createLink', 'java.nio.file.Files', alias_nio, target_nio);
end
