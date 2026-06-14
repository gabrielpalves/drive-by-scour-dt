function save_progress(data, DC, run_path)
    filename = sprintf('%04d.mat', DC);
    save(fullfile(run_path, filename), 'data');
end
