function m1_single_tx(config_path, outdir)
% One authorized transmission only. Never falls back after a GPU error.
cfg = jsondecode(fileread(config_path));
assert(isfolder(outdir), 'Runner must create a new output directory.');
assert(cfg.gates.mapping_confirmed && ~cfg.gates.full_fmc_started);
addpath(cfg.paths.kwave);
rng(cfg.seed, 'twister');
set(groot, 'defaultFigureVisible', 'off');
total_timer = tic;
try
    label = read_label_npy(cfg.paths.label_npy, cfg.model.label_shape);
    assert(all(isfinite(label(:))) && min(label(:)) >= cfg.model.c_defect_m_s);
    assert(max(label(:)) <= cfg.model.c_background_m_s);
    scale = cfg.grid.shape ./ cfg.model.label_shape;
    assert(all(scale == round(scale)));
    c = repelem(label, scale(1), scale(2));
    assert(isequal(size(c), cfg.grid.shape(:)'));
    original = read_letter(cfg, outdir);
    plot_model(original, label, cfg, outdir);

    nz = cfg.grid.shape(1); nx = cfg.grid.shape(2);
    dx = cfg.grid.spacing_m;
    dt = cfg.time.internal_dt_s; nt = cfg.time.internal_count;
    assert(abs(nt*dt - cfg.time.end_s) < eps);
    stride = round(cfg.time.output_dt_s / dt);
    assert(abs(stride*dt - cfg.time.output_dt_s) < eps);
    assert(nt / stride == cfg.time.output_count);
    kgrid = kWaveGrid(nz, dx, nx, dx);
    kgrid.setTime(nt, dt);
    t_internal = double(kgrid.t_array);
    t_output = t_internal(1:stride:end);
    medium.sound_speed = c;
    medium.sound_speed_ref = cfg.model.c_background_m_s;
    medium.density = cfg.model.density_kg_m3;
    dt_limit = checkStability(kgrid, medium);
    assert(dt <= dt_limit, 'Chosen dt exceeds k-Wave stability estimate.');

    x_array = cfg.array.center_x_m + ...
        ((0:cfg.array.elements-1) - (cfg.array.elements-1)/2)*cfg.array.pitch_m;
    x_indices = round(x_array/dx)+1;
    z_index = round(cfg.array.z_m/dx)+1;
    assert(all(abs((x_indices-1)*dx-x_array) < dx*1e-8));
    assert(abs(mean(x_array)-cfg.grid.domain_m(2)/2) < eps);
    assert(numel(unique(x_indices)) == cfg.array.elements);
    receiver_indices = sub2ind([nz,nx], repmat(z_index,1,cfg.array.elements), x_indices);
    sensor.mask = false(nz,nx); sensor.mask(receiver_indices) = true;
    sensor.record = cfg.diagnostics.record;
    [found, order] = ismember(receiver_indices, find(sensor.mask));
    assert(all(found));
    tx = cfg.array.transmitter_index_one_based;
    source.p_mask = false(nz,nx); source.p_mask(receiver_indices(tx)) = true;
    source.p_mode = cfg.source.mode;
    tau = t_internal-cfg.source.time_shift_s;
    arg = (pi*cfg.source.frequency_hz*tau).^2;
    source_wavelet = cfg.source.pressure_pa*(1-2*arg).*exp(-arg);
    source.p = single(source_wavelet);
    [~, peak_idx] = max(source_wavelet);
    parallel.gpu.enableCUDAForwardCompatibility(cfg.compute.cuda_forward_compatibility);
    gpu = gpuDevice; wait(gpu);
    meta = struct;
    meta.config = cfg;
    meta.matlab_version = version;
    meta.gpu_name = gpu.Name;
    meta.gpu_total_bytes = gpu.TotalMemory;
    meta.gpu_available_before_solver_bytes = gpu.AvailableMemory;
    meta.cfl_actual = max(double(c(:)))*dt/dx;
    meta.stability_dt_limit_infinite = isinf(dt_limit);
    if isfinite(dt_limit), meta.stability_dt_limit_s = dt_limit; end
    meta.sampled_source_peak_s = t_internal(peak_idx);
    meta.analytic_source_peak_s = cfg.source.time_shift_s;
    meta.snapshot_time_s = t_internal(end);
    meta.source_x_m = x_array(tx);
    meta.receiver_x_m = x_array;
    meta.source_z_m = (z_index-1)*dx;
    meta.pml_expanded_shape = [nz,nx] + 2*cfg.grid.pml_size;
    meta.solver_started_local = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss.SSS'));
    fprintf('SIMULATION_START=%s\n', meta.solver_started_local);
    solver_timer = tic;
    data = kspaceFirstOrder2D(kgrid, medium, source, sensor, ...
        'PMLInside', cfg.grid.pml_inside, 'PMLSize', cfg.grid.pml_size, ...
        'PMLAlpha', cfg.grid.pml_alpha, 'DataCast', cfg.compute.data_cast, ...
        'DataRecast', false, 'PlotSim', cfg.compute.plot_sim, ...
        'Smooth', false);
    wait(gpu);
    meta.solver_wall_s = toc(solver_timer);
    meta.solver_finished_local = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss.SSS'));
    meta.gpu_available_after_solver_bytes = gpu.AvailableMemory;
    fprintf('SIMULATION_END=%s\nSOLVER_SECONDS=%.6f\n', ...
        meta.solver_finished_local, meta.solver_wall_s);
    meta.returned_p_is_gpuArray = isa(data.p,'gpuArray');
    assert(meta.returned_p_is_gpuArray, 'Expected actual GPU output.');
    p_internal = single(gather(data.p(order,:)));
    p_final = single(gather(data.p_final));
    p_max_all = single(gather(data.p_max_all));
    p_min_all = single(gather(data.p_min_all));
    assert(isequal(size(p_internal),[cfg.array.elements,nt]));
    assert(isequal(size(p_final),[nz,nx]));
    meta.all_outputs_finite = all(isfinite(p_internal(:))) && ...
        all(isfinite(p_final(:))) && all(isfinite(p_max_all(:))) && all(isfinite(p_min_all(:)));
    assert(meta.all_outputs_finite, 'Nonfinite solver output: stop.');
    p_output = p_internal(:,1:stride:end);
    assert(isequal(size(p_output),[cfg.array.elements,cfg.time.output_count]));
    meta.global_peak_abs_pa = max([max(abs(p_max_all(:))),max(abs(p_min_all(:)))]);
    meta.final_peak_abs_pa = max(abs(p_final(:)));
    meta.receivers_peak_abs_pa = max(abs(p_internal(:)));
    meta.final_to_global_peak_ratio = meta.final_peak_abs_pa / max(meta.global_peak_abs_pa,realmin('single'));
    meta.source_high_frequency_energy_ratio = spectral_tail(source_wavelet,dt,0.5/cfg.time.output_dt_s);
    meta.receiver_high_frequency_energy_ratio = spectral_tail(p_internal,dt,0.5/cfg.time.output_dt_s);
    meta.alias_tail_below_threshold = meta.receiver_high_frequency_energy_ratio < cfg.diagnostics.spectral_tail_warning_ratio;
    late = t_internal >= cfg.diagnostics.late_time_start_s;
    meta.receiver_late_rms_pa = sqrt(mean(double(p_internal(:,late)).^2,'all'));
    meta.receiver_all_rms_pa = sqrt(mean(double(p_internal).^2,'all'));
    b = cfg.diagnostics.boundary_strip_cells;
    edge = false(nz,nx); edge([1:b,end-b+1:end],:)=true; edge(:,[1:b,end-b+1:end])=true;
    meta.final_boundary_rms_pa = sqrt(mean(double(p_final(edge)).^2));
    meta.final_interior_rms_pa = sqrt(mean(double(p_final(~edge)).^2));
    meta.pml_assessment_limit = 'Single final snapshot and receiver traces cannot isolate or quantify PML reflection; no homogeneous control or PML-thickness convergence experiment performed.';
    post_timer = tic;
    save(fullfile(outdir,'single_tx.mat'), 'p_internal','p_output','t_internal','t_output', ...
        'label','c','original','source_wavelet','p_final','p_max_all','p_min_all','meta','-v7.3');
    plot_outputs(p_output,p_internal,p_final,p_max_all,p_min_all,t_output,t_internal,cfg,outdir);
    meta.save_and_plot_s = toc(post_timer);
    meta.total_function_s = toc(total_timer);
    listing=dir(fullfile(outdir,'single_tx.mat')); meta.mat_file_bytes=listing.bytes;
    % Check the on-disk artifact, rather than only the in-memory arrays.
    mf=matfile(fullfile(outdir,'single_tx.mat'));
    assert(isequal(size(mf,'p_output'),[cfg.array.elements,cfg.time.output_count]));
    assert(isequal(mf.t_output,t_output));
    meta.disk_reopen_passed=true;
    write_json(fullfile(outdir,'metrics.json'),meta);
    fprintf('ALL_FINITE=%d ALIAS_TAIL=%.9g FINAL_GLOBAL_RATIO=%.9g\n', ...
        meta.all_outputs_finite,meta.receiver_high_frequency_energy_ratio,meta.final_to_global_peak_ratio);
    fprintf('M1_SINGLE_TX_OK=1\nOUTPUT=%s\n',outdir);
catch ME
    fid=fopen(fullfile(outdir,'failure.txt'),'w','n','UTF-8');
    if fid >= 0, fprintf(fid,'%s',getReport(ME,'extended','hyperlinks','off')); fclose(fid); end
    fprintf('M1_SINGLE_TX_OK=0\n');
    rethrow(ME); % No automatic retry or CPU fallback.
end
end

function a=read_label_npy(path,shape)
fid=fopen(path,'r','ieee-le'); assert(fid>=0); cleanup=onCleanup(@()fclose(fid));
magic=fread(fid,6,'*uint8')'; assert(isequal(magic,[147,uint8('NUMPY')]));
v=fread(fid,2,'*uint8'); assert(v(1)==1);
n=fread(fid,1,'uint16'); hdr=char(fread(fid,n,'*uint8')');
assert(contains(hdr,'''<f4''') && contains(hdr,'False'));
a=fread(fid,[shape(2),shape(1)],'*single')';
assert(numel(a)==prod(shape) && isempty(fread(fid,1,'uint8')));
end

function img=read_letter(cfg,outdir)
% MATLAB column-major reshape is exactly the required transpose of IDX's
% row-major stored 28x28 matrix; do NOT apply a second transpose.
folder=fullfile(outdir,'idx'); mkdir(folder);
files=gunzip(cfg.paths.emnist_images,folder);
fid=fopen(files{1},'r','ieee-be'); assert(fid>=0); cleanup=onCleanup(@()fclose(fid));
hdr=fread(fid,4,'uint32'); assert(hdr(1)==2051 && all(hdr(3:4)==28));
fseek(fid,16+cfg.model.sample_index_zero_based*prod(hdr(3:4)),'bof');
img=reshape(fread(fid,prod(hdr(3:4)),'*uint8'),hdr(3:4)');
end

function r=spectral_tail(x,dt,cutoff)
if isvector(x),x=x(:)';end
n=size(x,2); f=(0:floor(n/2))/(n*dt); y=fft(double(x),[],2);
energy=sum(abs(y(:,1:numel(f))).^2,1);
r=sum(energy(f>cutoff))/max(sum(energy),realmin);
end

function plot_model(original,label,cfg,outdir)
f=figure('Position',[100,100,1150,450]); tiledlayout(1,2,'TileSpacing','compact');
nexttile; imagesc(original,[0,255]); axis image; colormap(gca,gray(256));
title('EMNIST W: raw IDX transposed once'); xlabel('Column'); ylabel('Row');
nexttile; imagesc([0,cfg.grid.domain_m(2)*1e3],[0,cfg.grid.domain_m(1)*1e3],label);
axis image; set(gca,'YDir','reverse'); clim([cfg.model.c_defect_m_s,cfg.model.c_background_m_s]);
colormap(gca,turbo(256)); cb=colorbar; cb.Label.String='Sound speed (m/s)';
title('B: 56x56 support, centered in 200x200'); xlabel('x (mm)'); ylabel('Depth (mm)');
exportgraphics(f,fullfile(outdir,'confirmed_model.png'),'Resolution',cfg.diagnostics.figure_dpi); close(f);
end

function plot_outputs(p,praw,pfinal,pmax,pmin,t,traw,cfg,outdir)
peak=max(abs(p(:))); scale=max(peak,realmin('single'));
f=figure('Position',[100,100,1200,700]); tiledlayout(2,1,'TileSpacing','compact');
nexttile; imagesc(t*1e6,1:size(p,1),p); clim(double([-scale,scale]));
colormap(gca,turbo(256)); cb=colorbar; cb.Label.String='Pressure (Pa)';
title(sprintf('Tx %d: receiver B-scan, linear pressure',cfg.array.transmitter_index_one_based));
xlabel('Time from acquisition start (us)'); ylabel('Receiver index');
nexttile; imagesc(t*1e6,1:size(p,1),20*log10(max(abs(p)/scale,realmin('single'))));
clim([cfg.diagnostics.bscan_db_floor,0]); colormap(gca,parula(256)); cb=colorbar; cb.Label.String='Amplitude (dB re global trace peak)';
title('Same data: log magnitude, no gain or direct-wave subtraction');
xlabel('Time from acquisition start (us)'); ylabel('Receiver index');
exportgraphics(f,fullfile(outdir,'receiver_bscan.png'),'Resolution',cfg.diagnostics.figure_dpi);close(f);
f=figure('Position',[100,100,800,700]);
ax=(0:size(pfinal,2)-1)*cfg.grid.spacing_m*1e3;
az=(0:size(pfinal,1)-1)*cfg.grid.spacing_m*1e3;
imagesc(ax,az,pfinal);axis image;set(gca,'YDir','reverse');
lim=max(abs(pfinal(:)));clim(double([-lim,lim])); colormap(turbo(256));
cb=colorbar;cb.Label.String='Pressure (Pa)';
title(sprintf('Final pressure snapshot at %.3f us (PML excluded)',traw(end)*1e6));
xlabel('x (mm)');ylabel('Depth (mm)');
exportgraphics(f,fullfile(outdir,'wavefield_final.png'),'Resolution',cfg.diagnostics.figure_dpi);close(f);
f=figure('Position',[100,100,1150,450]);tiledlayout(1,2,'TileSpacing','compact');
nexttile; semilogy(traw*1e6,max(abs(praw),[],1)+realmin('single')); grid on;
xlabel('Time (us)');ylabel('Max abs pressure over 64 receivers (Pa)');title('Receiver amplitude history');
nexttile; imagesc(ax,az,max(abs(pmax),abs(pmin)));axis image;set(gca,'YDir','reverse');colorbar;
xlabel('x (mm)');ylabel('Depth (mm)');title('Peak abs pressure over all time (Pa)');
exportgraphics(f,fullfile(outdir,'stability_diagnostics.png'),'Resolution',cfg.diagnostics.figure_dpi);close(f);
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(value,'PrettyPrint',true));
end
