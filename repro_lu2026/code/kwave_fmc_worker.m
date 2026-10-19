function kwave_fmc_worker(config_path,outdir,txids,barrier)
% Independent process on the shared GPU. No automatic fallback or retry.
cfg=jsondecode(fileread(config_path));
assert(isfolder(outdir));
write_json(fullfile(outdir,'process.json'),struct('pid',feature('getpid')));
try
    addpath(cfg.paths.kwave); rng(cfg.seed,'twister');
    src=load(cfg.paths.single_tx_mat,'c','label','source_wavelet','t_internal','t_output');
    c=src.c; label=src.label; source_wavelet=src.source_wavelet;
    if isfield(cfg,'diagnostic') && isfield(cfg.diagnostic,'uniform_speed_m_s')
        c(:)=cfg.diagnostic.uniform_speed_m_s; label(:)=cfg.diagnostic.uniform_speed_m_s;
    end
    assert(isequal(size(c),cfg.grid.shape(:)'));
    k=kWaveGrid(size(c,1),cfg.grid.spacing_m,size(c,2),cfg.grid.spacing_m);
    k.setTime(cfg.time.internal_count,cfg.time.internal_dt_s);
    t_internal=double(k.t_array);
    assert(isequal(t_internal,src.t_internal));
    stride=round(cfg.time.output_dt_s/cfg.time.internal_dt_s);
    t_output=t_internal(1:stride:end);
    medium.sound_speed=c; medium.sound_speed_ref=cfg.model.c_background_m_s;
    medium.density=cfg.model.density_kg_m3;
    xs=cfg.array.center_x_m+((0:cfg.array.elements-1)-(cfg.array.elements-1)/2)*cfg.array.pitch_m;
    xi=round(xs/cfg.grid.spacing_m)+1; zi=round(cfg.array.z_m/cfg.grid.spacing_m)+1;
    rx=sub2ind(size(c),repmat(zi,1,numel(xi)),xi);
    sensor.mask=false(size(c)); sensor.mask(rx)=true; sensor.record={'p'};
    [ok,order]=ismember(rx,find(sensor.mask));assert(all(ok));
    source.p=single(source_wavelet); source.p_mode=cfg.source.mode;
    source.p_mask=false(size(c));
    parallel.gpu.enableCUDAForwardCompatibility(cfg.compute.cuda_forward_compatibility);
    g=gpuDevice; a=gpuArray.ones(64,64,'single'); b=fft2(a); wait(g); clear a b;
    write_json(fullfile(outdir,'ready.json'),struct('gpu',g.Name,'pid',feature('getpid')));
    if ~isempty(barrier)
        timeout=tic;
        while ~isfile(barrier)
            assert(toc(timeout)<cfg.benchmark.barrier_timeout_s,'Barrier timeout');pause(0.1);
        end
    end
    meta=struct('transmitters',txids,'gpu',g.Name,'solver_s',zeros(1,numel(txids)));
    meta.started_local=char(datetime('now','Format','yyyy-MM-dd HH:mm:ss.SSS'));
    work=tic;
    fmc=zeros(numel(txids),cfg.array.elements,cfg.time.output_count,'single');
    fmc_internal=zeros(numel(txids),cfg.array.elements,cfg.time.internal_count,'single');
    shotsdir=fullfile(outdir,'shots');mkdir(shotsdir);
    for q=1:numel(txids)
        tx=txids(q); source.p_mask(:)=false;source.p_mask(rx(tx))=true;
        wait(g);clock=tic;
        d=kspaceFirstOrder2D(k,medium,source,sensor, ...
            'PMLInside',cfg.grid.pml_inside,'PMLSize',cfg.grid.pml_size, ...
            'PMLAlpha',cfg.grid.pml_alpha,'DataCast',cfg.compute.data_cast, ...
            'DataRecast',false,'PlotSim',false,'Smooth',false);
        wait(g);meta.solver_s(q)=toc(clock);
        assert(isa(d.p,'gpuArray'),'GPU output expected');
        p_internal=single(gather(d.p(order,:)));assert(all(isfinite(p_internal(:))));
        p=p_internal(:,1:stride:end);
        fmc(q,:,:)=reshape(p,[1,size(p)]);
        fmc_internal(q,:,:)=reshape(p_internal,[1,size(p_internal)]);
        solver_s=meta.solver_s(q);
        save(fullfile(shotsdir,sprintf('tx_%03d.mat',tx)),'p','p_internal','tx','solver_s','-v7.3');
        fprintf('PROGRESS_TX=%d COMPLETE=%d/%d SOLVER_S=%.6f ELAPSED_S=%.3f\n', ...
            tx,q,numel(txids),solver_s,toc(work));
    end
    meta.compute_and_checkpoint_s=toc(work);
    meta.finished_local=char(datetime('now','Format','yyyy-MM-dd HH:mm:ss.SSS'));
    meta.all_finite=all(isfinite(fmc(:)));
    meta.solver_total_s=sum(meta.solver_s);meta.solver_mean_s=mean(meta.solver_s);
    meta.solver_median_s=median(meta.solver_s);
    save(fullfile(outdir,'fmc.mat'),'fmc','fmc_internal','txids','t_output','t_internal', ...
        'label','c','source_wavelet','cfg','meta','-v7.3');
    meta.work_with_final_save_s=toc(work);
    write_json(fullfile(outdir,'metrics.json'),meta);
    fprintf('WORKER_OK=1\n');
catch ME
    fid=fopen(fullfile(outdir,'failure.txt'),'w','n','UTF-8');
    if fid>=0,fprintf(fid,'%s',getReport(ME,'extended','hyperlinks','off'));fclose(fid);end
    rethrow(ME);
end
end

function write_json(path,data)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);closer=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(data,'PrettyPrint',true));
end
