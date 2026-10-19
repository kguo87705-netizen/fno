function analyze_reference(run_dir,figure_dir)
% No additional wave propagation. Examine the saved reference only.
set(groot,'defaultFigureVisible','off');
file=fullfile(run_dir,'worker_1','fmc.mat');
s=load(file,'fmc','txids','t_output','cfg','meta');
assert(isequal(s.txids,1:s.cfg.array.elements));
f=double(s.fmc); assert(isequal(size(f),[64,64,1000]) && all(isfinite(f(:))));
ft=permute(f,[2,1,3]);delta=f-ft;
nr=sqrt(sum(f.^2,3));nd=sqrt(sum(delta.^2,3));den=(nr+nr')/2;
relative=nd./max(den,realmin); upper=triu(true(size(relative)),1);
v=relative(upper);sorted=sort(v);p95=interp1(1:numel(sorted),sorted,1+0.95*(numel(sorted)-1));
metrics=struct;
metrics.global_relative_l2=norm(delta(:))/norm(f(:));
metrics.pair_count=numel(v);metrics.pair_mean=mean(v);metrics.pair_median=median(v);
metrics.pair_p95=p95;metrics.pair_max=max(v);
metrics.pair_denominator='mean of the two uncentered L2 trace norms';
metrics.max_abs_difference_pa=max(abs(delta(:)));
metrics.min_pair_mean_norm=min(den(upper));
[all_i,all_j]=find(upper);[~,worst]=max(v);
metrics.worst_pair=[all_i(worst),all_j(worst)];
old=load(s.cfg.paths.single_tx_mat,'p_output');
f32=squeeze(s.fmc(s.cfg.array.transmitter_index_one_based,:,:));
metrics.tx32_vs_previous_relative_l2=norm(double(f32(:))-double(old.p_output(:)))/norm(double(old.p_output(:)));
metrics.reference_bytes=dir(file).bytes;
mf=matfile(file);assert(isequal(size(mf,'fmc'),[64,64,1000]));metrics.disk_reopen_passed=true;
write_json(fullfile(run_dir,'reciprocity.json'),metrics);
fig=figure('Position',[80,80,1150,450]);tiledlayout(1,2,'TileSpacing','compact');
nexttile;imagesc(log10(max(relative,1e-10)));axis image;colorbar;
xlabel('Receiver j');ylabel('Transmitter i');title('log10 symmetric relative L2 reciprocity error');
nexttile;histogram(log10(max(v,1e-10)),40);grid on;
xlabel('log10 pair relative L2 error');ylabel('Number of off-diagonal pairs');
title(sprintf('2016 pairs; median %.3g, max %.3g',metrics.pair_median,metrics.pair_max));
exportgraphics(fig,fullfile(figure_dir,'reference_reciprocity.png'),'Resolution',150);close(fig);
pairs=[1,64;13,39;metrics.worst_pair];
fig=figure('Position',[80,80,1100,800]);tiledlayout(3,1,'TileSpacing','compact');
for q=1:size(pairs,1)
 i=pairs(q,1);j=pairs(q,2);nexttile;
 plot(s.t_output*1e6,squeeze(f(i,j,:)),'-','LineWidth',1);hold on;
 plot(s.t_output*1e6,squeeze(f(j,i,:)),'--','LineWidth',1);grid on;
 xlabel('Time from acquisition start (us)');ylabel('Pressure (Pa)');
 title(sprintf('Tx%d-Rx%d versus Tx%d-Rx%d; rel. L2 %.3g',i,j,j,i,relative(i,j)));
 legend('F(i,j)','F(j,i)','Location','best');
end
exportgraphics(fig,fullfile(figure_dir,'reference_reciprocal_traces.png'),'Resolution',150);close(fig);
fig=figure('Position',[80,80,1000,400]);plot(s.txids,s.meta.solver_s,'o-');grid on;
xlabel('Transmitter index');ylabel('Solver wall time (s)');title('64 sequential transmissions: actual per-shot time');
exportgraphics(fig,fullfile(figure_dir,'reference_shot_timings.png'),'Resolution',150);close(fig);
fprintf('RECIPROCITY_GLOBAL=%.9g MAX_PAIR=%.9g TX32_MATCH=%.9g\n', ...
 metrics.global_relative_l2,metrics.pair_max,metrics.tx32_vs_previous_relative_l2);
end

function write_json(path,data)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);closer=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(data,'PrettyPrint',true));
end
