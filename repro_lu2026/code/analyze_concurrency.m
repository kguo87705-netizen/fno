function analyze_concurrency(reference_dir,group_dirs,out_json,figure_dir)
% Validate concurrent outputs against the reference before comparing speed.
set(groot,'defaultFigureVisible','off');
r=load(fullfile(reference_dir,'worker_1','fmc.mat'),'fmc');
records=cell(1,numel(group_dirs));
for q=1:numel(group_dirs)
 summary=jsondecode(fileread(fullfile(group_dirs{q},'summary.json')));
 workers=dir(fullfile(group_dirs{q},'worker_*'));errors=[];shots=[];
 for w=1:numel(workers)
  s=load(fullfile(workers(w).folder,workers(w).name,'fmc.mat'),'fmc','txids');
  for k=1:numel(s.txids)
   ref=double(squeeze(r.fmc(s.txids(k),:,:)));now=double(squeeze(s.fmc(k,:,:)));
   errors(end+1)=norm(now(:)-ref(:))/norm(ref(:)); %#ok<AGROW>
  end
  shots=[shots,s.txids]; %#ok<AGROW>
 end
 assert(numel(shots)==summary.shots && numel(unique(shots))==numel(shots));
 summary.max_relative_l2_vs_reference=max(errors);
 summary.mean_relative_l2_vs_reference=mean(errors);
 records{q}=summary;
end
base=records{1}.seconds_per_shot;
for q=1:numel(records),records{q}.speedup_vs_1process=base/records{q}.seconds_per_shot;end
fid=fopen(out_json,'w','n','UTF-8');assert(fid>=0);fprintf(fid,'%s',jsonencode(records,'PrettyPrint',true));fclose(fid);
fig=figure('Position',[80,80,1100,420]);tiledlayout(1,2,'TileSpacing','compact');
counts=cellfun(@(x)x.processes,records);seconds=cellfun(@(x)x.seconds_per_shot,records);
memory=cellfun(@(x)x.gpu_peak_whole_card_MiB,records);
nexttile;bar(counts,seconds);xlabel('MATLAB processes');ylabel('Makespan / shots (s)');
title('Same 6 shots: compute and checkpoint time');grid on;
nexttile;bar(counts,memory/1024);xlabel('MATLAB processes');ylabel('Whole-GPU sampled peak (GiB)');
title('Includes desktop and other GPU users');grid on;
exportgraphics(fig,fullfile(figure_dir,'concurrency_benchmark.png'),'Resolution',150);close(fig);
end
