
warning('off', 'all');
addpath('C:/Users/xushaobo/easyspin/EasySpin-main/easyspin');
addpath('C:/Users/xushaobo/easyspin/EasySpin-main/easyspin/private');

Sys.S = 1;
Sys.D = [850 75];
Sys.g = 2.0023;
Sys.lwpp = [0.0 1.2];

Exp.mwFreq = 9.5;
Exp.Range = [260 440];
Exp.nPoints = 2048;

Opt.Verbosity = 0;
[B, spc] = pepper(Sys, Exp, Opt);
rng(42);
spc = spc + 0.003 * max(abs(spc)) * randn(size(spc));

data = [B(:) spc(:)];
csvwrite('C:/Users/xushaobo/radical-synthesis-workflow/data/experimental/demo_zfs_experimental.csv', data);
fprintf('Demo data saved: %d points\n', numel(B));
