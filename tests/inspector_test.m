%% Load trained net
clear all
close all
load trained_ffn2 % loads ffn and inputSize

%% 
layerIdx = 2;
lr = .1;
rho = .95;
focusIdx = 2;

inspector = FFNInspector(ffn, layerIdx, inputSize);
inspector.focus(focusIdx);

stepCalculator = NesterovMomentum();
parameterSchedule = MomentumSchedule(lr, rho);

trainer = Trainer();
trainer.model = inspector;
trainer.stepCalculator = stepCalculator;
trainer.parameterSchedule = parameterSchedule;

%%
trainer.train(100);

%%
xMax = gather(inspector.xMax);
colormap gray
image(reshape(xMax', 28, 28), 'CDataMapping', 'scaled');
