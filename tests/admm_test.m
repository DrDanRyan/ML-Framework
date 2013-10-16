clear all
load ICU_ffn_data

%%
sampleProp = .8;
sampler = StratifiedSampler(sampleProp);
layer1Size = 64;
nMaxoutUnits = 4;
inputDropout = 0;
hiddenDropout = 0;

rho = 5;
rTol = 1e-8; % pretraining tolerance for residual, h1star - h1;

targets(targets==-1) = 0;

trainer = GradientTrainer();
reporter = ConsoleReporter();
trainer.reporter = reporter;
stepper = NesterovMomentum();
trainer.stepCalculator = stepper;
split_schedule = EarlyStopping(500, 'lr0', .002, ...
                               'lrDecay', 1, ...
                               'momentum', .6, ...
                               'burnIn', 30, ...
                               'lookAhead', 5);
                            
ffn_schedule = EarlyStopping(500, 'lr0', .2, ...
                               'lrDecay', 1, ...
                               'momentum', .8, ...
                               'burnIn', 30, ...
                               'lookAhead', 15);                          
                            
trainer.trainingSchedule = split_schedule;

bottomLayer = ComboOutputLayer(MaxoutHiddenLayer(187, layer1Size, nMaxoutUnits), ...
                                 MeanSquaredError());
topLayer = LogisticOutputLayer(layer1Size);                              
splitNet = SplitNetwork(bottomLayer, topLayer, rho);

trainer.model = splitNet;

outputs = gpuArray.zeros(1, 4000);

for i = 1:5
   % Create trainIdx, validIdx and testIdx
   testIdx = hold_outs{i};
   trainSplit = setdiff(1:4000, testIdx);
   [trainIdx, validIdx] = sampler.sample(trainSplit, targets(trainSplit));
   trainSize = length(trainIdx);
   trainIn = inputs(:,trainIdx);
   trainTargets = targets(:,trainIdx);
   
   % Pretrain on trainIdx using admm
   splitNet.reset();
   splitNet.h1star = .01*gpuArray.randn(layer1Size, trainSize, 'single');
   splitNet.u = gpuArray.zeros(layer1Size, trainSize, 'single');
   trainer.model = splitNet;
   residual = Inf;
   while residual > rTol
      % Train bottom layer
      stepper.reset();
      split_schedule.reset();
      trainer.trainingSchedule = split_schedule;
      trainer.reporter = [];
      trainer.dataManager = FullBatch(trainIn, [], trainIn, []);
      splitNet.modelState = 'bottom';
      trainer.train();
      
      % Train top layer
      stepper.reset();
      split_schedule.reset();
      trainer.dataManager = FullBatch([], trainTargets, [], trainTargets);
      splitNet.modelState = 'top';
      trainer.train();
      
      % Update u
      splitNet.update_u();
      residual = .5*mean(mean((splitNet.h1 - splitNet.h1star).^2));
      fprintf('Fold: %d \t Residual: %d \n', i, residual);
   end
   
   % Rescale matrices based on dropout or ELSE implement dropout in
   % SplitNetwork
   ffn = FeedForwardNet('inputDropout', inputDropout, 'hiddenDropout', hiddenDropout);
   ffn.hiddenLayers = {splitNet.bottomLayer.hiddenLayer};
   ffn.outputLayer = splitNet.topLayer;
   stepper.reset();
   reporter.reset();
   ffn_schedule.reset();
   trainer.trainingSchedule = ffn_schedule;
   trainer.reporter = reporter;
   trainer.dataManager = FullBatch(trainIn, trainTargets, inputs(:,validIdx), targets(validIdx));
   trainer.model = ffn;
   trainer.train();
   outputs(testIdx) = ffn.output(inputs(:,testIdx));
end

score = compute_event1(outputs, targets, true)