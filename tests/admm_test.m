clear all
load ICU_ffn_data

%%
sampleProp = .85;
sampler = StratifiedSampler(sampleProp);
layer1Size = 16;
nMaxoutUnits = 2;
bottomDropout = .2;
topDropout = .5;
rho = 1;
targets(targets==-1) = 0;

rTol = 1e-10;

bottomEpochs = 30;
bottomLR = .2;
bottomMomentum = .8;

topEpochs = 30;
topLR = .1;
topMomentum = .8;

trainer = GradientTrainer();
stepper = NesterovMomentum();
trainer.stepCalculator = stepper;

bottom_schedule = BasicMomentumSchedule(bottomEpochs, bottomLR, bottomMomentum);
top_schedule = BasicMomentumSchedule(topEpochs, topLR, topMomentum);

bottomNet = FeedForwardNet('inputDropout', bottomDropout);
bottomNet.hiddenLayers = {MaxoutHiddenLayer(187, 64, nMaxoutUnits)};
bottomNet.outputLayer = ComboOutputLayer(MaxoutHiddenLayer(64, layer1Size, nMaxoutUnits), ...
                                 MeanSquaredError());
topNet = FeedForwardNet('inputDropout', topDropout);
topNet.outputLayer = LogisticOutputLayer(layer1Size);                              
splitNet = SplitNetwork(bottomNet, topNet, rho);
trainer.model = splitNet;

outputs = gpuArray.zeros(1, 4000);
for i = 1:5
   % Create trainIdx, validIdx and testIdx
   testIdx = hold_outs{i};
   trainSplit = setdiff(1:4000, testIdx);
   [trainIdx, validIdx] = sampler.sample(trainSplit, targets(trainSplit));
   trainSize = length(trainIdx);
   
   bottom_dataManager = FullBatch(inputs(:,trainIdx), [], [], []);
   top_dataManager = FullBatch([], targets(:,trainIdx), [], []);
   
   % Pretrain on trainIdx using admm
   splitNet.reset();
   splitNet.h1star = .01*gpuArray.randn(layer1Size, trainSize, 'single');
   splitNet.u = gpuArray.zeros(layer1Size, trainSize, 'single');
   residual = Inf;
   while residual > rTol
      % Train bottom layer
      stepper.reset();
      bottom_schedule.reset();
      trainer.trainingSchedule = bottom_schedule;
      trainer.dataManager = bottom_dataManager;
      splitNet.modelState = 'bottom';
      trainer.train();
      
      % Train top layer
      stepper.reset();
      bottom_schedule.reset();
      trainer.dataManager = top_dataManager;
      splitNet.modelState = 'top';
      trainer.train();
      
      % Update u
      splitNet.update_u();
      residual = .5*mean(mean((splitNet.h1 - splitNet.h1star).^2));
      
      % Compute validationLoss
      splitNet.modelState = 'full';
      trainOuts = splitNet.output(inputs(:,trainIdx));
      trainingLoss = splitNet.compute_loss(trainOuts, targets(:,trainIdx));
      validOuts = splitNet.output(inputs(:,validIdx));
      validationLoss = splitNet.compute_loss(validOuts, targets(:,validIdx));
      fprintf('Fold: %d   Residual: %.4f   train: %.4f   valid: %.4f\n', i, residual, trainingLoss, validationLoss);
   end

   splitNet.modelState = 'full';
   outputs(testIdx) = splitNet.output(inputs(:,testIdx));
end

score = compute_event1(outputs, targets, true)