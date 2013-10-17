clear all
load ICU_ffn_data

sampleProp = .85;
sampler = StratifiedSampler(sampleProp);
layer1Size = 2;
nMaxoutUnits = 8;
bottomDropout = 0;
topDropout = 0;
rho = 1;
targets(targets==-1) = 0;

bottomEpochs = 200;
bottomLR = .05;
bottomMomentum = .4;

topEpochs = 200;
topLR = .05;
topMomentum = .4;

trainer = GradientTrainer();
stepper = NesterovMomentum();
trainer.stepCalculator = stepper;

bottom_schedule = BasicMomentumSchedule(bottomEpochs, bottomLR, bottomMomentum);
top_schedule = BasicMomentumSchedule(topEpochs, topLR, topMomentum);
full_schedule = EarlyStopping(50, 'burnIn', 5, 'lookAhead', 3);

bottomNet = FeedForwardNet('inputDropout', bottomDropout);
bottomNet.outputLayer = ComboOutputLayer(MaxoutHiddenLayer(187, layer1Size, nMaxoutUnits), ...
                                 MeanSquaredError());
topNet = FeedForwardNet('inputDropout', topDropout);
topNet.outputLayer = LogisticOutputLayer(layer1Size);                              
splitNet = SplitNetwork(bottomNet, topNet, rho);
trainer.model = splitNet;

codes = gpuArray.zeros(layer1Size, 4000);
outputs = gpuArray.zeros(1, 4000);
for i = 1:5
   % Create trainIdx, validIdx and testIdx
   testIdx = hold_outs{i};
   trainSplit = setdiff(1:4000, testIdx);
   [trainIdx, validIdx] = sampler.sample(trainSplit, targets(trainSplit));
   trainSize = length(trainIdx);
   
   bottom_dataManager = FullBatch(inputs(:,trainIdx), [], [], []);
   top_dataManager = FullBatch([], targets(:,trainIdx), [], []);
   full_schedule.reset();
   
   % Pretrain on trainIdx using admm
   splitNet.reset();
   splitNet.h1star = .01*gpuArray.randn(layer1Size, trainSize, 'single');
   splitNet.u = gpuArray.zeros(layer1Size, trainSize, 'single');
   isContinue = true;
   while isContinue
      % Train bottom layer
      stepper.reset();
      bottom_schedule.reset();
      trainer.trainingSchedule = bottom_schedule;
      trainer.dataManager = bottom_dataManager;
      splitNet.modelState = 'bottom';
      trainer.train();
      splitNet.set_h1(inputs(:,trainIdx));
      
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
      isContinue = full_schedule.update([], [], validationLoss);
      fprintf('F: %d  Res: %d  T: %.4f   V: %.4f\n', i, residual, trainingLoss, validationLoss);
   end

   splitNet.modelState = 'full';
   codes(:,testIdx) = splitNet.bottomNet.output(inputs(:,testIdx));
   outputs(testIdx) = splitNet.output(inputs(:,testIdx));
end

score = compute_event1(outputs, targets)
lemeshow = compute_Lemeshow(outputs, targets, true)

posPts = codes(:,targets==1);
negPts = codes(:,targets==0);
figure()
plot(posPts(1,:), posPts(2,:), 'r+');
hold on
plot(negPts(1,:), negPts(2,:), 'bo');