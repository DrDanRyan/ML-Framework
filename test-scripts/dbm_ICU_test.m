clear all
load ICU_nnet_data
targets = single(gpuArray(targets));
data = single(gpuArray(data));

ensembleSize = 20;
models = cell(1, ensembleSize);
layer1 = LogisticHiddenLayer(237, 75, 'maxFanIn', 100);
layer1.params = {initMats.W{1}, initMats.b{1}};
layer2 = LogisticHiddenLayer(75, 50, 'maxFanIn', 100);
layer2.params = {initMats.W{2}, initMats.b{2}};
for j = 1:ensembleSize
   models{j} = FeedForwardNet();
   models{j}.hiddenLayers = {layer1.copy(), layer2.copy()};
   models{j}.outputLayer = LogisticOutputLayer(50);
end

%%
stepper = NesterovMomentum();
schedule = EarlyStopping(5000, 'lr0', 2.8, 'momentum', .85, ...
                        'lrDecay', .993, 'burnIn', 50, 'lookAhead', 40);
                     
dataManager = FullBatch();

trainer = GradientTrainer();
trainer.dataManager = dataManager;
trainer.stepCalculator = stepper;
trainer.trainingSchedule = schedule;
%trainer.reporter = ConsoleReporter();

sampler = StratifiedSampler(.8);
nFolds = 5;
hold_outs = CV_partition(4000, nFolds);
outputs = gpuArray.zeros(ensembleSize, 4000, 'single');
scores = zeros(ensembleSize, 1);
for i = 1:nFolds
   testSplit = hold_outs{i};
   trainSplit = setdiff(1:4000, testSplit);
   fprintf('Fold %d Starting... \n', i);
   for j = 1:ensembleSize
      fprintf('Model %d training... \n', j);
      trainIdx = sampler.sample(trainSplit, targets(trainSplit));
      validIdx = setdiff(trainSplit, trainIdx);
      dataManager.trainingInputs = data(:, trainIdx);
      dataManager.trainingTargets = targets(trainIdx);
      dataManager.validationInputs = data(:, validIdx);
      dataManager.validationTargets = targets(validIdx);
      trainer.model = models{j};
      trainer.reset();
      trainer.train();
      outputs(j, testSplit) = trainer.model.output(data(:, testSplit));
   end
   fprintf('\n\n');
end

for j = 1:ensembleSize
   scores(j,1) = compute_event1(gather(outputs(j,:)), gather(targets));
   scores(j,2) = compute_Lemeshow(gather(outputs(j,:)), gather(targets));
end
scores
