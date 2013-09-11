clear all
load Citi_data

%% parameters
N = 1;
nDims = 187;
Layer1 = 1000;
Layer2 = 2000;
nnets = cell(1, N);
preprocessors = cell(1, N);
nFolds = 4;
batchSize = 3000;
sampleSize = 3000;
learnRate = .1;
momentum = .95;
nEpochs = 100;
lrDecay = .9931; 
 
%% Initialize trainer
trainer = GradientTrainer();
trainer.stepCalculator = Rprop(learnRate);
trainer.reporter = ConsoleReporter();
trainer.trainingSchedule = ExpDecaySchedule(learnRate, momentum, nEpochs, lrDecay);

%% Initialize nnets and preprocessors
for i = 1:N
   nnets{i} = FeedForwardNet('dropout', false);
   nnets{i}.hiddenLayers = {ReluHiddenLayer(nDims, Layer1), ...
                               ReluHiddenLayer(Layer1, Layer2, 'initType', 'sparse', ...
                                                               'initScale', 15)};
   nnets{i}.outputLayer = LogisticOutputLayer(Layer2);

   %preprocessors{i} = RandomProjection(187, nDims);
   preprocessors{i} = Identity();
end

%% Train base learners on CV partition
hold_outs = cross_validate_partition(4000, nFolds);
gpuState = GPUState();
outputs = gpuState.zeros(N, 4000);

for foldIdx = 1:nFolds
   
   fprintf('\nFold %d Beginning.\n', foldIdx)
   validIdx = hold_outs{foldIdx};
   trainIdx = setdiff(1:4000, validIdx);
   
   trainInputs = single(gpuArray(data(:, trainIdx)));
   trainTargets = single(gpuArray(targets(:, trainIdx)));
   validInputs = single(gpuArray(data(:, validIdx)));
   validTargets = single(gpuArray(targets(:, validIdx)));
   
   for netIdx = 1:N
      sampleIdx = randsample(length(trainIdx), sampleSize)';
      tInputs = preprocessors{netIdx}.transform(trainInputs(:, sampleIdx));
      tTargets = trainTargets(:, sampleIdx);
      vInputs = preprocessors{netIdx}.transform(validInputs); 
      trainer.dataManager = BasicDataManager(batchSize, tInputs, tTargets, ...
                                             vInputs, validTargets);
      trainer.model = nnets{netIdx};
      trainer.reset();
      trainer.train();
      outputs(netIdx, validIdx) = nnets{netIdx}.output(vInputs);
      fprintf('\n');
   end
   
end

%% Determine threshold value that yields the best score
outputs = gather(outputs);
for j = 1:N
   event1 = Event1Score(outputs(j,:), targets);
   lemeshow = Lemeshow(outputs(j,:), targets);
   scores(j,:) = [event1, lemeshow]; %#ok<SAGROW>

   fprintf('\n\n Max Score: %.4f \t Lemeshow %.4f \n', ...
              event1, lemeshow);
end

means = mean(outputs, 1);
means = min(.99, max(.01, means));
mean_scores = [Event1Score(means, targets), Lemeshow(means, targets)]
