clear all
load Citi_data

%% parameters
N = 10;
nDims = 90;
Layer1 = 20;
Layer2 = 20;
nnets = cell(1, N);
preprocessors = cell(1, N);

nFolds = 5;
sampleSize = 2400;

learnRate = .01;
minRate = 1e-7;
maxRate = 50;
upFactor = 1.2;
downFactor = .5;
nEpochs = 60;

maxoutLayers = 5;
inputDropout = .1;

%momentum = .85;
%lrDecay = .998; 


 
%% Initialize trainer
trainer = GradientTrainer();
trainer.stepCalculator = IRprop(learnRate, 'maxRate', maxRate, ...
                                           'minRate', minRate, ...
                                           'upFactor', upFactor, ...
                                           'downFactor', downFactor);
trainer.reporter = ConsoleReporter();
trainer.trainingSchedule = BasicMomentumSchedule(learnRate, 0, nEpochs);

%% Initialize nnets and preprocessors
for i = 1:N
   nnets{i} = FeedForwardNet('inputDropout', inputDropout);
   nnets{i}.hiddenLayers = {MaxoutHiddenLayer(nDims, Layer1, maxoutLayers), ...
                               MaxoutHiddenLayer(Layer1, Layer2, maxoutLayers, ...
                                                   'initType', 'sparse', ...
                                                   'initScale', 15)};
   nnets{i}.outputLayer = SVMOutputLayer(Layer2, 'hingeExp', 2, 'L2Penalty', .001);

   preprocessors{i} = RandomProjection(187, nDims);
   %preprocessors{i} = Identity();
end
targets(targets == 0) = -1;

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
      trainer.dataManager = FullBatch(tInputs, tTargets, vInputs, validTargets);
      trainer.model = nnets{netIdx};
      trainer.reset();
      trainer.train();
      outputs(netIdx, validIdx) = nnets{netIdx}.output(vInputs);
      fprintf('\n');
   end
   
end

%% Determine threshold value that yields the best score
outputs = gather(outputs);
targets(targets == -1) = 0;
for j = 1:N
   event1 = Event1Score(outputs(j,:), targets);
   lemeshow = Lemeshow(outputs(j,:), targets);
   scores(j,:) = [event1, lemeshow]; %#ok<SAGROW>

   fprintf('\n\n Max Score: %.4f \t Lemeshow %.4f \n', ...
              event1, lemeshow);
end

% means = mean(outputs, 1);
% means = min(.99, max(.01, means));
% mean_scores = [Event1Score(means, targets), Lemeshow(means, targets)];
% fprintf('\n\n Ensemble Scores:    Event1 %.4f    Lemeshow %.4f \n', ...
%             mean_scores(1), mean_scores(2));
         
%% Stack new net on top using outputs as new trianing set
outputs = normalize(outputs);
hold_outs = cross_validate_partition(4000, nFolds);
stacked_outs = gpuState.zeros(1, 4000);
stacked_nnet = FeedForwardNet('dropout', true, 'inputDropout', 0);
stacked_nnet.hiddenLayers = {TanhHiddenLayer(N, 100)};
stacked_nnet.outputLayer = LogisticOutputLayer(100);

trainer = GradientTrainer();
trainer.stepCalculator = IRprop(.01);
trainer.reporter = ConsoleReporter();
trainer.trainingSchedule = BasicMomentumSchedule(.01, 0, 30);
trainer.model = stacked_nnet;

for foldIdx = 1:nFolds
   
   fprintf('\nFold %d Beginning.\n', foldIdx)
   validIdx = hold_outs{foldIdx};
   trainIdx = setdiff(1:4000, validIdx);
   
   trainInputs = single(gpuArray(outputs(:, trainIdx)));
   trainTargets = single(gpuArray(targets(:, trainIdx)));
   validInputs = single(gpuArray(outputs(:, validIdx)));
   validTargets = single(gpuArray(targets(:, validIdx)));
   
   trainer.dataManager = FullBatch(trainInputs, trainTargets, validInputs, validTargets);
   trainer.reset();
   trainer.train();
   stacked_outs(validIdx) = stacked_nnet.output(validInputs);
   fprintf('\n');
   
end

stacked_outs = gather(stacked_outs);
stackedEvent1 = Event1Score(stacked_outs, targets, true)
stackedLemeshow = Lemeshow(stacked_outs, targets, true)