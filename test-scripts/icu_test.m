clear all
close all
load Citi_data

%% parameters
Layer1 = 100;
Layer2 = 100;
Layer3 = 100;

nFolds = 5;

learnRate = .03;
momentum = .85;

minRate = 1e-7;
maxRate = 50;
upFactor = 1.2;
downFactor = .5;

L1Penalty = 0;
L2Penalty = 0;
nEpochs = 1000;

maxoutLayers = 3;
hiddenDropout = .2;
inputDropout = .1;


%lrDecay = .998; 

% maxCuts = 0;
% cutFactor = .25;
% lookAhead = 40;
 
%% Initialize trainer
trainer = GradientTrainer();
trainer.stepCalculator = NesterovMomentum();
% trainer.stepCalculator = IRprop(learnRate, 'maxRate', maxRate, ...
%                                            'minRate', minRate, ...
%                                            'upFactor', upFactor, ...
%                                            'downFactor', downFactor);
trainer.reporter = ConsoleReporter();
trainer.trainingSchedule = EarlyStopping(learnRate, momentum, nEpochs, ...
                                                'burnIn', 100, ... 
                                                'lookAhead', 20);

%% Initialize nnets and preprocessors
nnet = FeedForwardNet('inputDropout', inputDropout, 'hiddenDropout', hiddenDropout);
nnet.hiddenLayers = {MaxoutHiddenLayer(187, Layer1, maxoutLayers), ...
                     MaxoutHiddenLayer(Layer1, Layer2, maxoutLayers, ...
                                                'initType', 'sparse', ...
                                                'initScale', 15), ...
                     MaxoutHiddenLayer(Layer2, Layer3, maxoutLayers, ...
                                                'initType', 'sparse', ...
                                                'initScale', 15)};

nnet.outputLayer = SVMOutputLayer(Layer3, 'hingeExp', 2, ...
                                          'L1Penalty', L1Penalty, ...
                                          'L2Penalty', L2Penalty, ...
                                          'costRatio', 2);
targets(targets == 0) = -1;

%% Train base learners on CV partition
hold_outs = stratified_CV_partition(targets, nFolds);
gpuState = GPUState();
outputs = gpuState.zeros(1, 4000);

for foldIdx = 1:nFolds
   
   fprintf('\nFold %d Beginning.\n', foldIdx)
   validIdx = hold_outs{foldIdx};
   trainIdx = setdiff(1:4000, validIdx);
   
   trainInputs = single(gpuArray(data(:, trainIdx)));
   trainTargets = single(gpuArray(targets(:, trainIdx)));
   validInputs = single(gpuArray(data(:, validIdx)));
   validTargets = single(gpuArray(targets(:, validIdx)));
   trainer.dataManager = FullBatch(trainInputs, trainTargets, validInputs, validTargets);
   trainer.model = nnet;
   trainer.reset();
   trainer.train();
   outputs(validIdx) = nnet.output(validInputs);
   fprintf('\n');
end

%% Determine threshold value that yields the best score
outputs = gather(outputs);
targets(targets == -1) = 0;
event1 = Event1Score(outputs, targets, true);
lemeshow = Lemeshow(outputs, targets);
fprintf('\n\n Max Score: %.4f \t Lemeshow %.4f \n', ...
              event1, lemeshow);

