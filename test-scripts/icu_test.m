clear all
close all
load Citi_data

%% parameters
Layer1 = 500;
Layer2 = 300;

nFolds = 5;

learnRate = .1;
minRate = 1e-7;
maxRate = 50;
upFactor = 1.2;
downFactor = .5;
L1Penalty = 0;
L2Penalty = 1e-3;
nEpochs = 1000;

maxoutLayers = 2;
inputDropout = .4;

momentum = .85;
%lrDecay = .998; 


 
%% Initialize trainer
trainer = GradientTrainer();
trainer.stepCalculator = NesterovMomentum();
% trainer.stepCalculator = IRprop(learnRate, 'maxRate', maxRate, ...
%                                            'minRate', minRate, ...
%                                            'upFactor', upFactor, ...
%                                            'downFactor', downFactor);
trainer.reporter = ConsoleReporter();
trainer.trainingSchedule = StepDownMomentum(learnRate, momentum, nEpochs);

%% Initialize nnets and preprocessors
nnet = FeedForwardNet('inputDropout', inputDropout);
nnet.hiddenLayers = {MaxoutHiddenLayer(187, Layer1, maxoutLayers, 'L2Penalty', L2Penalty), ...
                     MaxoutHiddenLayer(Layer1, Layer2, maxoutLayers, ...
                                                'initType', 'sparse', ...
                                                'initScale', 15, ...
                                                'L2Penalty', L2Penalty)};

nnet.outputLayer = SVMOutputLayer(Layer2, 'hingeExp', 2, ...
                                          'L1Penalty', L1Penalty, ...
                                          'L2Penalty', L2Penalty, ...
                                          'costRatio', 4);
targets(targets == 0) = -1;

%% Train base learners on CV partition
hold_outs = cross_validate_partition(4000, nFolds);
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

