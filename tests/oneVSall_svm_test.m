clear all
load reduced_MNIST

targets(targets == 0) = -1;
testTargets(testTargets == 0) = -1;

ffn = FeedForwardNet('inputDropout', 0, 'hiddenDropout', .3);
ffn.hiddenLayers = {MaxoutHiddenLayer(717, 1024, 5, 'initScale', .005)};
ffn.outputLayer = SVMOutputLayer(1024, 10, 'L2Penalty', 1e-6, 'initScale', 0);

sampler = ProportionSubsampler(5/6);
[trainIdx, validIdx] = sampler.sample(1:60000);
trainInputs = inputs(:,trainIdx);
trainTargets = targets(:, trainIdx);
validInputs = inputs(:,validIdx);
validTargets = targets(:,validIdx);
clear inputs targets

trainer = GradientTrainer();
trainer.dataManager = DataManager({trainInputs, trainTargets}, {validInputs, validTargets}, ...
                                    'batchSize', 128);
trainer.stepCalculator = NesterovMomentum();
trainer.model = ffn;
trainer.parameterSchedule = MomentumSchedule(.03, .99, 'lrDecay', .9999, 'C', 500);
trainer.progressMonitor = EarlyStopping(3, 1.2, 'validationInterval', 500, ...
                                              'isComputeTrainLoss', false, ...
                                              'validLossFunction', @MNIST_validLossFunction);
trainer.train(30000);

%% Fine-tune with smaller learning rate and momentum
trainer.parameterSchedule = MomentumSchedule(.001, .5);
trainer.progressMonitor.bufferConst = 7;
trainer.train(5000);

%% Evaluate on test set
dummy = struct();
dummy.validationData = {testInputs, testTargets};
compute_MNIST_errors(trainer.progressMonitor.models{1}, dummy)