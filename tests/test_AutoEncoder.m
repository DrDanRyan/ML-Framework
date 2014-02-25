%% Train a basic AutoEncoder on MNIST
clear all
close all
load cifar10_train

%% Define model parameters
nUpdates = 1e5;
codeSize = 256;

nRows = 12;
nCols = 12;

lr0 = .001;
rho = .9;

validationInterval = 10;
batchSize = 128;
trainLossSampleSize = 1e4;

%% Setup model and trainer
encoder = TRecHiddenLayer(3072, codeSize, 'initScale', .05);
d1 = LinearZeroBiasHiddenLayer(codeSize, 3072);
d2 = MeanSquaredError();
decoder = ComboOutputLayer(d1, d2);

ae = AutoEncoder('isTiedWeights', true);
ae.encodeLayer = encoder;
ae.decodeLayer = decoder;

trainer = GradientTrainer();
trainer.model = ae;
trainer.stepCalculator = AdaDelta();
%trainer.parameterSchedule = MomentumSchedule(lr0, rho);
trainer.progressMonitor = BasicMonitor('validationInterval', validationInterval);
trainer.progressMonitor.reporter = Cifar10Reporter(nRows, nCols);
trainer.dataManager = DataManager({inputs}, {}, 'batchSize', batchSize, ...
                                  'trainLossSampleSize', trainLossSampleSize);
                               
%% Train
trainer.train(nUpdates);