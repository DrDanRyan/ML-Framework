clear all
load cifar10_train

%% Define parameters
codeSize = 512;
%nRows = 8;
%nCols = 8;

validProp = 1/5;
nanNoise = .2;

nSteps = 10;
lam = 0;

validationInterval = 10;
batchSize = 12800;
trainLossSampleSize = 1e4;
nUpdates = 1e5;

%% Corrupted inputs
mask = gpuArray.rand(784,60000,'single') < nanNoise;
corruptedInputs = inputs;
corruptedInputs(mask) = NaN;

%% Setup model and trainer
ae = ImputingAutoEncoder('nSteps', nSteps, 'lam', lam);
enc = LogisticHiddenLayer(784, codeSize);
dec = LogisticOutputLayer(codeSize, 784);
ae.encodeLayer = enc;
ae.decodeLayer = dec;

trainer = ImputingGradientTrainer();
trainer.model = ae;
trainer.stepCalculator = AdaDelta();
trainer.progressMonitor = BasicMonitor('validationInterval', validationInterval);
trainer.progressMonitor.reporter = MNISTReporter(nRows, nCols);
trainer.dataManager = ImputingDataManager({corruptedInputs}, ...
                                  {}, ...
                                  'batchSize', batchSize, ...
                                  'trainLossSampleSize', trainLossSampleSize);

%% Train model
trainer.train(nUpdates);