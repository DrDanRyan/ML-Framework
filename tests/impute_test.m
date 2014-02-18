clear all
load MNIST_train

%% Define parameters
codeSize = 64;
validProp = 1/6;
nanNoise = .2;
tol = 1e-2;

validationInterval = 10;
batchSize = 128;
trainLossSampleSize = 1e4;
nUpdates = 1e5;

%% Corrupted inputs
mask = gpuArray.rand(784,60000,'single') < nanNoise;
corruptedInputs = inputs;
corruptedInputs(mask) = NaN;

%% Setup model and trainer
ae = ImputingAutoEncoder('nSteps', 100, 'lam', 0);
enc = LogisticHiddenLayer(784, codeSize);
dec = LogisticOutputLayer(codeSize, 784);
ae.encodeLayer = enc;
ae.decodeLayer = dec;

trainer = ImputingGradientTrainer();
trainer.model = ae;
trainer.stepCalculator = AdaDelta();
trainer.progressMonitor = BasicMonitor();
trainer.progressMonitor.reporter = MNISTReporter(8, 8);
trainer.dataManager = ImputingDataManager({corruptedInputs}, ...
                                  {}, ...
                                  'batchSize', batchSize, ...
                                  'trainLossSampleSize', trainLossSampleSize);

%% Train model
trainer.train(nUpdates);