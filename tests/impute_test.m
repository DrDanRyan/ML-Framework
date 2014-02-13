clear all
load MNIST_train

%% Define parameters
codeSize = 16;
validProp = 1/6;
nanNoise = .2;
tol = 1e-2;

lr0 = .01;
rho = .8;
validationInterval = 100;
batchSize = 128;
trainLossSampleSize = 1e4;
nUpdates = 1e5;

%% Corrupted inputs
mask = gpuArray.rand(784,60000,'single') < nanNoise;
corruptedInputs = inputs;
corruptedInputs(mask) = NaN;

%% Setup model and trainer
ae = AutoEncoder('isTiedWeights', true, 'imputeTol', tol);
enc = LogisticHiddenLayer(784, codeSize);
dec = LogisticOutputLayer(codeSize, 784);
ae.encodeLayer = enc;
ae.decodeLayer = dec;

sampler = MulticlassSampler(validProp);
[trainIdx, validIdx] = sampler.sample(1:60000, targets);

trainer = GradientTrainer();
trainer.model = ae;
trainer.stepCalculator = AdaDelta();
trainer.parameterSchedule = MomentumSchedule(lr0, rho);
trainer.progressMonitor = BasicMonitor('validationInterval', validationInterval);
trainer.progressMonitor.reporter = MNISTReporter(4, 4);
trainer.dataManager = DataManager({inputs(:,trainIdx)}, ...
                                  {inputs(:,validIdx)}, ...
                                  'batchSize', batchSize, ...
                                  'trainLossSampleSize', trainLossSampleSize);

%% Train model
trainer.train(nUpdates);