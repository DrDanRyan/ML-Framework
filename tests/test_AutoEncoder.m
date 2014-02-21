%% Train a basic AutoEncoder on MNIST
clear all
close all
load MNIST_train

%% Define model parameters
nUpdates = 1e5;
codeSize = 25;
nRows = 5;
nCols = 5;

noiseType = 'none';
noiseLevel = .1;

lr0 = .01;
rho = .8;

validationInterval = 50;
batchSize = 512;
trainLossSampleSize = 1e4;


%% Setup model and trainer
encoder = LogisticHiddenLayer(784, codeSize, 'initScale', .1);
decoder = LogisticOutputLayer(codeSize, 784);
ae = DAE('noiseType', noiseType, 'noiseLevel', noiseLevel);
ae.encodeLayer = encoder;
ae.decodeLayer = decoder;

trainer = GradientTrainer();
trainer.model = ae;
trainer.stepCalculator = NesterovMomentum();
trainer.parameterSchedule = MomentumSchedule(lr0, rho);
trainer.progressMonitor = BasicMonitor('validationInterval', validationInterval);
trainer.progressMonitor.reporter = MNISTReporter(nRows, nCols);
trainer.dataManager = DataManager({inputs}, {}, 'batchSize', batchSize, ...
                                  'trainLossSampleSize', trainLossSampleSize);
                               
%% Train
trainer.train(nUpdates);