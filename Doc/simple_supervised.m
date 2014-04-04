clear all
load MNIST_train

%% Declare parameters
% Sampler
trainProp = .70;

% Model
inputSize = 784;
layer1Size = 512;
layer2Size = 1024;
targetSize = 10;

% DataManager
batchSize = 128;

% ParameterSchedule
lr = .05;
momentum = .70;

% ProgressMonitor
validationInterval = 200;

% Training
nUpdates = 2e4;

%% Prepare for training
trainer = Trainer();

% Create the sample object and draw disjoint training and validation sets
sampler = ProportionSampler(5/6);
[trainIdx, validIdx] = sampler.sample(1:60000);

% Split the inputs and targets according to the index vectors sampled above
trainInputs = inputs(:,trainIdx);
trainTargets = targets(:,trainIdx);
validInputs = inputs(:,validIdx);
validTargets = targets(:,validIdx);

% Clear old variables to save memory (probably not really necessary)
clear inputs targets

% Create and attach DataManager
trainer.dataManager = DataManager({trainInputs, trainTargets}, ...
                                  {validInputs, validTargets}, ...
                                  'batchSize', batchSize);
                               
% Create the HiddenLayer objects
layer1 = LogisticHiddenLayer(inputSize, layer1Size);
layer2 = LogisticHiddenLayer(layer1Size, layer2Size);                               

% Create the OutputLayer object
outputLayer = SoftmaxOutputLayer(layer2Size, targetSize);

% Create the Model object and attach the HiddenLayers as a cell array
% and the OutputLayer directly. Then attach the Model to the Trainer.
model = FeedForwardNet();
model.hiddenLayers = {layer1, layer2};
model.outputLayer = outputLayer;
trainer.model = model;

% Create and attach StepCalculator
trainer.stepCalculator = NesterovMomentum();

% Create and attach ParameterSchedule
trainer.parameterSchedule = MomentumSchedule(lr, momentum);

% Create and attach ProgressMonitor
trainer.progressMonitor = BasicMonitor('validationInterval', 100);

%% Perform training
trainer.train(nUpdates);




