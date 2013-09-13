clear all

%% Load Data and create DataManager
load test_data
batchSize = size(trainingInputs, 2);
trainingTargets(trainingTargets==0) = -1;
validaitonTargets(validationTargets==0) = -1;
dataManager = BasicDataManager(batchSize, trainingInputs, trainingTargets, ...
                               validationInputs, validationTargets);

%% Initialize Model
nnet = FeedForwardNet('dropout', true);
nnet.hiddenLayers = {MaxoutHiddenLayer(2, 500, 5)};
nnet.outputLayer = SVMOutputLayer(500);

%% Initialize Reporter
reporter = ConsoleReporter();

%% Initialize StepCalculator
stepper = IRprop(.01);

%% Initialize TrainingSchedule
schedule = BasicMomentumSchedule(.05, .9, 200);

%% Initialize Trainer
trainer = GradientTrainer();
trainer.dataManager = dataManager;
trainer.model = nnet;
trainer.reporter = reporter;
trainer.stepCalculator = stepper;
trainer.trainingSchedule = schedule;

%% Train the model
trainer.train();

%% Visualize results
[x, y] = meshgrid(-3:.05:3);
x = reshape(x, 1, []);
y = reshape(y, 1, []);
z = gather(nnet.output([x; y]));
[C, h] = contour(reshape(x, 121, []), reshape(y, 121, []), reshape(z, 121, []));
clabel(C, 0)
hold on
setA = gather([trainingInputs(:, 1:350), validationInputs(:, 1:150)]);
setB = gather([trainingInputs(:, 351:end), validationInputs(:, 151:end)]);
scatter(setA(1,:), setA(2,:), 'r+')
scatter(setB(1,:), setB(2,:), 'b*')