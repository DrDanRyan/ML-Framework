clear all

%% Load Data and create DataManager
load test_data
batchSize = size(trainingInputs, 2);
dataManager = BasicDataManager(batchSize, trainingInputs, trainingTargets, ...
                               validationInputs, validationTargets);

%% Initialize Model
nnet = FeedForwardNet('dropout', true);
nnet.hiddenLayers = {ReluHiddenLayer(2, 100, 'initScale', 1), ...
                     ReluHiddenLayer(100, 100, 'initType', 'sparse', 'initScale', 15), ...
                     ReluHiddenLayer(100, 100, 'initType', 'sparse', 'initScale', 15)};
nnet.outputLayer = LogisticOutputLayer(100);

%% Initialize Reporter
reporter = ConsoleReporter();

%% Initialize StepCalculator
stepper = Rprop(.01);

%% Initialize TrainingSchedule
schedule = BasicMomentumSchedule(.01, .9, 100);

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
[x, y] = meshgrid(-6:.05:6);
x = reshape(x, 1, []);
y = reshape(y, 1, []);
z = gather(nnet.output([x; y]));
[C, h] = contour(reshape(x, 241, []), reshape(y, 241, []), reshape(z, 241, []));
clabel(C, .5)
hold on
setA = gather([trainingInputs(:, 1:350), validationInputs(:, 1:150)]);
setB = gather([trainingInputs(:, 351:end), validationInputs(:, 151:end)]);
scatter(setA(1,:), setA(2,:), 'r+')
scatter(setB(1,:), setB(2,:), 'b*')