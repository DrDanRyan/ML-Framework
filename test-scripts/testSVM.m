clear all

%% Load Data and create DataManager
load test_data
trainingTargets(1:350) = -1;
validationTargets(1:150) = -1;
dataManager = BasicDataManager(trainingInputs, trainingTargets, ...
                               validationInputs, validationTargets, 100);


%% Initialize Model
nnet = FeedForwardNet('dropout', false);
nnet.hiddenLayers = {ReluHiddenLayer(2, 100, 'initScale', 1), ...
                     ReluHiddenLayer(100, 100, 'initType', 'sparse', 'initScale', 15)};
nnet.outputLayer = SVMOutputLayer(100, 'hingeExp', 1.5);

%% Initialize Reporter
reporter = ConsoleReporter();

%% Initialize StepCalculator
stepper = NAG();

%% Initialize TrainingSchedule
schedule = BasicMomentumSchedule(.01, .9, 50);

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
colorbar()
clabel(C, [0, 0])
hold on
setA = gather([trainingInputs(:, 1:350), validationInputs(:, 1:150)]);
setB = gather([trainingInputs(:, 351:end), validationInputs(:, 151:end)]);
scatter(setA(1,:), setA(2,:), 'r+')
scatter(setB(1,:), setB(2,:), 'b*')