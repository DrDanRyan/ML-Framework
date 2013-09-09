clear all
load Citi_data

%%
nFolds = 4;
nDims = 100;
Layer1 = 500;
Layer2 = 1000;

sampler = ProportionSubsampler(.7);
batchSize = 400;
learnRate = .5;
momentum = .9;
nEpochs = 200;
lrDecay = .99;

ensembleSize = 10;
models = cell(1, ensembleSize);
preprocessors = cell(1, ensembleSize);


%%
for i = 1:ensembleSize
   models{i} = FeedForwardNet();
   models{i}.hiddenLayers = {ReluHiddenLayer(nDims, Layer1), ...
                               ReluHiddenLayer(Layer1, Layer2, 'initType', 'sparse', 'initScale', 15)};
   models{i}.outputLayer = LogisticOutputLayer(Layer2);

   preprocessors{i} = RandomProjection(187, nDims);
end


%% 
trainer = GradientTrainer();
trainer.dataManager = BasicDataManager(batchSize);
trainer.stepCalculator = NAG();
trainer.reporter = ConsoleReporter();
trainer.trainingSchedule = ExpDecaySchedule(learnRate, momentum, nEpochs, lrDecay);

%%

outputs = crossvalidation_ensemble(nFolds, data, targets, models, preprocessors, trainer, sampler);

%%
for j = 1:ensembleSize
   event1 = Event1Score(outputs(j,:), targets);
   lemeshow = Lemeshow(outputs(j,:), targets);
   scores(j,:) = [event1, lemeshow]; %#ok<SAGROW>

   fprintf('\n\n Max Score: %.4f \t Lemeshow %.4f \n', ...
              event1, lemeshow);
end

means = mean(outputs);
means = max(.0075, means);
mean_scores = [Event1Score(means, targets), Lemeshow(means, targets)]