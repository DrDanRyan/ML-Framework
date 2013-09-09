clear all
load Citi_data

%%
nDims = 100;
Layer1 = 500;
Layer2 = 1000;

sampler = ProportionSubsampler(.7);
batchSize = 400;
learnRate = .5;
momentum = .9;
nEpochs = 40;
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

ensemble = train_ensemble(data, targets, models, preprocessors, trainer, sampler);