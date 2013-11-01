clear all
load ICU_ffn_data
targets(targets == -1) = 0;

nHidden1 = 64;
nHidden2 = 128;
maxEpochs = 2000;
lr0 = 1.0;
momentum = .75;
lookAhead = 70;
burnIn = 30;

ffn = FeedForwardNet('inputDropout', 0, 'hiddenDropout', 0);
ffn.hiddenLayers = {InterferenceLayer(187, nHidden1), InterferenceLayer(nHidden1, nHidden2)};
ffn.outputLayer = LogisticOutputLayer(nHidden2, 1);

sampler = StratifiedSampler(.8);
trainer = GradientTrainer();
trainer.stepCalculator = NesterovMomentum();
trainer.model = ffn;
trainer.reporter = ConsoleReporter();
trainer.trainingSchedule = EarlyStopping(maxEpochs, 'lr0', lr0, ...
                                                    'momentum', momentum, ...
                                                    'lookAhead', lookAhead, ...
                                                    'burnIn', burnIn);

outputs = gpuArray.zeros(1, 4000);                                                 
for i = 1:5
   testSplit = hold_outs{i};
   trainSplit = setdiff(1:4000, testSplit);
   [trainIdx, validIdx] = sampler.sample(trainSplit, targets(trainSplit));
   trainer.reset();
   trainer.dataManager = DataManager({inputs(:,trainIdx), targets(trainIdx)}, ...
                                       {inputs(:,validIdx), targets(validIdx)});
   trainer.train();
   outputs(testSplit) = ffn.output(inputs(:,testSplit));
end

outputs = gather(outputs);
event1 = compute_event1(outputs, targets)
lemeshow = compute_Lemeshow(outputs, targets, true)



