clear all
load ICU_ffn_data
targets(targets == -1) = 0;

nHidden = 256;
maxEpochs = 2000;
lr0 = .2;
momentum = .95;
lookAhead = 60;
burnIn = 30;

ffn = FeedForwardNet('inputDropout', 0, 'hiddenDropout', 0);
ffn.hiddenLayers = {InterferenceLayer(187, nHidden)};
ffn.outputLayer = LogisticOutputLayer(nHidden, 1);

sampler = StratifiedSampler(.85);
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



