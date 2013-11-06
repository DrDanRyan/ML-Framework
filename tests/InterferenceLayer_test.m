clear all
load ICU_ffn_data
targets(targets == -1) = 0;

nHidden = 128;

lr0 = .01;
lrDecay = [];

maxMomentum = .99;
C = 25;
slowMomentum = [];
slowEpochs = 0;

maxEpochs = 5000;
lookAhead = 100;
burnIn = 0;
batchSize = [];

ffn = FeedForwardNet('inputDropout', 0, 'hiddenDropout', .2);
ffn.hiddenLayers = {MaxoutHiddenLayer(187, nHidden, 5, 'initScale', .005, 'maxFanIn', 2), ...
                    MaxoutHiddenLayer(nHidden, nHidden, 5, 'initType', 'sparse', 'maxFanIn', 2), ...
                    MaxoutHiddenLayer(nHidden, nHidden, 5, 'initType', 'sparse', 'maxFanIn', 2)};
ffn.outputLayer = LogisticOutputLayer(nHidden, 1, 'initScale', .005);

sampler = StratifiedSampler(.85);
trainer = GradientTrainer();
trainer.stepCalculator = NesterovMomentum();
trainer.model = ffn;
trainer.reporter = ConsoleReporter();
trainer.lossFunction = @(y, t) compute_Lemeshow(y, t);
trainer.trainingSchedule = EarlyStopping(maxEpochs, 'lr0', lr0, ...
                                                    'lrDecay', lrDecay, ...
                                                    'maxMomentum', maxMomentum, ...
                                                    'slowMomentum', slowMomentum, ...
                                                    'C', C, ...
                                                    'lookAhead', lookAhead, ...
                                                    'burnIn', burnIn, ...
                                                    'slowEpochs', slowEpochs);

outputs = gpuArray.zeros(1, 4000);     
for i = 1:5
      bestValidationLoss = Inf;
      testSplit = hold_outs{i};
      trainSplit = setdiff(1:4000, testSplit);
      [trainIdx, validIdx] = sampler.sample(trainSplit, targets(trainSplit));
      trainer.dataManager = DataManager({inputs(:,trainIdx), targets(trainIdx)}, ...
                                             {inputs(:,validIdx), targets(validIdx)}, ...
                                             'batchSize', batchSize);
      trainer.reset();
      trainer.train();
      outputs(testSplit) = trainer.trainingSchedule.bestModel.output(inputs(:,testSplit));
%       for j = 1:4
%          trainer.reset();
%          trainer.train();
%          if trainer.trainingSchedule.bestValidationLoss < bestValidationLoss
%             bestModel = trainer.trainingSchedule.bestModel;
%             bestValidationLoss = trainer.trainingSchedule.bestValidationLoss;
%             outputs(testSplit) = bestModel.output(inputs(:,testSplit));
%          end
%       end
end

outputs = gather(outputs);
event1 = compute_event1(outputs, targets)
lemeshow = compute_Lemeshow(outputs, targets, true)



