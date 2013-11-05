clear all
load ICU_ffn_data
targets(targets == -1) = 0;

nHidden1 = 128;

lr0 = .01;
maxMomentum = .99;
C = 10;
slowMomentum = .5;
maxEpochs = 5000;
lookAhead = 20;
slowEpochs = 10;
burnIn = 30;

ffn = FeedForwardNet('inputDropout', .2, 'hiddenDropout', .5);
ffn.hiddenLayers = {MaxoutHiddenLayer(187, nHidden1, 5)};
ffn.outputLayer = LogisticOutputLayer(nHidden1, 1);

sampler = StratifiedSampler(.8);
trainer = GradientTrainer();
trainer.stepCalculator = NesterovMomentum();
trainer.model = ffn;
trainer.reporter = ConsoleReporter();
%trainer.lossFunction = @(y, t) -1*compute_event1(y, t);
trainer.trainingSchedule = EarlyStopping(maxEpochs, 'lr0', lr0, ...
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
                                             {inputs(:,validIdx), targets(validIdx)});
      trainer.reset();
      trainer.train();
      outputs(testSplit) = ffn.output(inputs(:,testSplit));
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



