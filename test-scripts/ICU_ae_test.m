clear all
load ICU_SVM_data_gpu
targets(targets==-1) = 0;

ae = AutoEncoder('inputDropout', .3, 'hiddenDropout', .3);
ae.encodeLayer = MaxoutHiddenLayer(187, 400, 5);
ae.decodeLayer = ComboOutputLayer(LinearHiddenLayer(400, 187), MeanSquaredError());

trainer = GradientTrainer();
trainer.reporter = ConsoleReporter();
trainer.stepCalculator = NesterovMomentum();
trainer.model = ae;
trainer.trainingSchedule = EarlyStopping(2000, 'lr0', .05, ...
                                               'momentum', .8, ...
                                               'burnIn', 30, ...
                                               'lookAhead', 20, ...
                                               'lrDecay', 1);
                                            
sampler = StratifiedSampler(.8);
trainIdx = sampler.sample(1:4000, targets);
validIdx = setdiff(1:4000, trainIdx);
trainInputs  = inputs(:, trainIdx);
validInputs = inputs(:, validIdx);
trainer.dataManager = FullBatch(trainInputs, trainInputs, validInputs, validInputs);
trainer.train();

%% 

net = FeedForwardNet('inputDropout', .3, 'hiddenDropout', .3);
net.hiddenLayers = {ae.encodeLayer};
net.outputLayer = LogisticOutputLayer(400);
trainer.model = net;
trainer.stepCalculator.reset();
trainer.dataManager = FullBatch(trainInputs, targets(trainIdx), validInputs, targets(validIdx));
trainer.trainingSchedule = EarlyStopping(2000, 'lr0', .1, ...
                                               'momentum', .8, ...
                                               'burnIn', 30, ...
                                               'lookAhead', 20, ...
                                               'lrDecay', .998);
                                            
trainer.train();
               