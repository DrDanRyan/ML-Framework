clear all
load good_ae1
%inputs(isnan(inputs)) = 0;
hiddenLayers = {ae.encodeLayer, []};
sampler = ProportionSubsampler(.8);
trainIdx = sampler.sample(1:8000);
validIdx = setdiff(1:8000, trainIdx);
trainInputs  = inputs2(:, trainIdx);
validInputs = inputs2(:, validIdx);

ae = AutoEncoder('inputDropout', .3, 'hiddenDropout', .3);
ae.encodeLayer = MaxoutHiddenLayer(400, 512, 5);
ae.decodeLayer = ComboOutputLayer(MaxoutHiddenLayer(512, 400, 5), MeanSquaredError());

trainer = GradientTrainer();
trainer.reporter = ConsoleReporter();
trainer.stepCalculator = NesterovMomentum();
trainer.model = ae;
trainer.trainingSchedule = EarlyStopping(2000, 'lr0', .1, ...
                                               'momentum', .8, ...
                                               'burnIn', 30, ...
                                               'lookAhead', 30, ...
                                               'lrDecay', 1);
                                            

trainer.dataManager = FullBatch(trainInputs, trainInputs, validInputs, validInputs);
trainer.train();

%% Begin supervised fine-tuning with cross-validation
clear inputs
load ICU_ffn_data
ffn = FeedForwardNet('inputDropout', .2, 'hiddenDropout', .5);
hiddenLayers{2} = ae.encodeLayer;
ffn.hiddenLayers = hiddenLayers;
nFolds = 5;
[outputs, testLoss] = CV_single_model(inputs, targets, nFolds, @ICU_fine_tune, ffn);
event1 = compute_event1(outputs, targets)
%lemeshow = compute_Lemeshow(outputs, targets, true)
               