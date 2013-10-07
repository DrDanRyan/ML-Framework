function model = autoencoder_maxout_ICU_test(inputs, targets)
%% Define parameters
sampleProp = .8;
reporter = [];

inputDropout = .2;
hiddenDropout = .2;
layer1Size = 192;
layer2Size = 128;
layer3Size = 96;
maxoutUnits = 5;

aelr0 = .3;
aeMomentum = .8;
aelrDecay = .996;
aeMaxEpochs = 1000;
aeBurnIn = 50;
aeLookAhead = 20;

ffnlr0 = .12;
ffnMomentum = .85;
ffnlrDecay = .995;
ffnMaxEpochs = 1000;
ffnBurnIn = 20;
ffnLookAhead = 10;

%% Create training and validation split for EarlyStopping
sampler = StratifiedSampler(sampleProp);
dataSize = size(targets, 2);
trainIdx = sampler.sample(1:dataSize, targets);
validIdx = setdiff(1:dataSize, trainIdx);
trainingInputs = inputs(:,trainIdx);
trainingTargets = targets(:,trainIdx);
validationInputs = inputs(:,validIdx);
validationTargets = targets(:,validIdx);

%% Train a stack of AutoEncoders

% Init autoencoder trainer
fprintf('Training First AutoEncoder...\n')
trainer = GradientTrainer();
trainer.trainingSchedule = EarlyStopping(aeMaxEpochs, 'lr0', aelr0, ...
                                                      'lrDecay', aelrDecay, ...
                                                      'momentum', aeMomentum, ...
                                                      'burnIn', aeBurnIn, ...
                                                      'lookAhead', aeLookAhead);
trainer.stepCalculator = NesterovMomentum;
trainer.reporter = reporter;
trainer.dataManager = FullBatch(trainingInputs, trainingInputs, ...
                                    validationInputs, validationInputs);

% Train first AutoEncoder                                 
encodeLayer = MaxoutHiddenLayer(187, layer1Size, maxoutUnits);
decodeLayer = LinearOutputLayer(layer1Size, 187);
ae1 = AutoEncoder('inputDropout', inputDropout, ...
                  'hiddenDropout', hiddenDropout, ...
                  'encodeLayer', encodeLayer, ...
                  'decodeLayer', decodeLayer);
               
trainer.model = ae1;
trainer.train();

% Train second AutoEncoder
fprintf('Training Second AutoEncoder...\n')
trainIn = trainer.model.encode(trainingInputs);
validIn = trainer.model.encode(validationInputs);
trainer.model = [];
trainer.reset();
trainer.dataManager = FullBatch(trainIn, trainIn, validIn, validIn);

encodeLayer = MaxoutHiddenLayer(layer1Size, layer2Size, maxoutUnits);
decodeLayer = LinearOutputLayer(layer2Size, layer1Size);
ae2 = AutoEncoder('inputDropout', hiddenDropout, ...
                  'hiddenDropout', hiddenDropout, ...
                  'encodeLayer', encodeLayer, ...
                  'decodeLayer', decodeLayer);

trainer.model = ae2;
trainer.train();

% Train third AutoEncoder
fprintf('Training Third AutoEncoder...\n')
trainIn = trainer.model.encode(trainIn);
validIn = trainer.model.encode(validIn);
trainer.model = [];
trainer.reset();
trainer.dataManager = FullBatch(trainIn, trainIn, validIn, validIn);

encodeLayer = MaxoutHiddenLayer(layer2Size, layer3Size, maxoutUnits);
decodeLayer = LinearOutputLayer(layer3Size, layer2Size);
ae3 = AutoEncoder('inputDropout', hiddenDropout, ...
                  'hiddenDropout', hiddenDropout, ...
                  'encodeLayer', encodeLayer, ...
                  'decodeLayer', decodeLayer);

trainer.model = ae3;
trainer.train();
                  
%% Convert AutoEncoders into FFN
fprintf('Training FeedForwardNet...\n')
model = FeedForwardNet('inputDropout', inputDropout, 'hiddenDropout', hiddenDropout);
model.hiddenLayers = {ae1.encodeLayer, ae2.encodeLayer, ae3.encodeLayer};
model.outputLayer = LogisticOutputLayer(layer3Size);

%% Fine-tune FFN
trainer = GradientTrainer();
trainer.trainingSchedule = EarlyStopping(ffnMaxEpochs, 'lr0', ffnlr0, ...
                                                       'lrDecay', ffnlrDecay, ...
                                                       'momentum', ffnMomentum, ...
                                                       'burnIn', ffnBurnIn, ...
                                                       'lookAhead', ffnLookAhead);
trainer.stepCalculator = NesterovMomentum();   
trainer.reporter = reporter;
trainer.dataManager = FullBatch(trainingInputs, trainingTargets, ...
                                    validationInputs, validationTargets);
trainer.model = model;
trainer.train();
fprintf('Fold Complete.\n\n')
end

