classdef EarlyStopping < TrainingSchedule
   
   properties
      params % {learnRate, momentum}
      maxEpochs % maximum number of training epochs
      epoch = 0; % current epoch

      burnIn
      lookAhead 
      bestValidationLoss = Inf; 
      bestEpoch = 0;
   end
   
   methods
      function obj = StepDownMomentum(lr0, momentum, maxEpochs, varargin)
         obj.params{1} = lr0;
         obj.params{2} = momentum;
         obj.maxEpochs = maxEpochs;
         
         p = inputParser();
         p.addParamValue('burnIn', 60);
         p.addParamValue('lookAhead', 20);
         parse(p, varargin{:});
         
         obj.burnIn = p.Results.burnIn;
         obj.lookAhead = p.Results.lookAhead;
      end
      
      function isContinue = update(obj, ~, ~, validationLoss)
         obj.epoch = obj.epoch + 1;
         if validationLoss <= obj.bestValidationLoss
            obj.bestValidationLoss = validationLoss;
            obj.bestEpoch = obj.epoch;
         end
         
         if obj.epoch <= obj.burnIn
            isContinue = true;
         else
            isContinue = (obj.epoch < obj.maxEpochs ...
                              && obj.epoch < obj.bestEpoch + obj.lookAhead);
         end
      end
      
      function reset(obj)
         obj.epoch = 0;
         obj.bestValidationLoss = Inf;
         obj.bestEpoch = 0;
      end
   end
   
end