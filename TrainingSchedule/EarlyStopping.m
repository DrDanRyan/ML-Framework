classdef EarlyStopping < TrainingSchedule
   
   properties
      params % {learnRate, momentum}
      lr0
      lrDecay % exponential decay rate for learning rate (only applied after burnIn period)
      
      maxEpochs % maximum number of training epochs
      epoch = 0; % current epoch

      burnIn % minimum number of epochs before stopping based on validationLoss criteria
      lookAhead % stop if bestValidationLoss has not improved for this many epochs
      bestValidationLoss = Inf; 
      bestEpoch = 0;
   end
   
   methods
      function obj = EarlyStopping(maxEpochs, varargin)
         obj.maxEpochs = maxEpochs;
         
         p = inputParser();
         p.addParamValue('burnIn', 60);
         p.addParamValue('lookAhead', 20);
         p.addParamValue('lr0', []);
         p.addParamValue('momentum', []);
         p.addParamValue('lrDecay', []);
         parse(p, varargin{:});
         
         obj.burnIn = p.Results.burnIn;
         obj.lookAhead = p.Results.lookAhead;
         if ~isempty(p.Results.lr0)
            obj.params{1} = p.Results.lr0;
            obj.lr0 = p.Results.lr0;
         end
         
         if ~isempty(p.Results.momentum)
            obj.params{2} = p.Results.momentum;
         end
         
         if ~isempty(p.Results.lrDecay)
            obj.lrDecay = p.Results.lrDecay;
         end
      end
      
      function isContinue = update(obj, ~, ~, validationLoss)
         obj.epoch = obj.epoch + 1;
         if validationLoss < obj.bestValidationLoss
            obj.bestValidationLoss = validationLoss;
            obj.bestEpoch = obj.epoch;
         end
         
         if obj.epoch <= obj.burnIn
            isContinue = true;
         else
            isContinue = (obj.epoch < obj.maxEpochs ...
                              && obj.epoch < obj.bestEpoch + obj.lookAhead);
         end
         
         if ~isempty(obj.lrDecay) && obj.epoch > obj.burnIn
            obj.params{1} = obj.params{1}*obj.lrDecay;
         end
      end
      
      function reset(obj)
         obj.epoch = 0;
         obj.bestValidationLoss = Inf;
         obj.bestEpoch = 0;
         obj.params{1} = obj.lr0;
      end
   end
   
end