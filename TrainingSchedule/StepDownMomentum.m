classdef StepDownMomentum < TrainingSchedule
   % Fixed learnRate and momentum. Fixed number of epochs.
   
   properties
      params % {learnRate, momentum}
      lr0
      maxEpochs
      epoch = 0;
      
      cutFactor
      nCuts = 0;
      maxCuts
      lookAhead
      
      EventValidationLoss = Inf;
      lastEvent = 0;
   end
   
   methods
      function obj = StepDownMomentum(lr0, momentum, maxEpochs, varargin)
         obj.lr0 = lr0;
         obj.params{1} = lr0;
         obj.params{2} = momentum;
         obj.maxEpochs = maxEpochs;
         
         p = inputParser();
         p.addParamValue('cutFactor', .1);
         p.addParamValue('lookAhead', 30)
         p.addParamValue('maxCuts', 2)
         parse(p, varargin{:});
         
         obj.cutFactor = p.Results.cutFactor;
         obj.maxCuts = p.Results.maxCuts;
         obj.lookAhead = p.Results.lookAhead;
      end
      
      function isContinue = update(obj, ~, ~, validationLoss)
         obj.epoch = obj.epoch + 1;
         isContinue = obj.epoch < obj.maxEpochs;
         
         if validationLoss < obj.EventValidationLoss
            % validation loss improved event
            obj.EventValidationLoss = validationLoss;
            obj.lastEvent = obj.epoch;
         else
            if obj.epoch >= obj.lastEvent + obj.lookAhead
               if obj.nCuts < obj.maxCuts
                  % cut learnRate event
                  obj.params{1} = obj.params{1}*obj.cutFactor;
                  obj.nCuts = obj.nCuts + 1;
                  obj.lastEvent = obj.epoch;
                  obj.EventValidationLoss = validationLoss;
                  fprintf('\n\n Learning Rate has been cut to %d \n\n', obj.params{1})
               else
                  % too many cuts --> terminate
                  isContinue = false;
               end
            end
         end
      end
      
      function reset(obj)
         obj.epoch = 0;
         obj.params{1} = obj.lr0;
         obj.lastEvent = 0;
         obj.EventValidationLoss = Inf;
         obj.nCuts = 0;
      end
   end
   
end