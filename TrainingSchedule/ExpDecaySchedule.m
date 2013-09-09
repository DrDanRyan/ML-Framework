classdef ExpDecaySchedule < TrainingSchedule
   % Fixed learnRate and momentum. Fixed number of epochs.
   
   properties
      params % {learnRate, momentum}
      maxEpochs
      lr0
      lrDecay
      epoch = 0;
   end
   
   methods
      function obj = ExpDecaySchedule(lr0, momentum, maxEpochs, lrDecay)
         obj.lr0 = lr0;
         obj.params{1} = lr0;
         obj.params{2} = momentum;
         obj.maxEpochs = maxEpochs;
         
         if nargin < 4
            obj.lrDecay = .9931; % 100 iteration half-life
         else
            obj.lrDecay = lrDecay;
         end
      end
      
      function isContinue = update(obj, ~, ~, ~)
         obj.epoch = obj.epoch + 1;
         isContinue = obj.epoch < obj.maxEpochs;
         obj.params{1} = obj.params{1}*obj.lrDecay;
      end
      
      function reset(obj)
         obj.epoch = 0;
         obj.params{1} = obj.lr0;
      end
   end
   
end