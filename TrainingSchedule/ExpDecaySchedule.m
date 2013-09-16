classdef ExpDecaySchedule < TrainingSchedule
   % A schedule for a learning rate and momentum StepCalculator where learning rate 
   % decays exponentially, momentum is fixed and a fixed number of epochs.
   
   properties
      params % {learnRate, momentum}
      maxEpochs % number of epochs to train before terminating
      lr0 % initial learning rate
      lrDecay % multiplies learnRate every epoch
      epoch = 0; % current epoch
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
         % Multiply learnRate by decay constant and increment epoch.
         % Terminate if maxEpochs reached.
         obj.epoch = obj.epoch + 1;
         isContinue = obj.epoch < obj.maxEpochs;
         obj.params{1} = obj.params{1}*obj.lrDecay;
      end
      
      function reset(obj)
         % Set epoch = 0, learnRate = lr0.
         obj.epoch = 0;
         obj.params{1} = obj.lr0;
      end
   end
   
end