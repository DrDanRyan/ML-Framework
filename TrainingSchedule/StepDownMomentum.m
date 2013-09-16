classdef StepDownMomentum < TrainingSchedule
   % A schedule for a learning rate and momentum StepCalculator that
   % multiplies the learning rate by cutFactor when learning appears to
   % have stalled. StepCalculator is also reset() when this cut happens so
   % that the model can begin fresh with the newly reduced learning rate.
   % After maxCuts or maxEpochs have been reached training is terminated.
   
   properties
      params % {learnRate, momentum}
      lr0 % initial learning rate
      maxEpochs % maximum number of training epochs
      epoch = 0; % current epoch
      
      cutFactor % multiplies learning rate every time training 'stalls'
      nCuts = 0; % current number of learning rate cuts so far
      maxCuts % training terminates if nCuts == maxCuts 
      
      % amount of time allowed to make progress since last 'event' before
      % learning is considered 'stalled'. 'event' can be either an
      % improvement in validationLoss from the previous 'event' or a
      % learning rate cut.
      lookAhead 
      EventValidationLoss = Inf; % validationLoss at previous event
      lastEvent = 0; % epoch of last 'event'
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
      
      function isContinue = update(obj, trainer, ~, validationLoss)
         % Checks if an 'event' occurs, namely an improvement in
         % validation loss from previous 'event' or a cut in the learning
         % rate. Sends signal to terminate training when maxEpochs or
         % maxCuts is reached.
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
                  trainer.stepCalculator.reset();
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
         % Set epoch = 0, learnRate = lr0, lastEvent = 0,
         % EventValidaitonLoss = Inf, and nCuts = 0.
         obj.epoch = 0;
         obj.params{1} = obj.lr0;
         obj.lastEvent = 0;
         obj.EventValidationLoss = Inf;
         obj.nCuts = 0;
      end
   end
   
end