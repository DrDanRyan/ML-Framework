classdef EarlyStopping < TrainingSchedule
   
   properties
      params % {learnRate, momentum}
      lr0
      lrDecay % exponential decay rate for learning rate (only applied after burnIn period)
      
      % momentum at epoch t:= min(maxMomentum, (t + C)/(t + 2*C))
      % for example, t=0 => rho=1/2, t=C => rho=2/3, t=8C => rho=9/10
      C
      maxMomentum
      slowMomentum
      
      maxEpochs % maximum number of training epochs
      slowEpochs % number of epochs to run with reduced momentum after stopping 
                 % criteria is triggered
      isSlowPhase = false % boolean indicating if in fine-tuning low momentum phase or not
      slowStart % the epoch after which the slow phase began
 
      epoch = 0; % current epoch
      burnIn % minimum number of epochs before stopping based on validationLoss criteria
      lookAhead % begin stopping bestValidationLoss has not improved for this many epochs
      bestValidationLoss = Inf
      bestEpoch = 0
      isStoreBestModel
      bestModel
   end
   
   methods
      function obj = EarlyStopping(maxEpochs, varargin)
         obj.maxEpochs = maxEpochs;
         
         p = inputParser();
         p.addParamValue('burnIn', 60);
         p.addParamValue('lookAhead', 20);
         p.addParamValue('lr0', []);
         p.addParamValue('maxMomentum', []);
         p.addParamValue('C', []);
         p.addParamValue('slowMomentum', []);
         p.addParamValue('slowEpochs', []);
         p.addParamValue('lrDecay', []);
         p.addParamValue('isStoreBestModel', false);
         parse(p, varargin{:});
         
         obj.burnIn = p.Results.burnIn;
         obj.lookAhead = p.Results.lookAhead;
         obj.lr0 = p.Results.lr0;
         obj.maxMomentum = p.Results.maxMomentum;
         obj.C = p.Results.C;
         obj.slowMomentum = p.Results.slowMomentum;
         obj.slowEpochs = p.Results.slowEpochs;
         obj.lrDecay = p.Results.lrDecay;
         obj.isStoreBestModel = p.Results.isStoreBestModel;
         
         obj.params{1} = obj.lr0;
         if ~isempty(obj.maxMomentum)
            if ~isempty(obj.C)
               obj.params{2} = min(obj.maxMomentum, (1 + obj.C)/(1 + 2*obj.C));
            else
               obj.params{2} = obj.maxMomentum;
            end
         end
      end
      
      function isContinue = update(obj, trainer, ~, validationLoss)
         obj.epoch = obj.epoch + 1;
         if validationLoss < obj.bestValidationLoss
            obj.bestValidationLoss = validationLoss;
            obj.bestEpoch = obj.epoch;
            obj.bestModel = trainer.model.copy();
         end
         
         % Determine whether to stop or continue and whether to enter slow
         % fine-tuning stage
         if obj.epoch <= obj.burnIn
            isContinue = true;
         elseif obj.epoch >= obj.maxEpochs
            isContinue = false;
         elseif ~obj.isSlowPhase 
            if (obj.epoch >= obj.bestEpoch + obj.lookAhead) % begin slow phase
               obj.isSlowPhase = true;
               if ~isempty(obj.slowMomentum)
                  obj.params{2} = obj.slowMomentum;
               end
               obj.slowStart = obj.epoch;
            end
            isContinue = true;
         else % in slow phase
            if obj.epoch >= obj.slowStart + obj.slowEpochs;
               isContinue = false;
            else
               isContinue = true;
            end
         end
         
         obj.update_params();
         
      end
      
      function update_params(obj)
         % update learning rate
         if ~isempty(obj.lrDecay) && obj.epoch > obj.burnIn
            obj.params{1} = obj.params{1}*obj.lrDecay;
         end
         
         % update momentum
         if ~(obj.isSlowPhase) && ~isempty(obj.maxMomentum)
            if ~isempty(obj.C)
               obj.params{2} = min(obj.maxMomentum, (obj.epoch + obj.C)/(obj.epoch + 2*obj.C));
            else
               obj.params{2} = obj.maxMomentum;
            end
         end
      end
      
      function reset(obj)
         obj.epoch = 0;
         obj.slowStart = [];
         obj.isSlowPhase = false;
         obj.bestValidationLoss = Inf;
         obj.bestEpoch = 0;
         obj.bestModel = [];
         obj.params{1} = obj.lr0;
      end
   end
   
end