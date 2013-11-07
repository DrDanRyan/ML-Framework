classdef MomentumSchedule < TrainingSchedule
   
   properties
      params % {learnRate, momentum}
      
      lr0
      lrDecay % exponential decay rate for learning rate
      lrBurnIn % number of epochs before lrDecay is applied
      
      % momentum at epoch t:= min(maxMomentum, (t + C)/(t + 2*C))
      % for example, t=0 => rho=1/2, t=C => rho=2/3, t=8C => rho=9/10
      C
      maxMomentum
      
      fastEpochs
      slowEpochs
      slowMomentum
      slowLearnRate
      
      epoch = 0 
   end
   
   methods
      function obj = MomentumSchedule(fastEpochs, lr0, maxMomentum, varargin{:})
         p = inputParser();
         p.addParamValue('lr0', []);
         p.addParamValue('lrDecay', []);
         p.addParamValue('lrBurnIn', 0);
         p.addParamValue('C', 0);
         p.addParamValue('maxMomentum', []);
         p.addParamValue('slowEpochs', 0);
         p.addParamValue('slowMomentum', []);
         p.addParamValue('slowLearnRate', []);
         parse(p, varargin{:});
         
         obj.lr0 = lr0;
         obj.lrDecay = p.Results.lrDecay;
         obj.lrBurnIn = p.Restuls.lrBurnIn;
         obj.C = p.Results.C;
         obj.maxMomentum = maxMomentum;
         obj.fastEpochs = fastEpochs;
         obj.slowEpochs = p.Results.slowEpochs;
         obj.slowMomentum = p.Results.slowMomentum;
         obj.slowLearnRate = p.Results.slowLearnRate;
         
         obj.params{1} = lr0;
         obj.params{2} = min(maxMomentum, (1+obj.C)/(1+2*obj.C));
      end
      
      function isContinue = update(obj, trainer, trainingLoss, validationLoss)
         
      end
      
      function reset(obj)
         
      end
   end
   
end

