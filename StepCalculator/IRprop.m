classdef IRprop < StepCalculator
   % "Improved Rprop" with weight backtracking. 
   % Should only be used in full batch mode (for mini-batches use RMSprop).
   
   properties
      upFactor
      downFactor
      maxRate
      minRate
      initialRate
      
      rates
      prevStep
      prevLoss
      prevGrad
   end
   
   methods
      function obj = IRprop(initialRate, varargin)
         p = inputParser();
         p.addParamValue('upFactor', 1.2);
         p.addParamValue('downFactor', .5);
         p.addParamValue('maxRate', 5);
         p.addParamValue('minRate', 1e-4);
         parse(p, varargin{:});
         
         obj.initialRate = initialRate;
         obj.upFactor = p.Results.upFactor;
         obj.downFactor = p.Results.downFactor;
         obj.maxRate = p.Results.maxRate;
         obj.minRate = p.Results.minRate;
      end
      
      
      function take_step(obj, x, t, model, ~)
         [grad, y] = model.gradient(x, t);
         loss = model.compute_loss(y, t);
         step = cell(size(grad));
         
         if isempty(obj.prevGrad)
            obj.rates = cell(size(grad));
            for i = 1:length(grad)
               obj.rates{i} = obj.initialRate*model.gpuState.ones(size(grad{i}));
               step{i} = obj.rates{i}.*sign(grad{i});
            end
         else
            isLossIncrease = loss > obj.prevLoss;
            for i = 1:length(grad)
               gradProduct = grad{i}.*obj.prevGrad{i};
               
               upIdx = gradProduct > 0;
               downIdx = gradProduct < 0;
               
               obj.rates{i}(upIdx) = min(obj.maxRate, ...
                                         obj.upFactor*obj.rates{i}(upIdx));
               obj.rates{i}(downIdx) = max(obj.minRate, ...
                                           obj.downFactor*obj.rates{i}(downIdx));
               
               grad{i}(downIdx) = 0;                         
               step{i} = obj.rates{i}.*sign(grad{i});
               if isLossIncrease
                  step{i}(downIdx) = -obj.prevStep{i}(downIdx);
               end
            end
         end
         model.increment_params(step);
         obj.prevStep = step;
         obj.prevLoss = loss;
         obj.prevGrad = grad;
      end
      
      function reset(obj)
         obj.rates = [];
         obj.prevGrad = [];
         obj.prevLoss = [];
         obj.prevStep = [];
      end
   end
end

