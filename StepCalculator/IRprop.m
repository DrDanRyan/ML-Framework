classdef IRprop < StepCalculator
   % "Improved Rprop" with weight backtracking. 
   % Should only be used in full batch mode (for mini-batches use RMSprop).
   %
   % Like Rprop except when a 'bad step' has been taken (gradient switched
   % signs and overall batch error has increased) weights are reset to
   % previous values
   
   properties
      upFactor
      downFactor
      maxRate
      minRate
      
      rates
      prevStep
      prevLoss
      prevGrad
   end
   
   methods
      function obj = IRprop(varargin)
         p = inputParser();
         p.addParamValue('upFactor', 1.2);
         p.addParamValue('downFactor', .5);
         p.addParamValue('maxRate', 5);
         p.addParamValue('minRate', 1e-6);
         parse(p, varargin{:});
         
         obj.upFactor = p.Results.upFactor;
         obj.downFactor = p.Results.downFactor;
         obj.maxRate = p.Results.maxRate;
         obj.minRate = p.Results.minRate;
      end
      
      
      function take_step(obj, x, t, model, params)
         % Check each connection to see if gradient has changed sign from
         % previous step. If so, check if error has increased or decreased.
         % If error increase backtrack previous weight changes on those
         % connections, if error decrease, only decrease learning rate on
         % those connections. If gradient has not changed signs, increase
         % learning rates on those connections.
         [grad, y] = model.gradient(x, t);
         loss = model.compute_loss(y, t);
         step = cell(size(grad));
         
         if isempty(obj.prevGrad)
            obj.rates = cell(size(grad));
            for i = 1:length(grad)
               obj.rates{i} = params{1}*model.gpuState.ones(size(grad{i}));
               step{i} = -obj.rates{i}.*sign(grad{i});
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
               step{i} = -obj.rates{i}.*sign(grad{i});
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
         % Clear rates and previous step, gradient and loss information
         obj.rates = [];
         obj.prevGrad = [];
         obj.prevLoss = [];
         obj.prevStep = [];
      end
   end
end

