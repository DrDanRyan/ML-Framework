classdef Rprop < StepCalculator
   % Basic resilient backprop (Rprop). Should only be used in full batch
   % mode (for mini-batches use RMSprop).
   
   properties
      upFactor
      downFactor
      maxRate
      minRate
      initialRate
      
      rates
      prevGrad
   end
   
   methods
      function obj = Rprop(initialRate, varargin)
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
         grad = model.gradient(x, t);
         obj.update_rates(grad, model);
         steps = cellfun(@(rate, grad) rate.*sign(grad), obj.rates, grad, ...
                          'UniformOutput', false);
         model.increment_params(steps);
         obj.prevGrad = grad;
      end
      
      function update_rates(obj, grad, model)
         if isempty(obj.prevGrad)
            obj.rates = cell(size(grad));
            for i = 1:length(grad)
               obj.rates{i} = obj.initialRate*model.gpuState.ones(size(grad{i}));
            end
         else
            for i = 1:length(grad)
               gradProduct = grad{i}.*obj.prevGrad{i};
               upIdx = gradProduct > 0;
               downIdx = gradProduct < 0;

               obj.rates{i}(upIdx) = min(obj.maxRate, ...
                                         obj.upFactor*obj.rates{i}(upIdx));
               obj.rates{i}(downIdx) = max(obj.minRate, ...
                                           obj.downFactor*obj.rates{i}(downIdx));
            end
         end
      end
      
      function reset(obj)
         obj.rates = [];
         obj.prevGrad = [];
      end
   end
end

