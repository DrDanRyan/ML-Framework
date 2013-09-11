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
         p.addParamValue('maxRate', 10);
         p.addParamValue('minRate', 1e-5);
         parse(p, varargin{:});
         
         obj.initialRate = initialRate;
         obj.upFactor = p.Results.upFactor;
         obj.downFactor = p.Results.downFactor;
         obj.maxRate = p.Results.maxRate;
         obj.minRate = p.Results.minRate;
      end
      
      
      function take_step(obj, x, t, model, ~)
         grad = model.gradient(x, t);
         if isempty(obj.prevGrad) % First step --> use initialRate
            obj.rates = cellfun(@(x) obj.initialRate*model.gpuState.ones(size(x)), grad, ...
                                'UniformOutput', false);
         end
         steps = cellfun(@(rate, grad) rate.*sign(grad), obj.rates, grad, ...
                         'UniformOutput', false);
         obj.model.increment_params(steps);
         obj.update_rates(grad);
      end
      
      function update_rates(obj, grad)
         % Multiply rates where gradient directions agree by upFactor
         obj.rates = cellfun(@(grad, prevGrad, rates) obj.upFactor*rates(grad.*prevGrad > 0), ...
                           grad, obj.prevGrad, obj.rates, 'UniformOutput', false);
                        
         % Multiply rates where gradient directions are opposite by
         % downFactor
         obj.rates = cellfun(@(grad, prevGrad, rates) obj.downFactor*rates(grad.*prevGrad < 0), ...
                            grad, obj.prevGrad, obj.rates, 'UniformOutput', false);
                         
         obj.prevGrad = grad;
      end
      
      function reset(obj)
         obj.rates = [];
         obj.prevGrad = [];
      end
   end
end

