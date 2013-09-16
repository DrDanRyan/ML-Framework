classdef Rprop < StepCalculator
   % Basic resilient backprop (Rprop). Should only be used in full batch
   % mode (for mini-batches use RMSprop). Method automatically adapts
   % learning rates at each connection so params from TrainingSchedule are
   % ignored.
   % 
   % Formula used for step:
   % step = learnRate*sign(gradient) 
   % where learnRate is adaptively adjusted for each connection
   
   properties
      upFactor % multiplier for learnRate when 2 successive gradients agree
      downFactor % multiplier for learnRate when 2 successive gradients disagree
      maxRate % maximum value allowed for learning rate
      minRate % minimum value allowed for learning rate
      initialRate % initial learning rate for all connections
      
      rates % current learning rates at each connection
      
      % previous model.gradient - used to determine whether to increase
      % learning rate (if current and previous gradients have the same
      % sign) or decrease learning rate (previous and current gradients
      % differ in sign)
      prevGrad 
   end
   
   methods
      function obj = Rprop(initialRate, varargin)
         p = inputParser();
         p.addParamValue('upFactor', 1.2);
         p.addParamValue('downFactor', .5);
         p.addParamValue('maxRate', 5);
         p.addParamValue('minRate', 1e-6);
         parse(p, varargin{:});
         
         obj.initialRate = initialRate;
         obj.upFactor = p.Results.upFactor;
         obj.downFactor = p.Results.downFactor;
         obj.maxRate = p.Results.maxRate;
         obj.minRate = p.Results.minRate;
      end
      
      
      function take_step(obj, x, t, model, ~)
         % Compares current model gradient with previous gradient and
         % independently adjusts learning rates for each model parameter
         % appropriately. Then step = learnRate*sign(gradient).
         grad = model.gradient(x, t);
         obj.update_rates(grad, model);
         steps = cellfun(@(rate, grad) rate.*sign(grad), obj.rates, grad, ...
                          'UniformOutput', false);
         model.increment_params(steps);
         obj.prevGrad = grad;
      end
      
      function update_rates(obj, grad, model)
         % Increases learning rates on connections where progress is being
         % made (gradient has not changed sign), decreases learning rate on
         % connections where gradient has changed sign.
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
         % Clear learned adaptive rates and previous gradient information
         obj.rates = [];
         obj.prevGrad = [];
      end
   end
end

