classdef StandardLayer < ParamsFunctions & matlab.mixin.Copyable & ...
                         MaxFanInConstraint & WeightDecayPenalty
   % A mixin that provides basic functionality for a standard layer
   % consisting of a linear layer (z = W*x + b) followed by a 
   % nonlinear function (y = f(z)).
   
   properties
      % params = {W, b}
      inputSize
      outputSize
   end
   
   methods
      function obj = StandardLayer(inputSize, outputSize, varargin)       
         obj = obj@ParamsFunctions(varargin{:});
         obj = obj@MaxFanInConstraint(varargin{:});
         obj = obj@WeightDecayPenalty(varargin{:});
         obj.inputSize = inputSize;
         obj.outputSize = outputSize;
         obj.init_params();
      end
      
      function init_params(obj)
         obj.params{1} = matrix_init(obj.outputSize, obj.inputSize, ...
                         obj.initType, obj.initScale, obj.gpuState);
         obj.params{2} = obj.gpuState.zeros(obj.outputSize, 1);
      end
      
      function increment_params(obj, delta)
         increment_params@ParamsFunctions(obj, delta);
         if obj.isMaxFanIn
            obj.impose_fanin_constraint();
         end
      end 
      
      function value = compute_z(obj, x)
         value = bsxfun(@plus, obj.params{1}*x, obj.params{2});
      end
      
      function [grad, dLdx] = grad_from_dLdz(obj, x, dLdz)
         grad{1} = dLdz*x'/size(x, 2);
         grad{2} = mean(dLdz, 2);

         if obj.isWeightDecay
            grad{1} = grad{1} + obj.compute_weight_decay_penalty();
         end
         
         dLdx = obj.params{1}'*dLdz;
      end
      
   end
end

