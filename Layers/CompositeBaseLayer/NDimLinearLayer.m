classdef NDimLinearLayer < CompositeBaseLayer & ParamsFunctions & ...
                           matlab.mixin.Copyable & WeightDecayPenalty & ...
                           MaxFanInConstraint
   % A layer that works like multiple linear layers all applied to the same
   % input resulting in a 3D output with dimensions: outputSize x batchSize x D.
   % This is like a maxout layer but with no nonlinearity applied to the
   % resulting groups of units (i.e. maxout would then take max(output, [], 3)).
   
   properties
      % params = {W, b} where W and b are 3-dimensional arrays
      inputSize
      outputSize
      
      % number of linear units per hidden unit grouping
      % (i.e. size of 3rd dimension of W and b)
      D  
   end
   
   methods
      function obj = NDimLinearLayer(inputSize, outputSize, D, varargin)
         obj = obj@ParamsFunctions(varargin{:});
         obj = obj@WeightDecayPenalty(varargin{:});
         obj = obj@MaxFanInConstraint(varargin{:});
         obj.inputSize = inputSize;
         obj.outputSize = outputSize;
         obj.D = D;   
         obj.init_params();
      end
      
      function init_params(obj)
         obj.params{2} = obj.gpuState.zeros(obj.outputSize, 1, obj.D);
         for idx = 1:obj.D
            obj.params{1}(:,:,idx) = matrix_init(obj.outputSize, ...
               obj.inputSize, obj.initType, obj.initScale, obj.gpuState);
         end
      end
      
      function y = feed_forward(obj, x, ~)
         y = obj.compute_z(x);
      end
      
      function [grad, dLdx] = backprop(obj, x, dLdy)
         N = size(x, 2);         
         dLdx = ...
            sum(pagefun(@mtimes, permute(obj.params{1}, [2, 1, 3]), dLdy), 3);
         grad{1} = pagefun(@mtimes, dLdy, x')/N; % L2 x L1 x D
         grad{2} = mean(dLdy, 2); % L2 x 1 x D
         
         if obj.isWeightDecay
            grad{1} = grad{1} + obj.compute_weight_decay_penalty();
         end
      end
      
      function value = compute_z(obj, x)
         % z has dimensions L2 x N x D
         value = pagefun(@mtimes, obj.params{1}, x);
         value = bsxfun(@plus, value, obj.params{2});
      end
      
      function increment_params(obj, delta)
         increment_params@ParamsFunctions(obj, delta);
         if obj.isMaxFanIn
            obj.impose_fanin_constraint();
         end
      end
      
   end
end

