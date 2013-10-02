classdef FeedForwardNet < SupervisedModel
% A concrete implementation of the Abstract SupervisedModel class.
% A general base class for a feed-forward neural network that can utilize
% dropout and gpu training. 

   properties
      hiddenLayers % cell array of HiddenLayer objects
      outputLayer % a single OutputLayer object
      gpuState % gpuState object used for array creation dependent on isGPU flag in object
      isDropout % boolean indicating wheter to use dropout
      hiddenDropout % proportion of hidden units that are replaced with zero (in [0, 1])
      inputDropout % proportion of inputs that are replaced with zero (in [0, 1])
      nestedGradShape % a row vector of integer values describing the nested gradient shape
      flatGradLength % the length of the flattened gradient
   end
   
   methods
      function obj = FeedForwardNet(varargin)
         p = inputParser;
         p.addParamValue('isDropout', true);
         p.addParamValue('hiddenDropout', .5);
         p.addParamValue('inputDropout', 0);
         p.addParamValue('gpu', []);         
         parse(p, varargin{:});
         
         obj.isDropout = p.Results.isDropout;
         obj.hiddenDropout = p.Results.hiddenDropout;
         obj.inputDropout = p.Results.inputDropout;
         
         if isempty(p.Results.gpu)
            obj.gpuState = GPUState();
         else
            obj.gpuState = GPUState(p.Results.gpu);
         end
      end
      
      function gather(obj)
         for i = 1:length(obj.hiddenLayers);
            obj.hiddenLayers{i}.gather();
         end
         obj.outputLayer.gather();
         obj.gpuState.isGPU = false;
      end
      
      function push_to_GPU(obj)
         for i = 1:length(obj.hiddenLayers)
            obj.hiddenLayers{i}.push_to_GPU();
         end
         obj.outputLayer.push_to_GPU();
         obj.gpuState.isGPU = true;
      end      
      
      function increment_params(obj, delta_params)
         % delta_params is a flat cell array; it must be rolled into proper
         % shape (as originally produced during the layer by layer gradient
         % computation) before updating each layer params.
         
         delta_params = obj.roll_gradient(delta_params);
         for i = 1:length(obj.hiddenLayers)
            obj.hiddenLayers{i}.increment_params(delta_params{i});
         end
         obj.outputLayer.increment_params(delta_params{end});
      end
      
      function [grad, output, varargout] = gradient(obj, x, t, varargin)
         % Computes the gradient for batch input x and target t for all parameters in
         % each hiddenLayer and outputLayer.
         p = inputParser;
         p.addParamValue('gradSquared', false);
         parse(p, varargin{:});
         
         if obj.isDropout
            [x, mask] = obj.dropout_mask(x);
         else
            mask = [];
         end
         
         % feed_forward through hiddenLayers
         y = obj.feed_forward(x, mask);
         
         % get outputLayer output and backpropagate loss
         if p.Results.gradSquared
            [grad, output, gradSquared, nonZeroTerms] = ...
               backprop_with_variance(obj, x, y, t, mask);
            varargout = {gradSquared, nonZeroTerms};
         else
            [grad, output] = backprop(obj, x, y, t, mask);
         end
      end
      
      function y = feed_forward(obj, x, mask)
         % feed_forward through hiddenLayers
         nHiddenLayers = length(obj.hiddenLayers);
         y = cell(1, length(obj.hiddenLayers)); % output from each hiddenLayer
         y{1} = obj.hiddenLayers{1}.feed_forward(x);
         if obj.isDropout
            y{1} = y{1}.*mask{1};
         end

         for i = 2:nHiddenLayers
            y{i} = obj.hiddenLayers{i}.feed_forward(y{i-1});
            if obj.isDropout
               y{i} = y{i}.*mask{i};
            end
         end
      end
      
      function [grad, output] = backprop(obj, x, y, t, mask)
         nHiddenLayers = length(obj.hiddenLayers);
         dLdy = cell(1, nHiddenLayers); % derivative of loss function wrt hiddenLayer output
         grad = cell(1, nHiddenLayers+1); % gradient of hiddenLayers and outputLayer (last idx)
         [grad{end}, dLdy{end}, output] = obj.outputLayer.backprop(y{end}, t);
         if obj.isDropout
            dLdy{end} = dLdy{end}.*mask{end};
         end
         
         for i = nHiddenLayers:-1:2
            [grad{i}, dLdy{i-1}] = obj.hiddenLayers{i}.backprop(y{i-1}, y{i}, dLdy{i});
            if obj.isDropout
               dLdy{i-1} = dLdy{i-1}.*mask{i-1};
            end
         end
         grad{1} = obj.hiddenLayers{1}.backprop(x, y{1}, dLdy{1});
         grad = obj.unroll_gradient(grad);
      end
      
      function [grad, output, gradSquared, nonZeroTerms] = ...
            backprop_with_variance(obj, x, y, t, mask)
         nHiddenLayers = length(obj.hiddenLayers);
         dLdy = cell(nHiddenLayers, 1); % derivative of loss function wrt hiddenLayer output
         grad = cell(1, nHiddenLayers+1); % gradient of hiddenLayers and outputLayer (last idx)
         gradSquared = cell(1, nHiddenLayers+1); % mean (over the minibatch) of the gradient squared
         nonZeroTerms = cell(1, nHiddenLayers+1); % number of non-zero terms in minibatch for each gradient component
         
         [grad{end}, dLdy{end}, output, gradSquared{end}, nonZeroTerms{end}] = ...
             obj.outputLayer.backprop_with_variance(y{end}, t);
         if obj.isDropout
            dLdy{end} = dLdy{end}.*mask{end};
         end
         
         for i = nHiddenLayers:-1:2
            [grad{i}, dLdy{i-1}, gradSquared{i}, nonZeroTerms{i}] = ...
               obj.hiddenLayers{i}.backprop_with_variance(y{i-1}, y{i}, dLdy{i});
            if obj.isDropout
               dLdy{i-1} = dLdy{i-1}.*mask{i-1};
            end
         end
         [grad{1}, ~, gradSquared{1}, nonZeroTerms{1}] = ...
            obj.hiddenLayers{1}.backprop_with_variance(x, y{1}, dLdy{1});
         grad = obj.unroll_gradient(grad);
         gradSquared = obj.unroll_gradient(gradSquared); % same shape as gradient
         nonZeroTerms = obj.unroll_gradient(nonZeroTerms); % same shape as gradient
      end
      
      function [x, mask] = dropout_mask(obj, x)   
         % Computes a binary (0, 1) mask  for both the inputs, x, and each hidden
         % layer. Zeros correspond to the units removed by dropout.
         
         % Apply dropout to the inputs
         x = x.*obj.gpuState.binary_mask(size(x), obj.inputDropout);
         
         % Mask for hidden layer outputs
         nHiddenLayers = length(obj.hiddenLayers);
         mask = cell(nHiddenLayers, 1);
         N = size(x, 2);
         for i = 1:nHiddenLayers
            L = obj.hiddenLayers{i}.outputSize;
            mask{i} = obj.gpuState.binary_mask([L, N], obj.hiddenDropout);
         end
      end
      
      function y = output(obj, x)
         nHiddenLayers = length(obj.hiddenLayers);
         
         if obj.isDropout
            y = (1-obj.inputDropout)*x;
            for i = 1:nHiddenLayers
               y = (1-obj.hiddenDropout)*obj.hiddenLayers{i}.feed_forward(y);
            end
         else
            y = x;
            for i = 1:nHiddenLayers
               y = obj.hiddenLayers{i}.feed_forward(y);
            end
         end
         y = obj.outputLayer.feed_forward(y);
      end
      
      function loss = compute_loss(obj, y, t)
         loss = obj.outputLayer.compute_loss(y, t);
      end
      
      function objCopy = copy(obj)
         objCopy = FeedForwardNet();
         
         % Handle class properties
         objCopy.hiddenLayers = cell(size(obj.hiddenLayers));
         for idx = 1:length(obj.hiddenLayers)
            objCopy.hiddenLayers{idx} = copy(obj.hiddenLayers{idx});
         end
         objCopy.outputLayer = copy(obj.outputLayer);
         
         % Value class properties
         objCopy.gpuState = obj.gpuState;
         objCopy.isDropout = obj.isDropout;
         objCopy.inputDropout = obj.inputDropout;
         objCopy.nestedGradShape = obj.nestedGradShape;
         objCopy.flatGradLength = obj.flatGradLength;
      end
      
      function reset(obj)
         for i = 1:length(obj.hiddenLayers)
            obj.hiddenLayers{i}.init_params();
         end
         obj.outputLayer.init_params();
      end
      
      function flatGrad = unroll_gradient(obj, nestedGrad)
         if isempty(obj.nestedGradShape)
            obj.compute_gradient_shapes(nestedGrad);
         end
         
         flatGrad = cell(1, obj.flatGradLength);
         startIdx = 1;
         for i = 1:length(nestedGrad)
            stopIdx = startIdx + obj.nestedGradShape(i) - 1;
            flatGrad(startIdx:stopIdx) = nestedGrad{i};
            startIdx = stopIdx + 1;
         end
      end
      
      function nestedGrad = roll_gradient(obj, flatGrad)
         startIdx = 1;
         nestedLength = length(obj.nestedGradShape);
         nestedGrad = cell(1, nestedLength);
         for i = 1:nestedLength
            stopIdx = startIdx + obj.nestedGradShape(i) - 1;
            nestedGrad{i} = flatGrad(startIdx:stopIdx);
            startIdx = stopIdx + 1;
         end
      end
      
      function compute_gradient_shapes(obj, nestedGrad)
         flatLength = 0;
         nestedShape = zeros(1, length(nestedGrad));
         for i = 1:length(nestedGrad)
            nestedShape(i) = length(nestedGrad{i});
            flatLength = flatLength + nestedShape(i);
         end
         obj.nestedGradShape = nestedShape;
         obj.flatGradLength = flatLength;
      end
   end
end

