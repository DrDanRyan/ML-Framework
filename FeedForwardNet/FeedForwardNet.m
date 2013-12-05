classdef FeedForwardNet < SupervisedModel
% A concrete implementation of the Abstract SupervisedModel class.
% A general base class for a feed-forward neural network that can utilize
% dropout and gpu training. 

   properties
      hiddenLayers % cell array of HiddenLayer objects (possibly empty)
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
         p.addParamValue('hiddenDropout', []);
         p.addParamValue('inputDropout', []);
         p.addParamValue('gpu', []);         
         parse(p, varargin{:});
         
         obj.hiddenDropout = p.Results.hiddenDropout;
         obj.inputDropout = p.Results.inputDropout;
         obj.isDropout = ~isempty(obj.inputDropout) || ~isempty(obj.hiddenDropout);
         obj.gpuState = GPUState(p.Results.gpu);
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
         
         if isempty(obj.hiddenLayers)
            obj.outputLayer.increment_params(delta_params);
            return;
         end
         
         delta_params = obj.roll_gradient(delta_params);
         for i = 1:length(obj.hiddenLayers)
            obj.hiddenLayers{i}.increment_params(delta_params{i});
         end
         obj.outputLayer.increment_params(delta_params{end});
      end
      
      function [grad, output, dLdx] = gradient(obj, batch)
         % Computes the gradient for batch input x and target t for all parameters in
         % each hiddenLayer and outputLayer.
         x = batch{1};
         t = batch{end};
         if obj.isDropout
            mask = obj.dropout_mask(x);
            x = x.*mask{1};
         else
            mask = [];
         end
         
         % feed_forward through hiddenLayers
         [y, ffExtras] = obj.feed_forward(x, mask);
         
         % get outputLayer output and backpropagate loss
         [grad, output, dLdx] = obj.backprop(x, y, t, ffExtras, mask);
      end
      
      function [y, ffExtras] = feed_forward(obj, x, mask)
         if isempty(obj.hiddenLayers)
            y = [];
            ffExtras = [];
            return
         end
         
         % feed_forward through hiddenLayers
         nHiddenLayers = length(obj.hiddenLayers);
         y = cell(1, nHiddenLayers); % output from each hiddenLayer
         ffExtras = cell(1, nHiddenLayers); % z = Wx + b from each hiddenLayer
         [y{1}, ffExtras{1}] = obj.hiddenLayers{1}.feed_forward(x);
         if obj.isDropout
            y{1} = y{1}.*mask{2};
         end

         for i = 2:nHiddenLayers
            [y{i}, ffExtras{i}] = obj.hiddenLayers{i}.feed_forward(y{i-1});
            if obj.isDropout
               y{i} = y{i}.*mask{i+1};
            end
         end
      end
      
      function [grad, output, dLdx] = backprop(obj, x, y, t, ffExtras, mask)
         if isempty(obj.hiddenLayers)
            [grad, dLdx, output] = obj.outputLayer.backprop(x, t);
            return;
         end
         
         nHiddenLayers = length(obj.hiddenLayers);
         grad = cell(1, nHiddenLayers+1); % gradient of hiddenLayers and outputLayer (last idx)
         [grad{end}, dLdx, output] = obj.outputLayer.backprop(y{end}, t);
                     
         if obj.isDropout
            dLdx = dLdx.*mask{end};
            mask{end} = [];
         end
         
         for i = nHiddenLayers:-1:2
            [grad{i}, dLdx] = obj.hiddenLayers{i}.backprop(y{i-1}, y{i}, ffExtras{i}, ...
               dLdx);
            if obj.isDropout
               dLdx = dLdx.*mask{i};
               mask{i} = [];
            end
         end
         [grad{1}, dLdx] = obj.hiddenLayers{1}.backprop(x, y{1}, ffExtras{1}, dLdx);
         if obj.isDropout
            dLdx = dLdx.*mask{1};
            mask{1} = [];
         end
         grad = obj.unroll_gradient(grad);
      end
      
      function mask = dropout_mask(obj, x)   
         % Computes a binary (0, 1) mask  for both the inputs, x, and each hidden
         % layer. Zeros correspond to the units removed by dropout.

         nHiddenLayers = length(obj.hiddenLayers);
         mask = cell(nHiddenLayers+1, 1);
         
         % Input mask
         mask{1} = obj.gpuState.binary_mask(size(x), obj.inputDropout);
         
         % hiddenLayers masks
         N = size(x, 2);
         for i = 1:nHiddenLayers
            L = obj.hiddenLayers{i}.outputSize;
            mask{i+1} = obj.gpuState.binary_mask([L, N], obj.hiddenDropout);
         end
      end
      
      function y = output(obj, x)
         nHiddenLayers = length(obj.hiddenLayers);
         x(isnan(x)) = 0;
         
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
      
      function loss = compute_loss(obj, batch)
         y = obj.output(batch{1});
         t = batch{end};
         loss = obj.compute_loss_from_output(y, t);
      end
      
      function loss = compute_loss_from_output(obj, y, t)
         loss = obj.outputLayer.compute_loss(y, t);
      end
      
      function objCopy = copy(obj)
         objCopy = FeedForwardNet();
         
         % Handle class properties
         objCopy.hiddenLayers = cell(size(obj.hiddenLayers));
         for idx = 1:length(obj.hiddenLayers)
            objCopy.hiddenLayers{idx} = copy(obj.hiddenLayers{idx});
         end
         
         if ~isempty(obj.outputLayer)
            objCopy.outputLayer = copy(obj.outputLayer);
         end
         
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

