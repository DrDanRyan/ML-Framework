classdef FeedForwardNet < Model
% A general base class for a feed-forward neural network that can utilize
% dropout and gpu training. Implements the Model interface.

   properties
      hiddenLayers % cell array of HiddenLayer objects
      outputLayer % a single OutputLayer object
      gpuState % gpuState object used for array creation dependent on isGPU flag in object
      isDropout % boolean indicating wheter to use dropout
      hiddenDropout % proportion of hidden units that are replaced with zero
      inputDropout % proportion of inputs that are replaced with zero (in [0, 1])
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
         obj.dropout = p.Results.hiddenDropout;
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
         % delta_params is a single layer (non-nested) cell array of the
         % same shape produced by gradient. Gradients are provided in
         % reverse order (outputLayer first, followed by last hiddenLayer
         % etc) as this is the order they were computed with backprop
         
         stopIdx = length(delta_params); 
         for i = 1:length(obj.hiddenLayers)
            nParams = length(obj.hiddenLayers{i}.params);
            startIdx = stopIdx - nParams + 1;
            obj.hiddenLayers{i}.increment_params(delta_params(startIdx:stopIdx));
            stopIdx = startIdx - 1;
         end
         obj.outputLayer.increment_params(delta_params(1:stopIdx));
      end
      
      function [grad, output] = gradient(obj, x, t)
         nHiddenLayers = length(obj.hiddenLayers);
         y = cell(nHiddenLayers, 1); % output from each hiddenLayer
         dLdy = cell(nHiddenLayers, 1); % derivative of loss function wrt hiddenLayer output
         
         if obj.isDropout
            [x, mask] = obj.dropout_mask(x);
         end
         
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
         
         [grad, dLdy{end}, output] = obj.outputLayer.backprop(y{end}, t);
         if obj.isDropout
            dLdy{end} = dLdy{end}.*mask{end};
         end
         
         for i = nHiddenLayers:-1:2
            [gradTEMP, dLdy{i-1}] = obj.hiddenLayers{i}.backprop(y{i-1}, y{i}, dLdy{i});
            grad = [grad, gradTEMP]; %#ok<AGROW>
            if obj.isDropout
               dLdy{i-1} = dLdy{i-1}.*mask{i-1};
            end
         end
         gradTEMP = obj.hiddenLayers{1}.backprop(x, y{1}, dLdy{1});
         grad = [grad, gradTEMP];
      end
      
      function [x, mask] = dropout_mask(obj, x)   
         
         % Dropout inputs
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
      end
      
      function reset(obj)
         for i = 1:length(obj.hiddenLayers)
            obj.hiddenLayers{i}.init_params();
         end
         obj.outputLayer.init_params();
      end
   end
end

