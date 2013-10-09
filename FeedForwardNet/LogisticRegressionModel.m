classdef LogisticRegressionModel < SupervisedModel
   %UNTITLED2 Summary of this class goes here
   %   Detailed explanation goes here
   
   properties
      outputLayer
      inputDropout
   end
   
   methods
      function obj = LogisticRegressionModel(inputSize, inputDropout, varargin)
         if nargin == 0
            return;
         end
         
         if nargin < 2
            inputDropout = 0;
         end
         obj.outputLayer = LogisticOutputLayer(inputSize, varargin{:});
         obj.inputDropout = inputDropout;
      end
      
      function [grad, output] = gradient(obj, x, t)
         if obj.inputDropout > 0
            x = x.*obj.outputLayer.gpuState.binary_mask(size(x), obj.inputDropout);
         end
         [grad, output] = obj.outputLayer.backprop(x, t);
      end
      
      function y = output(obj, x)
         if obj.inputDropout > 0
            x = (1-obj.inputDropout)*x;
         end
         y = obj.outputLayer.feed_forward(x);
      end
      
      function loss = compute_loss(obj, y, t)
         loss = obj.outputLayer.compute_loss(y, t);
      end
      
      function increment_params(obj, delta)
         obj.outputLayer.increment_params(delta);
      end
      
      function gather(obj)
         obj.outputLayer.gather();
      end
      
      function push_to_GPU(obj) % push data from main memory onto GPU memory
         obj.outputLayer.push_to_GPU();
      end
      
      function objCopy = copy(obj) % make an identical copy of the current model
         objCopy = LogisticRegressionModel();
         objCopy.outputLayer = obj.outputLayer.copy();
         objCopy.inputDropout = obj.inputDropout; 
      end
      
      function reset(obj) % reinitialize all the model parameters to initial random states
         obj.outputLayer.init_params();
      end
   end
   
end

