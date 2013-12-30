classdef CompositeHiddenLayer < HiddenLayer
   
   properties
      layers
      gradShape
      gradLength
   end
   
   methods
      function init_params(obj)
         for i = 1:length(obj.layers)
            if ismethod(obj.layers{i}, 'init_params')
               obj.layers{i}.init_params();
            end
         end
      end
      
      function y = feed_forward(obj, x, isSave)
         if nargin < 3
            isSave = false;
         end
         
         y = obj.layers{1}.feed_forward(x, isSave);
         for i = 2:length(obj.layers)
            y = obj.layers{i}.feed_forward(y, isSave);
         end
      end
      
      function [grad, dLdx] = backprop(obj, x, ~, dLdy)
         if isempty(obj.gradShape)
            obj.compute_grad_shape();
         end
         grad = cell(1, obj.gradLength);
         
         stopIdx = obj.gradLength;
         if obj.gradShape(end) == 0
            dLdx = obj.layers{end}.backprop(dLdy);
         else
            startIdx = stopIdx - obj.gradShape(end) + 1;
            [grad(startIdx:stopIdx), dLdx] = obj.layers{end}.backprop(dLdy);
            stopIdx = startIdx - 1;
         end
         
         for i = length(obj.layers)-1:-1:2
            if obj.gradShape(i) == 0
               dLdx = obj.layers{i}.backprop(dLdx);
            else
               startIdx = stopIdx - obj.gradShape(i) + 1;
               [grad(startIdx:stopIdx), dLdx] = obj.layers{i}.backprop(dLdx);
               stopIdx = startIdx - 1;
            end
         end
         
         if obj.gradShape(1) == 0
            dLdx = obj.layers{1}.backprop(x, dLdx);
         else
            [grad(1:stopIdx), dLdx] = obj.layers{1}.backprop(x, dLdx);
         end
      end
      
      function compute_grad_shape(obj)
         nLayers = length(obj.layers);
         obj.gradShape = zeros(1, nLayers);
         for i = 1:nLayers
            if isprop(obj.layers{i}, 'params')
               obj.gradShape(i) = length(obj.layers{i}.params);
            end
         end
         obj.gradLength = sum(obj.gradShape);
      end
      
      function increment_params(obj, delta)
         startIdx = 1;
         for i = 1:length(obj.layers)
            if obj.gradShape(i) > 0
               stopIdx = startIdx + obj.gradShape(i) - 1;               
               obj.layers{i}.increment_params(delta(startIdx:stopIdx));
               startIdx = stopIdx+1;
            end            
         end
      end
      
      function gather(obj)
         for i = 1:length(obj.layers)
            if obj.gradShape(i) > 0
               obj.layers{i}.gather();
            end            
         end
      end
      
      function push_to_GPU(obj)
         for i = 1:length(obj.layers)
            if obj.gradShape(i) > 0
               obj.layers{i}.push_to_GPU();
            end            
         end
      end
      
      function objCopy = copy(obj)
         objCopy = CompositeHiddenLayer();
         nLayers = length(obj.layers);
         objCopy.layers = cell(1, nLayers);
         for i = 1:nLayers
            objCopy.layers{i} = obj.layers{i}.copy();
         end
         objCopy.gradShape = obj.gradShape;
         objCopy.gradLength = obj.gradLength;
      end
      
   end
end

