classdef HybridOutputLayer < OutputLayer
   % An OutputLayer where multiple output unit types are combined (all
   % sharing the same inputs though). Useful for an AutoEncoder with
   % multiple data types.
   
   properties
      layers % a cell array of OutputLayers
      isLocallyLinear
      isDiagonalDy
   end
   
   methods
      
      function [idxs, nOutputTypes] = output_idxs(obj)
         nOutputTypes = length(obj.layers);
         idxs = obj.gpuState.zeros(nOutputTypes, 2);
         startIdx = 1;
         for i = 1:nOutputTypes
            stopIdx = startIdx + obj.layers{i}.outputSize - 1;
            idxs(i,:) = [startIdx, stopIdx];
            startIdx = stopIdx + 1;
         end
      end
      
      function [grad, dLdx, output] = backprop(obj, x, t)
         [idxs, nOutputTypes] = obj.output_idxs();
         layer = obj.layers{1};
         [grad, dLdx, output] = layer.backprop(x, t(idxs(1,1):idxs(1,2), :));
         for i = 2:nOutputTypes
            layer = obj.layers{i};
            [tempGrad, tempdLdx, tempoutput] = layer.backprop(x, t(idxs(i,1):idxs(i,2),:));
            grad = [grad, tempGrad];
            dLdx = dLdx + tempdLdx;
            output = [output; tempoutput];
         end
      end
      
      function loss = compute_loss(obj, y, t)
         [idxs, nOutputTypes] = obj.output_idxs();
         loss = 0;
         for i = 1:nOutputTypes
            loss = loss + obj.layers{i}.compute_loss(y(idxs(i,1):idxs(i,2),:), ...
                                                      t(idxs(i,1):idxs(i,2),:));
         end
      end
      
      function value = compute_Dy(obj, x, y)
         % pass
      end
      
      function value = compute_D2y(obj, x, y, Dy)
         % pass
      end
      
      function value = compute_z(obj, x)
         layer = obj.layers{1};
         value = layer.compute_z(x);
         for i = 2:length(obj.layers)
            layer = obj.layers{i};
            tempZ = layer.compute_z(x);
            value = [value; tempZ];
         end
      end
      
      function y = feed_forward(x)
         layer = obj.layers{1};
         y = layer.feed_forward(x);
         for i = 2:length(obj.layers)
            layer = obj.layers{i};
            tempY = layer.feed_forward(x);
            y = [y; tempY];
         end
      end
      
      function push_to_GPU(obj)
         for i = 1:length(obj.layers)
            obj.layers{i}.push_to_GPU();
         end
      end
      
      function gather(obj)
         for i = 1:length(obj.layers)
            obj.layers{i}.gather();
         end
      end
      
      function increment_params(obj, delta_params)
         startIdx = 1;
         for i = 1:length(obj.layers)
            layer = obj.layers{i};
            stopIdx = startIdx + length(layer.params) - 1;
            layer.increment_params(delta_params(startIdx:stopIdx));
            startIdx = stopIdx + 1;
         end
      end
      
      function init_params(obj)
         for i = 1:length(obj.layers)
            obj.layers{i}.init_params();
         end
      end
   end
end

