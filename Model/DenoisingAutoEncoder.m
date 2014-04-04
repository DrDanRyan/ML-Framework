classdef DenoisingAutoEncoder < AutoEncoder
   % Extends AutoEncoder by injecting noise into the inputs before the feed
   % forward pass begins. Currently, dropout, 'salt and pepper', and Gaussian 
   % noise are supported.
   
   properties     
      % string indicating type of input noise to use: 
      % 'none', 'dropout', 'salt and pepper', or 'Gaussian'
      noiseType 
      
      % a scalar indicating the level of noise (i.e. dropout prob, or std_dev)
      noiseLevel 
   end
   
   methods
      function obj = DenoisingAutoEncoder(varargin)
         obj = obj@AutoEncoder(varargin{:});
         p = inputParser;
         p.KeepUnmatched = true;
         p.addParamValue('noiseType', 'none');
         p.addParamValue('noiseLevel', .2);
         parse(p, varargin{:});
         
         obj.noiseType = p.Results.noiseType;
         obj.noiseLevel = p.Results.noiseLevel;
      end
      
      function [grad, xRecon] = gradient(obj, batch)
         x = batch{1};
         xNoisy = obj.inject_noise(x);
         h = obj.encodeLayer.feed_forward(xNoisy, true);
         [decodeGrad, dLdh, xRecon] = obj.decodeLayer.backprop(h, x);
         encodeGrad = obj.encodeLayer.backprop(xNoisy, h, dLdh);
         
         if obj.isTiedWeights
            grad = obj.tied_weights_grad(encodeGrad, decodeGrad);
         else
            obj.encodeGradSize = length(encodeGrad);
            grad = [encodeGrad, decodeGrad];
         end
      end
      
      function xNoisy = inject_noise(obj, x)
         switch obj.noiseType
            case 'dropout'
               xNoisy = x.*obj.gpuState.binary_mask(size(x), obj.noiseLevel);
            case 'salt and pepper'
               noiseIdx = obj.gpuState.binary_mask(size(x), 1-obj.noiseLevel);
               noiseVals = obj.gpuState.binary_mask([sum(noiseIdx(:)), 1], .5);
               xNoisy(logical(noiseIdx)) = noiseVals;
            case 'Gaussian'
               xNoisy = x + obj.noiseLevel*obj.gpuState.randn(size(x));
            otherwise
               % do nothing
         end
      end
      
      function objCopy = copy(obj)
         objCopy = DenoisingAutoEncoder();
         
         % Handle properties
         objCopy.encodeLayer = obj.encodeLayer.copy();
         objCopy.decodeLayer = obj.decodeLayer.copy();
         
         % Value properties
         objCopy.isTiedWeights = obj.isTiedWeights;
         objCopy.gpuState = obj.gpuState;
         objCopy.noiseType = obj.noiseType;
         objCopy.noiseLevel = obj.noiseLevel;
      end
   end
   
end

