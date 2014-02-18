classdef DAE < AutoEncoder
   % Denoising AutoEncoder
   
   properties     
      noiseType % string indicating type of input noise to use: 
                % 'none', 'dropout', 'Gaussian'
      noiseLevel % a scalar indicating the level of noise 
                 % (i.e. dropout prob, or std_dev)
   end
   
   methods
      function obj = DAE(varargin)
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
         if obj.isImputeValues
            x = obj.impute_values(batch{1});
         else
            x = batch{1};
         end
         
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
      
      function x = inject_noise(obj, x)
         switch obj.noiseType
            case 'none'
               % do nothing
            case 'dropout'
               x = x.*obj.gpuState.binary_mask(size(x), obj.noiseLevel);
            case 'Gaussian'
               x = x + obj.noiseLevel*obj.gpuState.randn(size(x));
         end
      end
      
      function objCopy = copy(obj)
         objCopy = DAE();
         
         % Handle properties
         objCopy.encodeLayer = obj.encodeLayer.copy();
         objCopy.decodeLayer = obj.decodeLayer.copy();
         
         % Value properties
         objCopy.isTiedWeights = obj.isTiedWeights;
         objCopy.isImputeValues = obj.isImputeValues;
         objCopy.imputeTol = obj.imputeTol;
         objCopy.gpuState = obj.gpuState;
         objCopy.noiseType = obj.noiseType;
         objCopy.noiseLevel = obj.noiseLevel;
      end
   end
   
end

