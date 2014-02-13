classdef DAE < AutoEncoder
   % Denoising AutoEncoder
   
   properties     
      noiseType % string indicating type of input noise to use: 'none', 'dropout', 'Gaussian'
      noiseLevel % a scalar indicating the level of noise (i.e. dropout prob, or std_dev)
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
         xTarget = batch{1}; % noise free, will keep NaN vals
         xIn = batch{1}; % noise will be added and then NaNs replaced with 0
         xIn = obj.inject_noise(xIn);
         [grad, xRecon] = gradient@AutoEncoder(obj, xIn, xTarget, []);
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
         x(isnan(x)) = 0;
      end
      
      function objCopy = copy(obj)
         objCopy = DAE();
         
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

