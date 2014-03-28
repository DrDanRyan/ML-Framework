classdef GPUState
   % A state object that handles array creation and random number generation
   % depending on the status of the isGPU flag. All GPU numerical arrays will be
   % single precision.
   
   properties
      isGPU % boolean flag indicating whether GPU is being used
   end
   
   methods
      function obj = GPUState(isGPU)
         % If no parameters given, flag is set based on whether a GPU is
         % detected by MATLAB.
         
         if nargin == 1 && ~isempty(isGPU)
            obj.isGPU = isGPU;
         else
            obj.isGPU = gpuDeviceCount > 0;
         end
      end
      
      function value = zeros(obj, varargin)
         if obj.isGPU
            value = gpuArray.zeros(varargin{:}, 'single');
         else
            value = zeros(varargin{:});
         end
      end
      
      function value = ones(obj, varargin)
         if obj.isGPU
            value = gpuArray.ones(varargin{:}, 'single');
         else
            value = ones(varargin{:});
         end
      end
      
      function value = nan(obj, varargin)
         if obj.isGPU
            value = gpuArray.nan(varargin{:}, 'single');
         else
            value = nan(varargin{:});
         end
      end
      
      function value = false(obj, varargin)
         if obj.isGPU
            value = gpuArray.false(varargin{:});
         else
            value = false(varargin{:});
         end
      end
      
      function value = eye(obj, varargin)
         if obj.isGPU
            value = gpuArray.eye(varargin{:}, 'single');
         else
            value = eye(varargin{:});
         end
      end
      
      function value = rand(obj, varargin)
         if obj.isGPU
            value = gpuArray.rand(varargin{:}, 'single');
         else
            value = rand(varargin{:});
         end
      end
      
      function value = randn(obj, varargin)
         if obj.isGPU
            value = gpuArray.randn(varargin{:}, 'single');
         else
            value = randn(varargin{:});
         end
      end
      
      function value = binary_mask(obj, dims, probOfZero)
         if obj.isGPU
            value = single(gpuArray.rand(dims, 'single') >= probOfZero);
         else
            value = rand(dims) >= probOfZero;
         end
      end
      
      function value = linspace(obj, varargin)
         if obj.isGPU
            value = single(gpuArray.linspace(varargin{:}));
         else
            value = linspace(varargin{:});
         end
      end
   end
end

