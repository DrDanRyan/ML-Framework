classdef ZeroPadding2D < CompositeHigherLayer & matlab.mixin.Copyable
   % Adds zero padding to the top, bottom, left and right of a multi-channel
   % square 2D input signal. Might be useful for a convolutional autoencoder to
   % perform an 'equal' style convolution using my Conv2D layers that normally
   % perform 'valid' style convolutions.
   
   properties
      rows
      cols
      left
      right
      top
      bottom
      gpuState
   end
   
   methods
      function obj = ZeroPadding2D(left, right, top, bottom)
         obj.left = left;
         obj.right = right;
         obj.top = top;
         obj.bottom = bottom;       
         obj.gpuState = GPUState();
      end
      
      function y = feed_forward(obj, x, ~)
         obj.gpuState.isGPU = isa(x, 'gpuArray');
         [C, N, obj.rows, obj.cols] = size(x);
         y = obj.gpuState.zeros(C, N, obj.rows+obj.top+obj.bottom, ...
                                      obj.cols+obj.left+obj.right);
         y(:,:,obj.top+1:obj.top+obj.rows, obj.left+1:obj.left+obj.cols) = x;
      end
      
      function dLdx = backprop(obj, dLdy)
         dLdx = dLdy(:,:,obj.top+1:obj.top+obj.rows, ...
                         obj.left+1:obj.left+obj.cols);
      end
      
   end
end

