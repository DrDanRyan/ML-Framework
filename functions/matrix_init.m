function value = matrix_init(rows, cols, initType, initScale, gpuState)
% Provides various matrix initialization schemes.

   if nargin < 5
      gpuState = GPUState();
   end

   switch initType
      case 'dense'
         if isempty(initScale)
            radius = 1/cols;
         else
            radius = initScale;
         end
         value = 2*radius*gpuState.rand([rows,cols]) - radius;
      case 'relu'
         if isempty(initScale)
            radius = 1/cols;
         else
            radius = initScale;
         end
         value = 2*radius*gpuState.rand([rows,cols]) - radius;
      case 'sparse'
         if isempty(initScale)
            nConnections = min(cols/2, 15);
         else
            nConnections = initScale;
         end
         value = gpuState.zeros([rows,cols]);
         for i = 1:rows
            value(i, randperm(cols, nConnections)) = ...
               2*gpuState.rand([1, nConnections])/nConnections - 1/nConnections;
         end
      case 'positive'
         if isempty(initScale)
            width = 1/cols;
         else
            width = initScale;
         end
         value = width*gpuState.rand([rows, cols]);
      case 'symmetric positive'
         if isempty(initScale)
            width = 1/cols;
         else
            width = initScale;
         end
         W = width*gpuState.rand([rows, cols]);
         value = (W + W')/2;
      otherwise
         exception = MException('VerifyInput:UnsupportedOption', ...
         sprintf('Unsupported initType: %s', initType));
         throw(exception);
   end

end

