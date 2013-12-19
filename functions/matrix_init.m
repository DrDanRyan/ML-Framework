function value = matrix_init(M, N, initType, initScale, gpuState)

   if nargin < 5
      gpuState = GPUState();
   end

   switch initType
      case 'dense'
         if isempty(initScale)
            radius = 1/N;
         else
            radius = initScale;
         end
         value = 2*radius*gpuState.rand([M,N]) - radius;
      case 'relu'
         if isempty(initScale)
            radius = 1/N;
         else
            radius = initScale;
         end
         value = 2*radius*gpuState.rand([M,N]) - radius;
      case 'sparse'
         if isempty(initScale)
            nConnections = min(N/2, 15);
         else
            nConnections = initScale;
         end
         value = gpuState.zeros([M,N]);
         for i = 1:M
            value(i, randperm(N, nConnections)) = ...
               2*gpuState.rand([1, nConnections])/nConnections - 1/nConnections;
         end
      case 'positive'
         if isempty(initScale)
            width = 1/N;
         else
            width = initScale;
         end
         value = width*gpuState.rand([M, N]);
      case 'symmetric positive'
         if isempty(initScale)
            width = 1/N;
         else
            width = initScale;
         end
         W = width*gpuState.rand([M, N]);
         value = (W + W')/2;
      otherwise
         exception = MException('VerifyInput:UnsupportedOption', ...
         sprintf('Unsupported initType: %s', initType));
         throw(exception);
   end

end

