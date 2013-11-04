function value = matrix_init(M, N, initType, initScale, gpuState)

   if nargin < 5
      gpuState = GPUState();
   end

   switch initType
      case 'dense'
         if isempty(initScale)
            radius = sqrt(6/(M + N));
         else
            radius = initScale;
         end
         value = 2*radius*gpuState.rand([M,N]) - radius;
      case 'sparse'
         if isempty(initScale)
            nConnections = 15;
         else
            nConnections = initScale;
         end
         value = gpuState.zeros([M,N]);
         for i = 1:M
            value(i, randperm(N, nConnections)) = gpuState.randn(1, nConnections);
         end
      case 'positive'
         if isempty(initScale)
            width = sqrt(6/(M + N));
         else
            width = initScale;
         end
         value = width*gpuState.rand([M, N]);
      case 'symmetric positive'
         if isempty(initScale)
            width = sqrt(6/(M + N));
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

