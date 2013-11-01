function value = matrix_init(M, N, initType, initScale, gpuState)

   if nargin < 5
      gpuState = GPUState();
   end

   switch initType
      case 'dense'
         value = dense_init(M, N, initScale, gpuState);
      case 'sparse'
         value = sparse_init(M, N, initScale, gpuState);
      case 'small positive'
         value = initScale*gpuState.rand([M, N]);
      otherwise
         exception = MException('VerifyInput:UnsupportedOption', ...
         sprintf('Unsupported initType: %s', initType));
         throw(exception);
   end

end

