function value = matrix_init(M, N, initType, initScale, gpuState)

   if nargin < 5
      gpuState = GPUState();
   end

   if strcmp(initType, 'dense')
      value = dense_init(M, N, initScale, gpuState);
   elseif strcmp(initType, 'sparse')
      value = sparse_init(M, N, initScale, gpuState);
   else
      exception = MException('VerifyInput:UnsupportedOption', ...
      sprintf('Unsupported initType: %s', initType));
      throw(exception);
   end

end

