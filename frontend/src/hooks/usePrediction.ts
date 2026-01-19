/**
 * Custom hook for price prediction using TanStack Query mutation.
 */
import { useMutation } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import type { PredictionRequest, PredictionResponse } from '../api/client';

export function usePrediction() {
  return useMutation<PredictionResponse, Error, PredictionRequest>({
    mutationFn: (request: PredictionRequest) => apiClient.predictPrice(request),
  });
}
