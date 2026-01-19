/**
 * Custom hook for fetching model metadata using TanStack Query.
 */
import { useQuery } from '@tanstack/react-query';
import { apiClient, MetadataResponse } from '../api/client';

export function useMetadata() {
  return useQuery<MetadataResponse, Error>({
    queryKey: ['metadata'],
    queryFn: () => apiClient.getMetadata(),
    staleTime: 5 * 60 * 1000,
    retry: 2,
  });
}
