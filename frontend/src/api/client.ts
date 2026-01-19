/**
 * API client using native fetch.
 * Handles communication with the FastAPI backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export interface PredictionRequest {
  area: number;
  rooms: number;
  bathrooms: number;
  age: number;
  location: string;
}

export interface PredictionResponse {
  predicted_price: number;
  inputs: {
    area: number;
    rooms: number;
    bathrooms: number;
    age: number;
    location: string;
  };
}

export interface MetadataResponse {
  numeric_features: string[];
  supported_locations: string[];
  model_version: string;
  n_features: number;
}

class APIClient {
  private baseURL: string;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }

  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
    }
    return response.json();
  }

  async predictPrice(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await fetch(`${this.baseURL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    return this.handleResponse<PredictionResponse>(response);
  }

  async getMetadata(): Promise<MetadataResponse> {
    const response = await fetch(`${this.baseURL}/metadata`);
    return this.handleResponse<MetadataResponse>(response);
  }

  async healthCheck(): Promise<{ status: string; service: string; model_loaded: boolean }> {
    const response = await fetch(`${this.baseURL}/health`);
    return this.handleResponse(response);
  }
}

export const apiClient = new APIClient(API_BASE_URL);
