/**
 * Main prediction form component.
 */
import { useState, FormEvent } from 'react';
import { usePrediction } from '../hooks/usePrediction';
import { useMetadata } from '../hooks/useMetadata';
import { PredictionRequest } from '../api/client';

export function PredictionForm() {
  const { data: metadata, isLoading: metadataLoading } = useMetadata();
  const mutation = usePrediction();

  const [formData, setFormData] = useState<PredictionRequest>({
    area: 100,
    rooms: 3,
    bathrooms: 2,
    age: 5,
    location: '',
  });

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    mutation.mutate(formData);
  };

  const handleChange = (field: keyof PredictionRequest) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    setFormData((prev) => ({
      ...prev,
      [field]: field === 'location' ? e.target.value : Number(e.target.value),
    }));
  };

  if (metadataLoading) {
    return <div className="loading">Loading model metadata...</div>;
  }

  return (
    <div className="prediction-form">
      <h1>Tunisia House Price Predictor</h1>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="area">Area (m²)</label>
          <input
            id="area"
            type="number"
            min="1"
            step="0.1"
            value={formData.area}
            onChange={handleChange('area')}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="rooms">Rooms</label>
          <input
            id="rooms"
            type="number"
            min="1"
            step="1"
            value={formData.rooms}
            onChange={handleChange('rooms')}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="bathrooms">Bathrooms</label>
          <input
            id="bathrooms"
            type="number"
            min="1"
            step="1"
            value={formData.bathrooms}
            onChange={handleChange('bathrooms')}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="age">Age (years)</label>
          <input
            id="age"
            type="number"
            min="0"
            step="1"
            value={formData.age}
            onChange={handleChange('age')}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="location">Location</label>
          <select
            id="location"
            value={formData.location}
            onChange={handleChange('location')}
            required
          >
            <option value="">Select a location...</option>
            {metadata?.supported_locations.map((loc) => (
              <option key={loc} value={loc}>
                {loc}
              </option>
            ))}
          </select>
        </div>

        <button type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? 'Predicting...' : 'Predict Price'}
        </button>
      </form>

      {mutation.isError && (
        <div className="error">
          <strong>Error:</strong> {mutation.error.message}
        </div>
      )}

      {mutation.isSuccess && mutation.data && (
        <div className="result">
          <h2>Predicted Price</h2>
          <p className="price">
            €{mutation.data.predicted_price.toLocaleString('fr-FR', {
              minimumFractionDigits: 2,
              maximumFractionDigits: 2,
            })}
          </p>
          <div className="inputs-summary">
            <h3>Input Summary</h3>
            <ul>
              <li>Area: {mutation.data.inputs.area} m²</li>
              <li>Rooms: {mutation.data.inputs.rooms}</li>
              <li>Bathrooms: {mutation.data.inputs.bathrooms}</li>
              <li>Age: {mutation.data.inputs.age} years</li>
              <li>Location: {mutation.data.inputs.location}</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
