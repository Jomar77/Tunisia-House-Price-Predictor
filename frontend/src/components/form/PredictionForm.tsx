/**
 * Main prediction form component.
 */
import { useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import { usePrediction } from '../../hooks/usePrediction';
import { useMetadata } from '../../hooks/useMetadata';
import type { PredictionRequest } from '../../api/client';

function ColdStartLoading(props: { messages: string[] }) {
  const [loadingTick, setLoadingTick] = useState(0);

  useEffect(() => {
    const interval = window.setInterval(() => {
      setLoadingTick((tick) => tick + 1);
    }, 2000);

    return () => window.clearInterval(interval);
  }, []);

  const message = props.messages[loadingTick % props.messages.length];
  const seconds = loadingTick * 2;

  return (
    <div className="loading-panel" aria-live="polite" aria-busy="true">
      <div key={loadingTick} className="loading-message loading-fade-in">
        {message}
      </div>

      {seconds >= 30 && (
        <div className="loading-tip loading-fade-in">
          Tip: If this takes more than 30 seconds, try reloading the page.
        </div>
      )}
    </div>
  );
}

export function PredictionForm() {
  const { data: metadata, isLoading: metadataLoading } = useMetadata();
  const mutation = usePrediction();

  const validationRanges = {
    area: { min: 20, max: 22000 },
    rooms: { min: 1, max: 50 },
    bathrooms: { min: 1, max: 20 },
    age: { min: 0, max: 100 },
  } as const;

  const [formData, setFormData] = useState<PredictionRequest>({
    area: 100,
    rooms: 3,
    bathrooms: 2,
    age: 5,
    location: '',
  });
  const [validationError, setValidationError] = useState<string | null>(null);

  const loadingMessages = useMemo(
    () => [
      'Warming up the backend (cold start)…',
      'Fetching model metadata…',
      'First request can take a few seconds…',
      'Almost there…',
    ],
    []
  );

  const validateForm = (data: PredictionRequest): string | null => {
    if (data.area < validationRanges.area.min || data.area > validationRanges.area.max) {
      return `Area must be between ${validationRanges.area.min} and ${validationRanges.area.max}.`;
    }
    if (data.rooms < validationRanges.rooms.min || data.rooms > validationRanges.rooms.max) {
      return `Rooms must be between ${validationRanges.rooms.min} and ${validationRanges.rooms.max}.`;
    }
    if (
      data.bathrooms < validationRanges.bathrooms.min ||
      data.bathrooms > validationRanges.bathrooms.max
    ) {
      return `Bathrooms must be between ${validationRanges.bathrooms.min} and ${validationRanges.bathrooms.max}.`;
    }
    if (data.age < validationRanges.age.min || data.age > validationRanges.age.max) {
      return `Age must be between ${validationRanges.age.min} and ${validationRanges.age.max}.`;
    }
    if (!data.location.trim()) {
      return 'Please select a location.';
    }
    return null;
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const error = validateForm(formData);
    if (error) {
      setValidationError(error);
      return;
    }
    setValidationError(null);
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
    return <ColdStartLoading messages={loadingMessages} />;
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
            min={validationRanges.area.min}
            max={validationRanges.area.max}
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
            min={validationRanges.rooms.min}
            max={validationRanges.rooms.max}
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
            min={validationRanges.bathrooms.min}
            max={validationRanges.bathrooms.max}
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
            min={validationRanges.age.min}
            max={validationRanges.age.max}
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

        {validationError && (
          <div className="error">
            <strong>Error:</strong> {validationError}
          </div>
        )}

        <button type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? 'Predicting...' : 'Predict Price'}
        </button>

        {mutation.isPending && (
          <ColdStartLoading messages={loadingMessages} />
        )}
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
