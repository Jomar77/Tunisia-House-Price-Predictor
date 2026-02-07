-- Drift monitoring queries (template)
-- Assumes a table like:
--   prediction_logs(created_at, area, rooms, bathrooms, age, location, predicted_price)

-- 1) Feature distribution summary (last 7d)
SELECT
  date_trunc('day', created_at) AS day,
  COUNT(*) AS n,
  AVG(area) AS area_mean,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY area) AS area_p50,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY area) AS area_p95,
  AVG(predicted_price) AS pred_mean,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY predicted_price) AS pred_p95
FROM prediction_logs
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY 1
ORDER BY 1;

-- 2) Location mix shift (last 7d)
SELECT
  location,
  COUNT(*) AS n,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
FROM prediction_logs
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY 1
ORDER BY n DESC
LIMIT 25;
