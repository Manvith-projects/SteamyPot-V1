"""
app.py  --  Flask API for Dynamic Surge Pricing
================================================
Exposes a single endpoint that accepts real-time delivery context and
returns a fully computed pricing decision.

Endpoint
--------
POST /calculate-price
  Input  (JSON):
    {
      "hour": 19,
      "day_of_week": 5,
      "is_holiday": 0,
      "weather": "Rain",
      "traffic_level": 4,
      "active_orders": 85,
      "available_riders": 22,
      "avg_prep_time_min": 20.0,
      "zone_id": 7,
      "distance_km": 8.5,
      "hist_demand_trend": 1.2,
      "hist_cancel_rate": 0.08,
      "base_delivery_fee": 45.0     (optional -- auto-calculated from distance if omitted)
    }

  Output (JSON):
    {
      "surge_multiplier": 1.65,
      "final_delivery_fee": 74.25,
      "recommended_discount": 0.0,
      "pricing_reason": "High demand in your area -- moderate surge applied; Peak hour detected",
      "is_peak_hour": true
    }

Business design
---------------
* The endpoint is stateless -- it does not touch a database.  All
  context needed to price comes in the request body so the API can be
  horizontally scaled behind a load balancer.
* The safety layer (safety_layer.py) is applied *after* ML prediction
  to enforce hard business caps (surge <= 2.5x, discount <= 30 %).
* Model artefacts (joblib) are loaded once at startup and kept in
  memory for sub-millisecond inference.
"""

import os
import json
import traceback
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

from feature_engineering import engineer_features, ALL_NUMERIC, CATEGORICAL
from safety_layer import apply_safety_rules, BASE_FEE_DEFAULT

# ---------------------------------------------------------------------------
# App & model loading
# ---------------------------------------------------------------------------
app = Flask(__name__)

OUTPUT_DIR = "outputs"

# Load artefacts at import time (kept in module globals)
_reg_model = None
_clf_model = None
_transformer = None

def _load_models():
    """Load ML artefacts into globals. Called once at startup."""
    global _reg_model, _clf_model, _transformer
    _reg_model = joblib.load(os.path.join(OUTPUT_DIR, "best_regression_model.joblib"))
    _clf_model = joblib.load(os.path.join(OUTPUT_DIR, "best_classification_model.joblib"))
    _transformer = joblib.load(os.path.join(OUTPUT_DIR, "transformer.joblib"))
    print("[app] Models loaded successfully.")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Simple liveness probe."""
    return jsonify({"status": "ok", "models_loaded": _reg_model is not None})


# ---------------------------------------------------------------------------
# Main pricing endpoint
# ---------------------------------------------------------------------------

@app.route("/calculate-price", methods=["POST"])
def calculate_price():
    """
    Accept a JSON payload with delivery context, run the ML pipeline,
    apply safety rules, and return a pricing decision.

    Steps:
      1. Parse & validate input.
      2. Wrap into a single-row DataFrame.
      3. Run feature engineering (cyclical encoding, ratios, etc.).
      4. Transform through the fitted sklearn ColumnTransformer.
      5. Predict surge_multiplier (regression) and is_peak_hour (classification).
      6. Apply safety layer (caps, discount logic, reason string).
      7. Return JSON response.
    """
    try:
        data = request.get_json(force=True)

        # --- 1. Validate required fields -----------------------------------
        required = [
            "hour", "day_of_week", "is_holiday", "weather",
            "traffic_level", "active_orders", "available_riders",
            "avg_prep_time_min", "zone_id", "distance_km",
            "hist_demand_trend", "hist_cancel_rate",
        ]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        base_fee = float(data.get("base_delivery_fee", BASE_FEE_DEFAULT))

        # If no explicit base_fee, compute distance-adjusted default:
        #   formula mirrors dataset_generator: 25 + 1.5*zone + 2*distance
        if "base_delivery_fee" not in data:
            zone = float(data.get("zone_id", 7))
            dist = float(data.get("distance_km", 5))
            base_fee = round(25.0 + 1.5 * zone + 2.0 * dist, 2)
            base_fee = max(20, min(90, base_fee))

        # --- 2. Single-row DataFrame --------------------------------------
        row = {k: [data[k]] for k in required}
        df = pd.DataFrame(row)

        # --- 3. Feature engineering ----------------------------------------
        df_eng = engineer_features(df)

        # --- 4. Transform --------------------------------------------------
        X = _transformer.transform(df_eng)

        # --- 5. Predict ----------------------------------------------------
        raw_surge = float(_reg_model.predict(X)[0])
        is_peak = int(_clf_model.predict(X)[0])

        # Demand/supply ratio for discount calibration
        ds_ratio = float(df_eng["demand_supply_ratio"].iloc[0])
        dist_km = float(data["distance_km"])

        # --- 6. Safety layer -----------------------------------------------
        decision = apply_safety_rules(
            raw_surge=raw_surge,
            is_peak=is_peak,
            base_fee=base_fee,
            demand_supply_ratio=ds_ratio,
            distance_km=dist_km,
        )

        # --- 7. Response ---------------------------------------------------
        return jsonify({
            "surge_multiplier":     decision.surge_multiplier,
            "final_delivery_fee":   decision.final_delivery_fee,
            "recommended_discount": decision.recommended_discount,
            "pricing_reason":       decision.pricing_reason,
            "is_peak_hour":         decision.is_peak_hour,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _load_models()
    print("[app] Starting Flask server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
