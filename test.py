from src.data_management import load_model

pipeline_path = "outputs/ml_pipeline/predict_saleprice/v1"
price_pipeline = load_model(f"{pipeline_path}/best_regressor_pipeline.pkl")

print("Kolumner som användes för träning:")
try:
    print(price_pipeline.feature_names_in_)
    print(f"Antal kolumner: {len(price_pipeline.feature_names_in_)}")
except AttributeError:
    print("Pipeline har inte attributet 'feature_names_in_'. Kontrollera om detta är en sklearn-kompatibel pipeline.")

print("Steps in pipeline:")
print(price_pipeline.named_steps)


print("Kolumner i input-data (X_live):")
print(X_live.columns.tolist())
print(f"Antal kolumner i X_live: {X_live.shape[1]}")

