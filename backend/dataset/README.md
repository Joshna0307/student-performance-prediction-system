### Model evaluation and dataset

The demonstration dataset in `backend/dataset/sample_students.csv` is a tiny sample for testing ingestion. The ANN training script generates a larger synthetic training dataset. It should not be used to make real educational decisions. Model attribution in the current API is an input-summary visualization, not SHAP values. The XAI module, real institutional dataset integration, auth/roles, PDF export, subject marks UI, simulator UI, deployment configuration and production security are remaining implementation milestones.
