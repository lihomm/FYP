from lime import lime_tabular

# Function to generate LIME explanation
def get_lime_explanation(model, processed_data, index):
    # Exclude the target variable if it's included in the processed_data
    features = processed_data.drop(columns=['isFraud'], errors='ignore')

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=features.to_numpy(),
        feature_names=features.columns.tolist(),
        class_names=['Non-Fraud', 'Fraud'],
        mode='classification'
    )
    # Select the instance to explain
    instance = features.iloc[index].to_numpy().reshape(1, -1)
    predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
    exp = explainer.explain_instance(instance[0], predict_fn, num_features=10)
    return {"explanation": exp.as_list(), "interpretation": exp.as_html()}


