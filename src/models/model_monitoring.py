import mlflow


def log_metrics(metrics: dict):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def check_metrics(metrics: dict, thresholds: dict):
    alerts = {}
    for key, value in metrics.items():
        if key in thresholds and value < thresholds[key]:
            alerts[key] = value
    return alerts
