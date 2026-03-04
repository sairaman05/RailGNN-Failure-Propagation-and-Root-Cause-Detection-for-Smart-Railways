"""
Kafka configuration for Railway Failure Detection System
"""

# Kafka broker settings
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

# Topic names
TOPICS = {
    "raw_sensors": "railway.sensors.raw",
    "processed_features": "railway.sensors.processed",
    "alerts": "railway.alerts",
    "model_predictions": "railway.predictions",
}

# Producer settings
PRODUCER_CONFIG = {
    "bootstrap_servers": KAFKA_BOOTSTRAP_SERVERS,
    "value_serializer": None,  # We'll use JSON manually
    "key_serializer": None,
    "acks": "all",
    "retries": 3,
    "batch_size": 16384,
    "linger_ms": 10,
    "compression_type": "gzip",
}

# Consumer settings
CONSUMER_CONFIG = {
    "bootstrap_servers": KAFKA_BOOTSTRAP_SERVERS,
    "group_id": "railway-feature-processor",
    "auto_offset_reset": "earliest",
    "enable_auto_commit": True,
    "auto_commit_interval_ms": 1000,
    "max_poll_records": 500,
    "session_timeout_ms": 30000,
    "heartbeat_interval_ms": 10000,
}

# Streaming parameters
STREAM_CONFIG = {
    "simulation_speed_ms": 100,     # ms between records when simulating
    "rolling_window_size": 20,       # records for rolling stats
    "temporal_sequence_len": 12,     # time steps for TGNN
    "alert_cooldown_seconds": 60,    # seconds between repeated alerts
    "batch_flush_interval": 5,       # seconds between feature batches
}

# Risk thresholds
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.8,
    "critical": 0.95,
}

# Feature engineering settings
FEATURE_CONFIG = {
    "rolling_windows": [5, 10, 20],   # multiple window sizes
    "trend_window": 10,
    "anomaly_z_threshold": 3.0,       # z-score for anomaly flag
    "sensors": ["vibration", "temperature", "load", "current"],
}