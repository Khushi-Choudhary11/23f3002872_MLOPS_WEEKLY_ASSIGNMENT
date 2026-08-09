from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, BigQuerySource
from feast.types import Float32, Int64

# ---------------------------------------------------
# Entity
# ---------------------------------------------------

iris = Entity(
    name="iris_id",
    join_keys=["iris_id"],
)
# ---------------------------------------------------
# Data Source
# ---------------------------------------------------

iris_source = BigQuerySource(
    name="iris_source",
    table="mlops-project-500020.iris_feast.iris_features",
    timestamp_field="event_timestamp",
)

# ---------------------------------------------------
# Feature View
# ---------------------------------------------------

iris_features = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    source=iris_source,
)

# ---------------------------------------------------
# Week 8 MLSecOps: isolated source/view for poisoning experiments.
# Points at its own table (iris_features_week8_poison) so poisoned data
# never touches the table backing iris_features above. Each poisoning
# level is a distinct event_timestamp snapshot in that table; querying
# get_historical_features with a given level's timestamp returns that
# level's feature values via Feast's point-in-time join.
# ---------------------------------------------------

iris_source_poison = BigQuerySource(
    name="iris_source_poison",
    table="mlops-project-500020.iris_feast.iris_features_week8_poison",
    timestamp_field="event_timestamp",
)

iris_features_poison = FeatureView(
    name="iris_features_poison",
    entities=[iris],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    source=iris_source_poison,
)