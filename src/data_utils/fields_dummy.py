from src.config import n_feats

meta_fields = {
    "pid": int,
}

active_drug = {
    "active_drug": str,
}

five_d_asc_list = [
    "oceanic_boundlessness",
    "anxious_ego_dissolution",
    "visionary_restructuralization",
    "auditory_alterations",
    "reduction_of_vigilance"
]

five_d_asc = {
    name: float for name in five_d_asc_list
}

labels = {
    **active_drug,
    **five_d_asc,
}

features = {
    f"f{i}": float for i in range(1, n_feats + 1)
}

fields = {
    **meta_fields,
    **labels,
    **features,
}

sorted_features = sorted(features.keys())

# Sorted lexicopraphically within each field type
sorted_fields = (
    sorted(meta_fields.keys()) +
    list(active_drug.keys()) +
    five_d_asc_list +
    sorted_features
)
