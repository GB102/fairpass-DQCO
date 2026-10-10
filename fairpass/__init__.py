"""FairPass: fairness-aware contact scheduling for open ground-station networks.

Pipeline: public data (SatNOGS, CelesTrak) -> SGP4 passes -> pass-success model
fitted on real observations -> candidate-contact QUBO -> classical / hybrid QAOA
scheduling -> schedule recommendation with uncertainty.
"""

MODEL_VERSION = "fairpass-0.1.0"
