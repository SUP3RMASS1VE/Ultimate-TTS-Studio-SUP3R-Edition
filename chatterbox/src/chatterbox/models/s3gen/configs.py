"""Configuration parameters for Conditional Flow Matching (CFM) models."""


class CFMParams:
    """Parameters for Conditional Flow Matching."""
    def __init__(
        self,
        solver='euler',
        t_scheduler='cosine',
        training_cfg_rate=0.2,
        inference_cfg_rate=0.7,
        sigma_min=1e-4
    ):
        self.solver = solver
        self.t_scheduler = t_scheduler
        self.training_cfg_rate = training_cfg_rate
        self.inference_cfg_rate = inference_cfg_rate
        self.sigma_min = sigma_min


# Default CFM parameters
CFM_PARAMS = CFMParams()
