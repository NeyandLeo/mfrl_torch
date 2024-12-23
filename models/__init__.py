from . import mf
from . import IL
MFQ = mf.MFModel
IL = IL.ILModel
def choose_model(model_name):
    if model_name == "MFQ":
        model = MFQ()
    elif model_name == "IL":
        model = IL()
    return model

