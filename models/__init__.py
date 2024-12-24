from . import mfq
from . import IL
MFQ = mfq.MFQModel
IL = IL.ILModel
def choose_model(model_name,input_channels=5,num_actions=21):
    if model_name == "MFQ":
        model = MFQ(input_channels,num_actions)
    elif model_name == "IL":
        model = IL(input_channels,num_actions)
    return model

