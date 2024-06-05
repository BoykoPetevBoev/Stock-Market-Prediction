from tensorflow.keras.metrics import MeanAbsoluteError

model.compile(
    metrics=[MeanAbsoluteError()]
)