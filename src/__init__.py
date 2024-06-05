from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer = Adam(
        learning_rate=0.001
    )
)

from tensorflow.keras.losses import MeanSquaredError

model.compile(
    loss = MeanSquaredError()
)

from tensorflow.keras.metrics import MeanAbsoluteError

model.compile(
    metrics=[MeanAbsoluteError()]
)

model.compile(
    metrics=['accuracy']
)

from tensorflow.keras.metrics import Precision

model.compile(
    metrics=[Precision()]
)

from tensorflow.keras.metrics import Recall

model.compile(
    metrics=[Recall()]
)

from tensorflow.keras.losses import CategoricalCrossentropy

model.compile(
    loss=CategoricalCrossentropy()
)

from tensorflow.keras.losses import BinaryCrossentropy

model.compile(
    loss=BinaryCrossentropy()
)

from tensorflow.keras.losses import SparseCategoricalCrossentropy

model.compile(
    loss=SparseCategoricalCrossentropy(),
)

evaluate_result = model.evaluate(
    x=x_test, 
    y=y_test,
)