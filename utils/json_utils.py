import dataclasses
import json

import numpy as np


class DataclassAndNumpyJSONEncoder(json.JSONEncoder):
    """A JSON encoder that supports dataclasses and numpy.

    Usage:
        json.dumps(dataclass_instance, cls=DataclassAndNumpyJSONEncoder)
        json.dump(dataclass_instance, file, cls=DataclassAndNumpyJSONEncoder)
    """

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif isinstance(o, (np.ndarray, np.number)):
            return o.tolist()
        return super().default(o)
