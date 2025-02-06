import numpy as np

def linear_map(input_value : float, input_min : float, input_max : float, output_min : float, 
               output_max : float, positive_slope : bool) -> float:
    """
    Maps an input value from the range `[input_min, input_max]` to the range `[output_min, output_max]`
    in a linear way. The slope of the linear transformation can be positive or negative.

    Parameters
    ----------
    input_value : float
        The input value to be mapped;
    input_min : float
        The minimum value of the input range;
    input_max : float
        The maximum value of the input range;
    output_min : float
        The minimum value of the output range;
    output_max : float
        The maximum value of the output range;
    positive_slope : bool
        If True, the slope of the linear transformation is positive, otherwise it is negative.

    Returns
    -------
    output_value : float
        The mapped value.
    """

    # Clipping over the boundaries
    if input_value <= input_min:
        return output_max
    elif input_value >= input_max:
        return output_min
    else:
        normalized_value = (input_value - input_min) / (input_max - input_min)
        output = np.abs((2 * int(positive_slope) - 1) * normalized_value * (output_max - output_min) + output_max)
        if output < output_min:
            return output_min
        elif output > output_max:
            return output_max
        else:
            return output