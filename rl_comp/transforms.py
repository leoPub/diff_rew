import torch as th


class Transform:
    def transform(self, input_data):
        raise NotImplementedError

    def infer_output_info(self, input_shape, input_dtype):
        raise NotImplementedError


class OneHot(Transform):
    def __init__(self, output_dimension):
        self.output_dimension = output_dimension

    def transform(self, input_tensor):
        output_onehot = input_tensor.new(*input_tensor.shape[:-1], self.output_dimension).zero_()
        output_onehot.scatter_(-1, input_tensor.long(), 1)
        return output_onehot.float()

    def infer_output_info(self, input_shape, input_dtype):
        return (self.output_dimension,), th.float32
