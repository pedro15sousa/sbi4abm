# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

"""Various PyTorch utility functions."""

import warnings
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import Tensor, float32
from torch.distributions import Independent, Uniform, Categorical, constraints, Distribution

from sbi4abm.sbi import utils as utils
from sbi4abm.sbi.types import Array, OneOrMore, ScalarFloat


def process_device(device: str, prior: Optional[Any] = None) -> str:
    """Set and return the default device to cpu or gpu.

    Throws an AssertionError if the prior is not matching the training device not.
    """

    if not device == "cpu":
        if device == "gpu":
            device = "cuda"
        try:
            torch.zeros(1).to(device)
            warnings.warn(
                """GPU was selected as a device for training the neural network. Note
                   that we expect **no** significant speed ups in training for the
                   default architectures we provide. Using the GPU will be effective
                   only for large neural networks with operations that are fast on the
                   GPU, e.g., for a CNN or RNN `embedding_net`."""
            )
        except (RuntimeError, AssertionError):
            warnings.warn(f"Device {device} not available, falling back to CPU.")
            device = "cpu"

    if prior is not None:
        prior_device = prior.sample((1,)).device
        training_device = torch.zeros(1, device=device).device
        assert (
            prior_device == training_device
        ), f"""Prior ({prior_device}) device must match training device (
            {training_device}). When training on GPU make sure to pass a prior
            initialized on the GPU as well, e.g., `prior = torch.distributions.Normal
            (torch.zeros(2, device='cuda'), scale=1.0)`."""

    return device


def tile(x, n):
    if not utils.is_positive_int(n):
        raise TypeError("Argument `n` must be a positive integer.")
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if not utils.is_nonnegative_int(num_batch_dims):
        raise TypeError("Number of batch dimensions must be a non-negative integer.")
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


def merge_leading_dims(x, num_dims):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to
    one."""
    if not utils.is_positive_int(num_dims):
        raise TypeError("Number of leading dims must be a positive integer.")
    if num_dims > x.dim():
        raise ValueError(
            "Number of leading dims can't be greater than total number of dims."
        )
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    if not utils.is_positive_int(num_reps):
        raise TypeError("Number of repetitions must be a positive integer.")
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)


def tensor2numpy(x):
    return x.detach().cpu().numpy()


def logabsdet(x):
    """Returns the log absolute determinant of square matrix x."""
    # Note: torch.logdet() only works for positive determinant.
    _, res = torch.slogdet(x)
    return res


def random_orthogonal(size):
    """
    Returns a random orthogonal matrix as a 2-dim tensor of shape [size, size].
    """

    # Use the QR decomposition of a random Gaussian matrix.
    x = torch.randn(size, size)
    q, _ = torch.qr(x)
    return q


def get_num_parameters(model):
    """
    Returns the number of trainable parameters in a model of type nets.Module
    :param model: nets.Module containing trainable parameters
    :return: number of trainable parameters in model
    """
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters


def create_alternating_binary_mask(features, even=True):
    """
    Creates a binary mask of a given dimension which alternates its masking.

    :param features: Dimension of mask.
    :param even: If True, even values are assigned 1s, odd 0s. If False, vice versa.
    :return: Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


def create_mid_split_binary_mask(features):
    """
    Creates a binary mask of a given dimension which splits its masking at the midpoint.

    :param features: Dimension of mask.
    :return: Binary mask split at midpoint of type torch.Tensor
    """
    mask = torch.zeros(features).byte()
    midpoint = features // 2 if features % 2 == 0 else features // 2 + 1
    mask[:midpoint] += 1
    return mask


def create_random_binary_mask(features):
    """
    Creates a random binary mask of a given dimension with half of its entries
    randomly set to 1s.

    :param features: Dimension of mask.
    :return: Binary mask with half of its entries set to 1s, of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    weights = torch.ones(features).float()
    num_samples = features // 2 if features % 2 == 0 else features // 2 + 1
    indices = torch.multinomial(
        input=weights, num_samples=num_samples, replacement=False
    )
    mask[indices] += 1
    return mask


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def cbrt(x):
    """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)


def get_temperature(max_value, bound=1 - 1e-3):
    """
    For a dataset with max value 'max_value', returns the temperature such that

        sigmoid(temperature * max_value) = bound.

    If temperature is greater than 1, returns 1.

    :param max_value:
    :param bound:
    :return:
    """
    max_value = torch.Tensor([max_value])
    bound = torch.Tensor([bound])
    temperature = min(-(1 / max_value) * (torch.log1p(-bound) - torch.log(bound)), 1)
    return temperature


def gaussian_kde_log_eval(samples, query):
    N, D = samples.shape[0], samples.shape[-1]
    std = N ** (-1 / (D + 4))
    precision = (1 / (std ** 2)) * torch.eye(D)
    a = query - samples
    b = a @ precision
    c = -0.5 * torch.sum(a * b, dim=-1)
    d = -np.log(N) - (D / 2) * np.log(2 * np.pi) - D * np.log(std)
    c += d
    return torch.logsumexp(c, dim=-1)


class BoxUniform(Independent):
    def __init__(
        self,
        low: ScalarFloat,
        high: ScalarFloat,
        reinterpreted_batch_ndims: int = 1,
        device: str = "cpu",
    ):
        """Multidimensional uniform distribution defined on a box.

        A `Uniform` distribution initialized with e.g. a parameter vector low or high of
         length 3 will result in a /batch/ dimension of length 3. A log_prob evaluation
         will then output three numbers, one for each of the independent Uniforms in
         the batch. Instead, a `BoxUniform` initialized in the same way has three
         /event/ dimensions, and returns a scalar log_prob corresponding to whether
         the evaluated point is in the box defined by low and high or outside.

        Refer to torch.distributions.Uniform and torch.distributions.Independent for
         further documentation.

        Args:
            low: lower range (inclusive).
            high: upper range (exclusive).
            reinterpreted_batch_ndims (int): the number of batch dims to
                                             reinterpret as event dims.
            device: device of the prior, defaults to "cpu", should match the training
                device when used in SBI.
        """

        super().__init__(
            Uniform(
                low=torch.as_tensor(low, dtype=torch.float32, device=device),
                high=torch.as_tensor(high, dtype=torch.float32, device=device),
                validate_args=False,
            ),
            reinterpreted_batch_ndims,
        )

    @property
    def low(self):
        return self.base_dist.low

    @property
    def high(self):
        return self.base_dist.high
    

class IntegerUniform(Independent):
    def __init__(
            self,
            low: ScalarFloat,
            high: ScalarFloat,
            reinterpreted_batch_ndims: int = 0,
            device: str = "cpu",
    ):
        """Integer uniform distribution defined on [low, high]."""
        self.low = low
        self.high = high
        values = torch.arange(low.item(), high.item() + 1, device=device)
        probs = torch.ones_like(values, dtype=torch.float) / len(values)
        super().__init__(
            Categorical(probs=probs),
            reinterpreted_batch_ndims,
        )

    def sample(self, sample_shape=torch.Size()):
        indices = super().sample(sample_shape)
        return (self.low + indices).to(torch.int)

    @property
    def mean(self):
        return (self.low + self.high) / 2    



# class Composite(Distribution):
#     arg_constraints = {}

#     def __init__(self, low_uni, high_uni, low_cat, high_cat, validate_args=None):
#         self.low_uni = low_uni
#         self.high_uni = high_uni

#         self.uniform = Uniform(
#             low=torch.as_tensor(self.low_uni, dtype=torch.float32), 
#             high=torch.as_tensor(self.high_uni, dtype=torch.float32), 
#             validate_args=validate_args
#         )

#         self.low_cat = low_cat
#         self.high_cat = high_cat

#         values = torch.arange(self.low_cat, self.high_cat + 1)
#         probs = torch.ones_like(values, dtype=torch.float) / len(values)
#         self.categorical = Categorical(probs=probs, validate_args=validate_args)

#         batch_shape = self.uniform.batch_shape
#         event_shape = (2,)  # Two-dimensional event shape
#         super().__init__(batch_shape, event_shape, validate_args=validate_args)

#     def sample(self, sample_shape=torch.Size()):
#         box_sample = self.uniform.sample(sample_shape)
#         indices = self.categorical.sample(sample_shape)
#         # int_sample = (self.low_cat + indices).to(torch.int).unsqueeze(-1)
#         int_sample = (self.low_cat + indices).to(torch.int).unsqueeze(-1)
#         return torch.cat([box_sample, int_sample], dim=-1)
    
#     def log_prob(self, value):
#         box_value, int_value = value[..., :-1], value[..., -1].to(torch.int64) - self.low_cat
#         box_log_prob = self.uniform.log_prob(box_value)
#         int_log_prob = self.categorical.log_prob(int_value).unsqueeze(-1)
#         return box_log_prob + int_log_prob

#     @constraints.dependent_property(is_discrete=False, event_dim=1)
#     def support(self):
#         # return CompositeConstraint(self.uniform.support, [cat.support for cat in self.categoricals], self.low_uni, self.low_cat)
#         return CompositeConstraint(self.uniform.support, self.categorical.support, self.low_uni, self.low_cat)
    

# class CompositeUniform(Independent):
#     def __init__(
#             self,
#             low: ScalarFloat,
#             high: ScalarFloat,
#             reinterpreted_batch_ndims: int = 1,
#             device: str = "cpu",
#     ):
#         self.low_uni = low[0]
#         self.high_uni = high[0]
#         self.low_cat = low[1]
#         self.high_cat = high[1]
#         # self.low_uni = low[:-2]
#         # self.high_uni = high[:-2]
#         # self.low_cat = low[-2:]
#         # self.high_cat = high[-2:]
#         base_dist = Composite(self.low_uni, self.high_uni, self.low_cat, self.high_cat)
#         super().__init__(
#             base_dist, 
#             reinterpreted_batch_ndims,
#         )

#     def sample(self, sample_shape=torch.Size()):
#         return self.base_dist.sample(sample_shape)

#     def log_prob(self, value):
#         return self.base_dist.log_prob(value)

#     @property
#     def mean(self):
#         return self.base_dist.mean

#     @constraints.dependent_property
#     def support(self):
#         return self.base_dist.support
    
    
# class CompositeConstraint(constraints.Constraint):
#     def __init__(self, uniform_support, categoricals_support, low_uni, low_cat):
#         self.uniform_support = uniform_support
#         self.int_support = categoricals_support
#         self.low_uni = low_uni
#         self.low_cat = low_cat

#     @property
#     def is_discrete(self):
#         return self.box_support.is_discrete or self.categoricals_support.is_discrete

#     @property
#     def event_dim(self):
#         return max(self.box_support.event_dim, self.categoricals_support.event_dim)

#     def check(self, value):
#         box_samples, int_samples = value[..., :-1], value[..., -1].to(torch.int64) - self.low_cat
#         box_samples = torch.squeeze(box_samples, dim=-1)
#         box_check = self.uniform_support.check(box_samples)
#         # int_check = self.int_support.check(int_samples).unsqueeze(-1)
#         int_check = self.int_support.check(int_samples)
#         print("\n")
#         print("box samples: ", box_samples)
#         print("int samples: ", int_samples)
#         print("box check: ", box_check)
#         print("int check: ", int_check)
#         print("\n")
#         return box_check & int_check










class Composite(Distribution):
    arg_constraints = {}

    def __init__(self, low_uni, high_uni, low_cat, high_cat, validate_args=None):
        # Handle uniform parameters
        self.low_uni = torch.as_tensor(low_uni, dtype=torch.float32)
        self.high_uni = torch.as_tensor(high_uni, dtype=torch.float32)
        print(self.low_uni)

        self.uniform = Uniform(low=self.low_uni, high=self.high_uni, validate_args=validate_args)

        # Handle categorical parameters
        self.low_cat = torch.as_tensor(low_cat, dtype=torch.int32)
        self.high_cat = torch.as_tensor(high_cat, dtype=torch.int32)

        if self.low_cat.ndimension() == 0:
            self.low_cat = self.low_cat.unsqueeze(0)
            self.high_cat = self.high_cat.unsqueeze(0)

        self.num_cats = len(self.low_cat)
        print(self.num_cats)
        probs = []
        for low, high in zip(self.low_cat, self.high_cat):
            values = torch.arange(low, high + 1)
            probs.append(torch.ones_like(values, dtype=torch.float) / len(values))
        print("merda")
        print(probs)
        self.categorical = Categorical(probs=torch.stack(probs), validate_args=validate_args)
        print(self.categorical)

        batch_shape = self.uniform.batch_shape
        event_shape = (self.low_uni.numel() + self.num_cats,)
        print(event_shape)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        box_sample = self.uniform.sample(sample_shape)
        indices = self.categorical.sample(sample_shape).to(torch.int)
        int_samples = [
            (self.low_cat[i] + indices[..., i]).unsqueeze(-1)
            for i in range(self.num_cats)
        ]
        int_samples = torch.cat(int_samples, dim=-1)
        return torch.cat([box_sample, int_samples], dim=-1)
    
    def log_prob(self, value):
        box_values, int_values = value[..., :len(self.low_uni)], value[..., len(self.low_uni):].to(torch.int64)
        box_log_prob = self.uniform.log_prob(box_values)
        int_log_prob = self.categorical.log_prob(int_values - self.low_cat)
        print("CONAAAA")
        print(int_log_prob)
        print(box_log_prob)

        # Sum the tensors element-wise
        total_log_probs = int_log_prob.sum(dim=-1, keepdim=True) + box_log_prob.sum(dim=-1, keepdim=True)
        return total_log_probs

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self):
        return CompositeConstraint(
            self.uniform.support,
            self.categorical.support,
            self.low_uni, self.low_cat
        )
    

class CompositeUniform(Independent):
    def __init__(
            self,
            low: ScalarFloat,
            high: ScalarFloat,
            reinterpreted_batch_ndims: int = 1,
            device: str = "cpu",
    ):
        # self.low_uni = low[:-1]
        # self.high_uni = high[:-1]
        # self.low_cat = low[-1]
        # self.high_cat = high[-1]
        self.low_uni = low[:-2]
        self.high_uni = high[:-2]
        self.low_cat = low[-2:]
        self.high_cat = high[-2:]
        base_dist = Composite(self.low_uni, self.high_uni, self.low_cat, self.high_cat)
        super().__init__(
            base_dist, 
            reinterpreted_batch_ndims,
        )

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape)

    def log_prob(self, value):
        return self.base_dist.log_prob(value)

    @property
    def mean(self):
        return self.base_dist.mean

    @constraints.dependent_property
    def support(self):
        return self.base_dist.support
    
    
class CompositeConstraint(constraints.Constraint):
    def __init__(self, uniform_support, categoricals_support, low_uni, low_cat):
        self.uniform_support = uniform_support
        self.categorical_support = categoricals_support
        self.low_uni = low_uni
        self.low_cat = low_cat

    @property
    def is_discrete(self):
        return any(support.is_discrete for support in self.uniform_supports) or self.categorical_support.is_discrete

    @property
    def event_dim(self):
        return max(support.event_dim for support in self.uniform_supports) + self.categorical_support.event_dim

    def check(self, value):
        box_samples, int_samples = value[..., :len(self.low_uni)], value[..., len(self.low_uni):].to(torch.int64)
        # box_samples = torch.squeeze(box_samples, dim=-1)
        box_check = self.uniform_support.check(box_samples)
        int_check = self.categorical_support.check(int_samples - self.low_cat)

        # Initialize the result tensor with True
        if box_check.ndimension() == 1:
            result = torch.tensor(True, dtype=torch.bool)
            # Check if there is any False in the single sample
            if not box_check.all() or not int_check.all():
                result = torch.tensor(False, dtype=torch.bool)
        else:
            result = torch.ones(box_check.shape[0], dtype=torch.bool)
            # Iterate over each sample
            for i in range(box_check.shape[0]):
                # Check if there is any False in the corresponding rows of tensor1 or tensor2
                if not box_check[i].all() or not int_check[i].all():
                    result[i] = False
        print(result)
        return result





def ensure_theta_batched(theta: Tensor) -> Tensor:
    r"""
    Return parameter set theta that has a batch dimension, i.e. has shape
     (1, shape_of_single_theta)

     Args:
         theta: parameters $\theta$, of shape (n) or (1,n)
     Returns:
         Batched parameter set $\theta$
    """

    # => ensure theta has shape (1, dim_parameter)
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    return theta


def ensure_x_batched(x: Tensor) -> Tensor:
    """
    Return simulation output x that has a batch dimension, i.e. has shape
    (1, shape_of_single_x).

    Args:
         x: simulation output of shape (n) or (1,n).
     Returns:
         Batched simulation output x.
    """

    # ensure x has shape (1, shape_of_single_x). If shape[0] > 1, we assume that
    # the batch-dimension is missing, even though ndim might be >1 (e.g. for images)
    if x.shape[0] > 1 or x.ndim == 1:
        x = x.unsqueeze(0)

    return x


def atleast_2d_many(*arys: Array) -> OneOrMore[Tensor]:
    """Return tensors with at least dimension 2.

    Tensors or arrays of dimension 0 or 1 will get additional dimension(s) prepended.

    Returns:
        Tensor or list of tensors all with dimension >= 2.
    """
    if len(arys) == 1:
        arr = arys[0]
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        return atleast_2d(arr)
    else:
        return [atleast_2d_many(arr) for arr in arys]


def atleast_2d(t: Tensor) -> Tensor:
    return t if t.ndim >= 2 else t.reshape(1, -1)


def maybe_add_batch_dim_to_size(s: torch.Size) -> torch.Size:
    """
    Take a torch.Size and add a batch dimension to it if dimensionality of size is 1.

    (N) -> (1,N)
    (1,N) -> (1,N)
    (N,M) -> (N,M)
    (1,N,M) -> (1,N,M)

    Args:
        s: Input size, possibly without batch dimension.

    Returns: Batch size.

    """
    return s if len(s) >= 2 else torch.Size([1]) + s


def atleast_2d_float32_tensor(arr: Union[Tensor, np.ndarray]) -> Tensor:
    return atleast_2d(torch.as_tensor(arr, dtype=float32))


def batched_first_of_batch(t: Tensor) -> Tensor:
    """
    Takes in a tensor of shape (N, M) and outputs tensor of shape (1,M).
    """
    return t[:1]


def assert_all_finite(quantity: Tensor, description: str = "tensor") -> None:
    """Raise if tensor quantity contains any NaN or Inf element."""

    msg = f"NaN/Inf present in {description}."
    assert torch.isfinite(quantity).all(), msg


def check_if_prior_on_device(
    device: Union[str, torch.device], prior: Optional[Any] = None
) -> None:
    """Try to sample from the prior, and check that the returned data is on the correct
    trainin device. If the prior is `None`, simplys pass.

    Args:
        device: target torch training device
        prior: any simulator outputing torch `Tensor`
    """
    if prior is None:
        pass
    else:
        prior_device = prior.sample((1,)).device
        training_device = torch.zeros(1, device=device).device
        assert prior_device == training_device, (
            f"Prior device '{prior_device}' must match training device "
            f"'{training_device}'. When training on GPU make sure to "
            "pass a prior initialized on the GPU as well, e.g., "
            "prior = torch.distributions.Normal"
            "(torch.zeros(2, device='cuda'), scale=1.0)`."
        )


