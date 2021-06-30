# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Different model implementation plus a general port for all the models."""
from typing import Any, Dict, Mapping, Optional, Tuple
from functools import partial

from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
# import frozendict
from jax import random
import jax.numpy as jnp

from nerfies import configs
from nerfies import glo
from nerfies import model_utils
from nerfies import modules
from nerfies import types
from nerfies import warping


class NerfModel(nn.Module):
  """Nerf NN Model with both coarse and fine MLPs.

  Attributes:
    num_coarse_samples: int, the number of samples for coarse nerf.
    num_fine_samples: int, the number of samples for fine nerf.
    use_viewdirs: bool, use viewdirs as a condition.
    near: float, near clip.
    far: float, far clip.
    noise_std: float, std dev of noise added to regularize sigma output.
    nerf_trunk_depth: int, the depth of the first part of MLP.
    nerf_trunk_width: int, the width of the first part of MLP.
    nerf_condition_depth: int, the depth of the second part of MLP.
    nerf_condition_width: int, the width of the second part of MLP.
    activation: the activation function used in the MLP.
    sigma_activation: the activation function applied to the sigma density.
    nerf_skips: which layers to add skip layers in the NeRF model.
    alpha_channels: int, the number of alpha_channelss.
    rgb_channels: int, the number of rgb_channelss.
    use_stratified_sampling: use stratified sampling.
    use_white_background: composite rendering on to a white background.
    num_nerf_point_freqs: degree of positional encoding for positions.
    num_nerf_viewdir_freqs: degree of positional encoding for viewdirs.
    use_linear_disparity: sample linearly in disparity rather than depth.

    num_appearance_embeddings: the number of appearance exemplars.
    num_appearance_features: the dimension size for the appearance code.
    num_camera_embeddings: the number of camera exemplars.
    num_camera_features: the dimension size for the camera code.
    num_warp_freqs: the number of frequencies for the warp positional encoding.
    num_warp_embeddings: the number of embeddings for the warp GLO encoder.
    num_warp_features: the number of features for the warp GLO encoder.

    use_appearance_metadata: if True use the appearance metadata.
    use_camera_metadata: if True use the camera metadata.
    use_warp: whether to use the warp field or not.
    use_warp_jacobian: if True the model computes and returns the Jacobian of
      the warped points.
    use_weights: if True return the density weights.
    warp_kwargs: extra keyword arguments for the warp field.
  """
  num_coarse_samples: int
  num_fine_samples: int
  use_viewdirs: bool
  near: float
  far: float
  noise_std: Optional[float]
  nerf_trunk_depth: int
  nerf_trunk_width: int
  nerf_condition_depth: int
  nerf_condition_width: int
  nerf_skips: Tuple[int]
  alpha_channels: int
  rgb_channels: int
  use_stratified_sampling: bool

  # Lantent codes
  num_appearance_embeddings: int      # appearance
  num_appearance_features: int
  num_camera_embeddings: int          # camera
  num_camera_features: int
  num_warp_embeddings: int            # warp field
  num_warp_features: int
  # Let's first reproduce hypernerf:
  # @ambient_latent = @warp_latent
  # No need for extra ambient latent
  num_ambient_embeddings: int         # ambient field
  num_ambient_features: int

  # Input positional encoding
  num_nerf_point_freqs: int
  num_nerf_viewdir_freqs: int
  num_warp_freqs: int
  num_ambient_freqs: int

  num_ambient_dims: int
  activation: types.Activation = nn.relu
  sigma_activation: types.Activation = nn.relu

  # Switches
  use_white_background: bool = False
  use_linear_disparity: bool = False
  use_sample_at_infinity: bool = True

  warp_field_type: str = 'se3'
  use_appearance_metadata: bool = False
  use_camera_metadata: bool = False
  use_warp: bool = False
  use_ambient: bool = False
  sep_ambient_latent: bool = False  # If True use seperate latent for ambient field
  use_warp_jacobian: bool = False
  use_weights: bool = False
  warp_kwargs: Mapping[str, Any] = FrozenDict()
  ambient_kwargs: Mapping[str, Any] = FrozenDict()

  metadata_encoded: bool = False

  def setup(self):
    # Positional encoding
    self.point_encoder = model_utils.vmap_module(
        modules.SinusoidalEncoder, num_batch_dims=2)( # B, S
            num_freqs=self.num_nerf_point_freqs)
    self.viewdir_encoder = model_utils.vmap_module(
        modules.SinusoidalEncoder, num_batch_dims=1)( # B
            num_freqs=self.num_nerf_viewdir_freqs)
    # Embedder
    if self.use_appearance_metadata:
      self.appearance_encoder = glo.GloEncoder(
          num_embeddings=self.num_appearance_embeddings,
          features=self.num_appearance_features)
    if self.use_camera_metadata:
      self.camera_encoder = glo.GloEncoder(
          num_embeddings=self.num_camera_embeddings,
          features=self.num_camera_features)
    # Nerf MLPs
    self.nerf_coarse = modules.NerfMLP(
        nerf_trunk_depth=self.nerf_trunk_depth,
        nerf_trunk_width=self.nerf_trunk_width,
        nerf_condition_depth=self.nerf_condition_depth,
        nerf_condition_width=self.nerf_condition_width,
        activation=self.activation,
        skips=self.nerf_skips,
        alpha_channels=self.alpha_channels,
        rgb_channels=self.rgb_channels)
    if self.num_fine_samples > 0:
      self.nerf_fine = modules.NerfMLP(
          nerf_trunk_depth=self.nerf_trunk_depth,
          nerf_trunk_width=self.nerf_trunk_width,
          nerf_condition_depth=self.nerf_condition_depth,
          nerf_condition_width=self.nerf_condition_width,
          activation=self.activation,
          skips=self.nerf_skips,
          alpha_channels=self.alpha_channels,
          rgb_channels=self.rgb_channels)
    else:
      self.nerf_fine = None
    # Warp field
    if self.use_warp:
      self.warp_field = warping.create_warp_field(
          field_type=self.warp_field_type,
          num_freqs=self.num_warp_freqs,
          num_embeddings=self.num_warp_embeddings,
          num_features=self.num_warp_features,
          num_batch_dims=2,
          **self.warp_kwargs)
    # Ambient field
    if self.use_ambient:
      assert self.sep_ambient_latent or self.use_warp, \
        'Must use_warp if ambient field does not have its own latent.'
      create_ambient_field = partial(
          warping.create_warp_field,
          field_type='ambient',
          num_freqs=self.num_ambient_freqs,
          ambient_dims=self.num_ambient_dims,
          num_batch_dims=2,
          **self.ambient_kwargs)
      if self.sep_ambient_latent:
        self.ambient_field = create_ambient_field(
            num_embeddings=self.num_ambient_embeddings,
            num_features=self.num_ambient_features)
      else:
        self.ambient_field = create_ambient_field(
            num_embeddings=self.num_warp_embeddings,
            num_features=self.num_warp_features)
      # Whether the warp field should return latent
      self.warp_ret_latent = self.use_ambient and not self.sep_ambient_latent

  def __call__(
      self,
      rays_dict: Dict[str, Any],
      warp_alpha: float,
      metadata_encoded: bool = False,
      use_warp: bool = True,
  ):
    """Nerf Model.

    Args:
      rays_dict: a dictionary containing the ray information. Contains:
        'origins': the ray origins. (device_batch, 3)
        'directions': tip-aligned ray directions. (device_batch, 3)
        'viewdirs': (optional) unit vectors which are viewing directions.
        'metadata': a dictionary of metadata indices
          - warping
          - appearance
          Each has shape (device_batch, 1)
      warp_alpha (float): the alpha for the positional encoding.
      metadata_encoded (Bool): if True, assume the metadata is already encoded.
      use_warp (Bool): if True use the warp field (if also enabled in the model).
      deterministic (Bool): whether evaluation should be deterministic.

    Returns:
      ret: list, [(rgb, disp, acc), (rgb_coarse, disp_coarse, acc_coarse)]
    """
    # ------------- RAY EXTRACTION -------------
    origins = rays_dict['origins']          # (device_batch, 3)
    directions = rays_dict['directions']    # (device_batch, 3)
    metadata = rays_dict['metadata']        # {(device_batch, 1/encoded)}
    if 'viewdirs' in rays_dict:             # (device_batch, 3)
      viewdirs = rays_dict['viewdirs']
    else:  # viewdirs are normalized rays_d
      viewdirs = directions

    # -------------- COARSE MODEL --------------
    # 1. Stratified sampling along rays
    #    - z_vals.shape: (device_batch, num_course_samples,  )
    #    - points.shape: (device_batch, num_course_samples, 3)
    z_vals, points = model_utils.sample_along_rays(
        self.make_rng('coarse'), origins, directions, self.num_coarse_samples,
        self.near, self.far, self.use_stratified_sampling,
        self.use_linear_disparity)

    # 2. Apply warping to points.
    #    - points.shape: (device_batch, num_coarse_samples, 3)
    #    - warp_jacobian.shape: (device_batch, num_coarse_samples, 3, 3)
    if self.use_warp and use_warp:
      metadata_channels = self.num_warp_features if metadata_encoded else 1
      warp_metadata = metadata['warp']  # (device_batch, metadata_channels)
      warp_metadata = jnp.broadcast_to( # (device_batch, num_coarse_samples, 1/encoded)
          warp_metadata[:, None, :],
          shape=(*points.shape[:2], metadata_channels))
      # Feed points into the warp_field, which has been vmapped to first 2 
      # dims (device_batch, num_course_samples) of @points, @warp_metadata.
      # The warp field does embedding on its own, windowed by @warp_alpha.
      warp_ret = self.warp_field(
          points, warp_metadata,  # vmapped
          warp_alpha,
          metadata_encoded,
          self.use_warp_jacobian,
          self.warp_ret_latent)
      points = warp_ret['warped_points']
      if self.use_warp_jacobian:
        warp_jacobian = warp_ret['jacobian']
      if self.warp_ret_latent:
        warp_latent = warp_ret['latent']

    # 2.5 Predict ambient coordinates.
    if self.use_ambient:
      if self.sep_ambient_latent: # TODO: use own embedding
        pass
      else:  # use warp latent 
        ambient_ret = self.ambient_field(
            points, warp_latent,              # vmapped
            True, warp_alpha, False, False)   # TODO: Should have own alpha 
      ambient_w = ambient_ret['ambient_w']
      print(ambient_w.shape)

    # 3. Apply postional encoding to (warpped) points.
    #    - points_embed.shape: (device_batch, num_coarse_samples, embedded_dims)
    #    - point_endcoder vmapped to first 2 dims (device_batch, num_coarse_samples)
    points_embed = self.point_encoder(points)
    
    # 4. Append condition inputs (encoded viewdir, appearance latent)
    #    - Each condition has shape (device_batch, embedded_dims)
    #    - condition_inputs.shape: (device_batch, sum(embedded_dims))
    condition_inputs = []
    if self.use_viewdirs:
      viewdirs_embed = self.viewdir_encoder(viewdirs)
      condition_inputs.append(viewdirs_embed)
    if self.use_appearance_metadata:
      if metadata_encoded:
        appearance_code = metadata['appearance']
      else:
        appearance_code = self.appearance_encoder(metadata['appearance'])
      condition_inputs.append(appearance_code)
    if self.use_camera_metadata:
      if metadata_encoded:
        camera_code = metadata['camera']
      else:
        camera_code = self.camera_encoder(metadata['camera'])
      condition_inputs.append(camera_code)
    # The condition inputs have a shape of (B, C) now rather than (B, S, C)
    # since we assume all samples have the same condition input. We might want
    # to change this later.
    # Broadcasted to (device_batch, num_coarse_samples, sum(embedded_dims)) 
    # in the @nerf_coarse MLP.
    condition_inputs = (
        jnp.concatenate(condition_inputs, axis=-1)
        if condition_inputs else None)

    # 5. Query the coarse network
    #    - coarse_raw: {'rgb'  : (device_batch, num_coarse_samples, 3)
    #                   'alpha': (device_batch, num_coarse_samples, 1)}
    coarse_raw = self.nerf_coarse(points_embed, condition=condition_inputs)
    # Add noises to regularize the density predictions if needed
    coarse_raw = model_utils.noise_regularize(
        self.make_rng('coarse'), coarse_raw, self.noise_std,
        self.use_stratified_sampling)

    # 6. Volumetric rendering.
    rgb, exp_depth, med_depth, disp, acc, weights = (
        model_utils.volumetric_rendering(
            coarse_raw,
            z_vals,
            directions,
            use_white_background=self.use_white_background,
            sigma_activation=self.sigma_activation,
            sample_at_infinity=self.use_sample_at_infinity))
    coarse = {
      'rgb': rgb,               # (device_batch, 3)
      'depth': exp_depth,       # (device_batch,  )
      'med_depth': med_depth,   # (device_batch,  )
      'disp': disp,             # (device_batch,  )
      'acc': acc                # (device_batch,  )
    }
    if self.use_warp and use_warp and self.use_warp_jacobian:
      coarse['warp_jacobian'] = warp_jacobian
    if self.use_weights:
      coarse['weights'] = weights
    out = {'coarse': coarse}


    # --------------- FINE MODEL ---------------
    if self.num_fine_samples > 0:
      # 1. Hierarchical sampling along rays based on coarse weights
      #    - z_vals.shape: (device_batch, num_fine_samples,  )
      #    - points.shape: (device_batch, num_fine_samples, 3)
      z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
      z_vals, points = model_utils.sample_pdf(
          self.make_rng('fine'),
          z_vals_mid,
          weights[..., 1:-1],
          origins,
          directions,
          z_vals,
          self.num_fine_samples,
          self.use_stratified_sampling,
      )
      
      # 2. Apply warping to points.
      #    - points.shape: (device_batch, num_fine_samples, 3)
      if self.use_warp and use_warp:
        metadata_channels = self.num_warp_features if metadata_encoded else 1
        warp_metadata = jnp.broadcast_to(
            metadata['warp'][:, None, :],
            shape=(*points.shape[:2], metadata_channels))
        warp_ret = self.warp_field(
            points, warp_metadata,  # vmapped
            metadata_encoded, warp_alpha,
            False, self.warp_ret_latent)
        points = warp_ret['warped_points']
      if self.warp_ret_latent:
        warp_latent = warp_ret['latent']
    
      # 2.5 Predict ambient coordinates.
      if self.use_ambient:
        if self.sep_ambient_latent: # TODO: use own embedding
          pass
        else:  # use warp latent 
          ambient_ret = self.ambient_field(
              points, warp_latent,    # vmapped
              True, warp_alpha)       # TODO: Should have own alpha 
        ambient_w = ambient_ret['ambient_w']
        print(ambient_w.shape)
      
      # 3. Apply postional encoding to (warpped) points.
      #    - points_embed.shape: (device_batch, num_fine_samples, embedded_dims)
      #    - point_endcoder vmapped to first 2 dims (device_batch, num_fine_samples)
      points_embed = self.point_encoder(points)

      # 4. Append condition inputs (encoded viewdir, appearance latent)
      #    - SKIPPED. We have the save rays as in coarse model
      #    - condition_inputs.shape: (device_batch, sum(embedded_dims))
      # 5. Query the fine network
      #    - fine_raw: {'rgb'  : (device_batch, num_fine_samples, 3)
      #                 'alpha': (device_batch, num_fine_samples, 1)}
      fine_raw = self.nerf_fine(points_embed, condition=condition_inputs)
      fine_raw = model_utils.noise_regularize(
          self.make_rng('fine'), fine_raw, self.noise_std,
          self.use_stratified_sampling)

      # 6. Volumetric rendering.
      rgb, exp_depth, med_depth, disp, acc, weights = (
          model_utils.volumetric_rendering(
              fine_raw,
              z_vals,
              directions,
              use_white_background=self.use_white_background,
              sigma_activation=self.sigma_activation,
              sample_at_infinity=self.use_sample_at_infinity))
      fine = {
        'rgb': rgb,
        'depth': exp_depth,
        'med_depth': med_depth,
        'disp': disp,
        'acc': acc,
      }
      if self.use_weights:
        fine['weights'] = weights
      out['fine'] = fine

    return out


def nerf(key,
         config: configs.ModelConfig,
         batch_size: int,
         num_appearance_embeddings: int,
         num_camera_embeddings: int,
         num_warp_embeddings: int,
         num_ambient_embeddings: int,
         near: float,
         far: float,
         use_warp_jacobian: bool = False,
         use_weights: bool = False):
  """Neural Randiance Field.

  Args:
    key: jnp.ndarray. Random number generator.
    config: model configs.
    batch_size: the evaluation batch size used for shape inference.

    num_appearance_embeddings: the number of appearance embeddings.
    num_camera_embeddings: the number of camera embeddings.
    num_warp_embeddings: the number of warp embeddings.
    num_ambient_embeddings: the number of ambient embeddings.

    near: the near plane of the scene.
    far: the far plane of the scene.
    use_warp_jacobian: if True the model computes and returns the Jacobian of
      the warped points.
    use_weights: if True return the density weights from the NeRF.

  Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """
  num_nerf_point_freqs = config.num_nerf_point_freqs
  num_nerf_viewdir_freqs = config.num_nerf_viewdir_freqs
  num_coarse_samples = config.num_coarse_samples
  num_fine_samples = config.num_fine_samples
  use_viewdirs = config.use_viewdirs
  noise_std = config.noise_std
  use_stratified_sampling = config.use_stratified_sampling
  use_white_background = config.use_white_background
  nerf_trunk_depth = config.nerf_trunk_depth
  nerf_trunk_width = config.nerf_trunk_width
  nerf_condition_depth = config.nerf_condition_depth
  nerf_condition_width = config.nerf_condition_width
  nerf_skips = config.nerf_skips
  alpha_channels = config.alpha_channels
  rgb_channels = config.rgb_channels
  use_linear_disparity = config.use_linear_disparity

  model = NerfModel(
      num_coarse_samples=num_coarse_samples,
      num_fine_samples=num_fine_samples,
      use_viewdirs=use_viewdirs,
      near=near,
      far=far,
      noise_std=noise_std,
      nerf_trunk_depth=nerf_trunk_depth,
      nerf_trunk_width=nerf_trunk_width,
      nerf_condition_depth=nerf_condition_depth,
      nerf_condition_width=nerf_condition_width,
      activation=config.activation,
      sigma_activation=config.sigma_activation,
      nerf_skips=nerf_skips,
      alpha_channels=alpha_channels,
      rgb_channels=rgb_channels,
      use_stratified_sampling=use_stratified_sampling,
      use_white_background=use_white_background,
      use_sample_at_infinity=config.use_sample_at_infinity,
      use_linear_disparity=use_linear_disparity,
      use_weights=use_weights,
      
      # input postional encoding
      num_nerf_point_freqs=num_nerf_point_freqs,
      num_nerf_viewdir_freqs=num_nerf_viewdir_freqs,
      num_warp_freqs=config.num_warp_freqs,
      num_ambient_freqs=config.num_ambient_freqs,
      # TODO: num_ambient_to_template_freqs

      # warp/ambient fields
      use_warp=config.use_warp,
      use_warp_jacobian=use_warp_jacobian,
      warp_field_type=config.warp_field_type,
      warp_kwargs=config.warp_kwargs,
      use_ambient=config.use_ambient,
      num_ambient_dims=config.num_ambient_dims,
      ambient_kwargs=config.ambient_kwargs,

      # latents
      use_appearance_metadata=config.use_appearance_metadata,
      num_appearance_embeddings=num_appearance_embeddings,    # specified by datasource at last
      num_appearance_features=config.appearance_metadata_dims,
      use_camera_metadata=config.use_camera_metadata,      
      num_camera_embeddings=num_camera_embeddings,            # specified by datasource at last
      num_camera_features=config.camera_metadata_dims,
      num_warp_embeddings=num_warp_embeddings,
      num_warp_features=config.num_warp_features,
      num_ambient_embeddings=num_ambient_embeddings,
      num_ambient_features=config.num_ambient_features,
      
  )

  print(model)

  init_rays_dict = {
      'origins': jnp.ones((batch_size, 3), jnp.float32),
      'directions': jnp.ones((batch_size, 3), jnp.float32),
      'metadata': {
          'warp': jnp.ones((batch_size, 1), jnp.uint32),
          'camera': jnp.ones((batch_size, 1), jnp.uint32),
          'appearance': jnp.ones((batch_size, 1), jnp.uint32),
      }
  }

  '''
  NerfModel.__call__(
      self,
      rays_dict: Dict[str, Any],
      warp_alpha: float = None,
      metadata_encoded=False,
      use_warp=True
  )
  '''
  key, key1, key2 = random.split(key, 3)
  params = model.init({
      'params': key,
      'coarse': key1,
      'fine': key2
  }, init_rays_dict, warp_alpha=0.0)['params']

  return model, params
