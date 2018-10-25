import numpy as np
import cv2
from tqdm import tqdm

from utils import pickle_load
from utils import pickle_save


default_params = {
  "fx_d": 366.696,
  "fy_d": 366.696,
  "cx_d": 251.72,
  "cy_d": 206.95,
  "cx_rgb": 959.5,
  "cy_rgb": 539.5,
  "fx_rgb": 1081.37,
  "fy_rgb": 1081.37,
  "trans_x": 52.928,
  "trans_y": 0.453,
  "trans_z": -0.706
}


def depth_to_world(depth, params, h_coord, v_coord):
  """
  Turn depth data into world coordinate.

  Parameters
  ----------
  depth: Depth image of shape [height, width]

  params: Camera intrinsic parameters.

  h_coord: Horizontal coordinate in image coordinate system.

  v_coord: Vertical coordinate in image coordinate system.

  Return
  ------
  A numpy ndarray of shape [height, width, 3] as x, y, z coodinate of each
  pixel.

  """
  z = depth + params['trans_z']
  x = h_coord * z / params['fx_d'] + params['trans_x']
  y = v_coord * z / params['fy_d'] + params['trans_y']
  return np.stack([x, y, z], axis=-1)


def linear_interpolation(indices, limit):
  """
  Linearly interpolate non-integral indices.

  Parameters
  ----------
  indices: Non-integral indices. An 1D vector.

  limit: Limitation of the index.

  Return
  ------
  A tuple of (floor, weight_f, ceiling, weight_c, invalid).

  floor: Floors of given indices.

  weight_f: Weights of values at floor indices.

  ceiling: Ceilings of given indices.

  weight_c: Weights of values at ceiling indices.

  invalid: Invalid indices.

  """
  floor = np.floor(indices).astype(np.int16)
  ceiling = np.ceil(indices).astype(np.int16)

  weight_f = 1 - (indices - floor)
  weight_c = 1 - (ceiling - indices)

  floor_invalid = np.logical_or(floor < 0, floor >= limit)
  floor[floor_invalid] = 0

  ceiling_invalid = np.logical_or(ceiling < 0, ceiling >= limit)
  ceiling[ceiling_invalid] = 0

  weight_f[floor_invalid] = 0
  weight_f[ceiling_invalid] = 1

  weight_c[floor_invalid] = 1
  weight_c[ceiling_invalid] = 0

  invalid = np.logical_and(
    floor_invalid,
    ceiling_invalid
  )
  return floor, np.expand_dims(weight_f, -1), \
          ceiling, np.expand_dims(weight_c, -1), invalid


def world_to_color(params, pcloud, color):
  """
  Encolor point cloud.

  Parameter
  ---------
  params: Camera intrinsic paramters.

  pcloud: Point cloud of shape [height, width, 3].

  color: Color image.

  """
  x, y, z = pcloud[..., 0], pcloud[..., 1], pcloud[..., 2]
  x = x * params['fx_rgb'] / z + params['cx_rgb']
  y = y * params['fy_rgb'] / z + params['cy_rgb']
  x1, xw1, x2, xw2, inv1 = linear_interpolation(x, color.shape[1])
  y1, yw1, y2, yw2, inv2 = linear_interpolation(y, color.shape[0])
  invalid = np.logical_or(inv1, inv2)
  depth_color = color[y1, x1] * xw1 * yw1 + \
                color[y2, x1] * xw1 * yw2 + \
                color[y1, x2] * xw2 * yw1 + \
                color[y2, x2] * xw2 * yw2
  depth_color[invalid] = 0
  return depth_color


def align(params, load_prefix, save_path):
  """
  Align depth and color images. Save everything into a single pickle object.

  Parameters
  ----------
  params: Camera intrinsic parameters.

  load_prefix: Path to load data. Will load color stream from
  `load_prefix`_color.avi, depth stream from `load_prefix`_depth.pkl, and body
  stream from `load_prefix`_body.pkl.

  save_path: Path to save result data.

  """
  color_src = cv2.VideoCapture(load_prefix + '_color.avi')
  depth_src = pickle_load(load_prefix + '_depth.pkl')
  body_src = pickle_load(load_prefix + '_body.pkl')

  depth_height = depth_src[0].shape[0]
  depth_width = depth_src[0].shape[1]

  h_coord = np.tile(
    np.reshape(np.arange(1, depth_width + 1), [1, -1]),
    [depth_height, 1]
  ) - params['cx_d']
  v_coord = np.tile(
    np.reshape(np.arange(1, depth_height + 1), [-1, 1]),
    [1, depth_width]
  ) - params['cy_d']

  pcloud_frames = []
  depth_frames = []
  color_frames = []
  body_frames = []
  for depth, body in tqdm(zip(depth_src, body_src)):
    _, color = color_src.read()
    pcloud = depth_to_world(depth, params, h_coord, v_coord)
    pcloud_frames.append(pcloud)
    color = world_to_color(params, pcloud, color)
    color_frames.append(color)
    body_frames.append(body)
    depth_frames.append(depth)

  data = {
    'pclouds': pcloud_frames,
    'depths': depth_frames,
    'colors': color_frames,
    'bodies': body_frames
  }

  pickle_save(save_path, data)


if __name__ == '__main__':
  align(default_params, 'test', 'test.pkl')
