# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch on GCP's ML Engine."""

import datetime
import os
import subprocess
import sys

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import cloud_tpu as cloud
from tensor2tensor.utils import registry
import tensorflow as tf

FLAGS = tf.flags.FLAGS
PACKAGE_NAME = 'trainer'
VERSION = 0.1

CONSOLE_URL = 'https://console.cloud.google.com/mlengine/jobs/'

# TODO(rsepassi):
# * Enable multi-machine sync/async training

def job_dir():
  # The flag --job-dir is parsed differently before and after switching to absl
  return getattr(FLAGS, 'job-dir', '') or getattr(FLAGS, 'job_dir', '')


def flags_as_args():
  """Convert FLAGS to list of args suitable for passing on cmd line."""
  if hasattr(FLAGS, 'flag_values_dict'):
    args_dict = FLAGS.flag_values_dict()
  else:
    args_dict = dict(FLAGS.__dict__['__flags'])
  del args_dict['cloud_mlengine']
  # Configured later
  del args_dict['t2t_usr_dir']
  args_dict.pop('h', None)
  args_dict.pop('helpfull', None)
  args_dict.pop('helpshort', None)
  args_dict.pop('help', None)
  args = []
  for name, val in args_dict.items():
    if val is None:
      continue
    if name.startswith('autotune'):
      continue
    args.extend(['--%s' % name, str(val)])
  return args


def _get_additional_packages():
  if FLAGS.packages:
    return FLAGS.packages.split(',')
  return []


def machine_config(num_gpus=1, use_tpu=False, master_type=None):
  """Return dict specifying machine config for trainingInput."""
  scale_tier = 'BASIC_GPU'
  if use_tpu:
    scale_tier = 'BASIC_TPU'
  elif num_gpus <= 0:
    scale_tier = 'BASIC'
  elif num_gpus > 1:
    scale_tier = 'CUSTOM'

  config = {'scaleTier': scale_tier}

  if scale_tier == 'CUSTOM':
    assert num_gpus > 1
    if num_gpus not in [4, 8]:
      raise ValueError('Must use exactly 1, 4, or 8 GPUs.')
    config['masterType'] = ('complex_model_m_gpu'
                            if num_gpus == 4 else 'complex_model_l_gpu')

  if master_type:
    config['masterType'] = master_type

  return config


def configure_job():
  """Construct jobSpec for ML Engine job."""
  train_dir = FLAGS.output_dir
  assert train_dir.startswith('gs://')

  # See documentation:
  # https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#traininginput
  training_input = {
      'pythonModule': 'tensor2tensor.bin.t2t_trainer',
      'args': flags_as_args(),
      'region': cloud.default_region(),
      'runtimeVersion': '1.4',
      'pythonVersion': '3.5' if sys.version_info.major == 3 else '2.7',
      'jobDir': train_dir,
  }
  training_input.update(
      machine_config(
          num_gpus=FLAGS.worker_gpu,
          use_tpu=FLAGS.use_tpu,
          master_type=FLAGS.cloud_mlengine_master_type))
  if FLAGS.hparams_range:
    assert FLAGS.autotune_objective
    tf.logging.info('Configuring hyperparameter tuning.')
    training_input['hyperparameters'] = configure_autotune(
        FLAGS.hparams_range,
        FLAGS.autotune_objective,
        FLAGS.autotune_maximize,
        FLAGS.autotune_max_trials,
        FLAGS.autotune_parallel_trials,
    )

  if training_input['scaleTier'] == 'CUSTOM':
    assert 'masterType' in training_input
  else:
    assert 'masterType' not in training_input

  timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  job_name = '%s_%s_t2t_%s' % (FLAGS.model, FLAGS.problems, timestamp)
  job_spec = {'jobId': job_name, 'trainingInput': training_input}
  return job_spec


def launch_job(job_spec):
  """Launch job on ML Engine."""
  project_id = 'projects/{}'.format(cloud.default_project())
  credentials = GoogleCredentials.get_application_default()
  cloudml = discovery.build('ml', 'v1', credentials=credentials)
  request = cloudml.projects().jobs().create(body=job_spec, parent=project_id)
  request.execute()


def autotune_paramspecs(hparams_range):
  rhp = common_hparams.RangedHParams()
  registry.ranged_hparams(hparams_range)(rhp)
  return rhp.to_parameter_specs(name_prefix='hp_')


def configure_autotune(hparams_range,
                       objective,
                       maximize=True,
                       max_trials=10,
                       parallel_trials=1):
  return {
      'goal': 'MAXIMIZE' if maximize else 'MINIMIZE',
      'params': autotune_paramspecs(hparams_range),
      'maxTrials': max_trials,
      'maxParallelTrials': parallel_trials,
      'hyperparameterMetricTag': objective,
  }


def configure_trainer_package(job_spec, t2t_tar):
  assert t2t_tar.startswith('gs://')
  job_spec['trainingInput']['packageUris'] = [t2t_tar]
  if FLAGS.t2t_usr_dir:
    usr_args = ['--t2t_usr_dir', os.path.basename(FLAGS.t2t_usr_dir)]
    job_spec['trainingInput']['args'].extend(usr_args)


def build_t2t_python_package():
  """Writes setup.py file and builds trainer package."""
  if FLAGS.t2t_usr_dir:
    top_dir = os.path.dirname(os.path.abspath(FLAGS.t2t_usr_dir))
  else:
    top_dir = '.'
  setup_file_path = os.path.join(top_dir, 'setup.py')
  # TODO(bgb): Allow the user to pin tensor2tensor to a specific version.
  packages = ['tensor2tensor']
  packages.extend(_get_additional_packages())
  # TODO(bgb): Add where clause to find_packages() to only include usr_dir.
  setup_py = """
from setuptools import find_packages
from setuptools import setup
setup(
    name='{package_name}',
    version='{version}',
    packages=find_packages(),
    install_requires={pypi_packages},
    include_package_data=True
)
""".format(
    package_name=PACKAGE_NAME,
    version=VERSION,
    pypi_packages=str(packages))
  with tf.gfile.Open(setup_file_path, 'w') as f:
    f.write(setup_py)
  command = ['python', setup_file_path, 'sdist']
  popen = subprocess.Popen(command, subprocess.PIPE, cwd=top_dir)
  popen.wait()


def upload_trainer_package_to_gcs(train_dir):
  """Upload trainer package to GCS.

  Args:
    train_dir: The GCS directory in which to stage the trainer package.
  Returns:
    The path to the trainer package staged in GCS."""
  tf.logging.info('Uploading trainer package to %s.', train_dir)
  src_base = '{}-{}.tar.gz'.format(PACKAGE_NAME, VERSION)
  package_path = os.path.join(os.getcwd(), 'dist', src_base)
  final_destination = os.path.join(train_dir, src_base)
  cloud.shell_run(
      ('gsutil cp {package_path} '
       '{final_destination}'),
      package_path=package_path,
      final_destination=final_destination)
  return final_destination


def launch():
  """Launch t2t_trainer on Cloud ML Engine."""
  assert not FLAGS.cloud_tpu
  assert not job_dir()
  assert FLAGS.output_dir.startswith('gs://')
  assert FLAGS.data_dir.startswith('gs://')
  assert FLAGS.worker_replicas <= 1
  assert FLAGS.ps_replicas <= 0

  build_t2t_python_package()
  job_spec = configure_job()
  job_name = job_spec['jobId']
  tf.logging.info('Launching job %s with ML Engine spec:\n%s', job_name,
                  job_spec)
  assert cloud.confirm()
  train_dir = FLAGS.output_dir
  trainer_package_gcs_path = upload_trainer_package_to_gcs(train_dir)
  configure_trainer_package(job_spec, trainer_package_gcs_path)
  launch_job(job_spec)
  tf.logging.info('Launched %s. See console to track: %s.', job_name,
                  CONSOLE_URL)
