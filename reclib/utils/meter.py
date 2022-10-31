__all__ = ('Meter', 'SimpleMeter', 'AverageMeter', 'LossMeter', 'AccuracyMeter', 'Monitor')

import numpy as np


class Meter:
  def __init__(self, name, default=0., fmt=':.4f'):
    self.name = name
    self.fmt = fmt
    self.value = default

  def update(self, *args):
    raise NotImplementedError

  def __repr__(self):
    fmt = '{name} {value' + self.fmt + '}'
    return fmt.format(name=self.name, value=self.value)


class SimpleMeter(Meter):
  def update(self, value):
    self.value = value


class AverageMeter(Meter):
  def __init__(self, name, default=0., fmt=':.4f'):
    super().__init__(name, default=default, fmt=fmt)
    self.running_value = 0.
    self.count = 0

  def update(self, value, count=1):
    self.running_value += value
    self.count += count
    self.value = self.running_value / self.count


class LossMeter(AverageMeter):
  def __init__(self, name, default=0., fmt=':.4e'):
    super().__init__(name, default=default, fmt=fmt)


class AccuracyMeter(AverageMeter):
  def __init__(self, name, default=0., fmt=':6.2%'):
    super().__init__(name, default=default, fmt=fmt)


class Monitor:
  def __init__(self, meter, cmp=np.greater):
    self.meter = meter
    self.cmp = cmp
    self.best_value = None

  def update(self):
    if self.best_value is None or self.cmp(self.meter.value, self.best_value):
      self.best_value = self.meter.value
      return True
    return False
