"""Explicit opt-in training reward metrics; no changes to rewards or gradients."""
import math


def component_metrics(values):
    result = {}
    for key in ('accuracy', 'format', 'score'):
        rows = values.get(key)
        if rows is None:
            raise ValueError('Missing training reward component: ' + key)
        rows = [float(v) for v in rows]
        if not rows or not all(math.isfinite(v) and 0 <= v <= 1 for v in rows):
            raise ValueError('Invalid training reward component: ' + key)
        result['train/' + ('reward' if key == 'score' else key)] = sum(rows) / len(rows)
        result['train/' + key + '_count'] = len(rows)
    if len({result['train/' + key + '_count'] for key in ('accuracy', 'format', 'score')}) != 1:
        raise ValueError('Reward component denominators differ')
    return result
