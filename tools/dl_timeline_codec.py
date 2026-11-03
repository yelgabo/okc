"""Pure CPU timeline summaries and bounded, lossless judge-log transport.

Event intervals describe CUDA activity, including waits within kernels. Their union
is not SM utilization and their durations are not additive exposed stage costs.
"""
import base64
import json
import struct
import zlib


def _put_uint(data, value):
    if value < 0:
        raise ValueError('negative timeline integer')
    while value >= 128:
        data.append((value & 127) | 128)
        value >>= 7
    data.append(value)


def _get_uint(raw, at):
    value = 0
    shift = 0
    while shift < 70:
        part = raw[at]
        at += 1
        value |= (part & 127) << shift
        if part < 128:
            return value, at
        shift += 7
    raise ValueError('invalid timeline integer')


def summarize(events):
    if not events:
        raise ValueError('no CUDA activities in isolated replay')
    events = sorted(events)
    origin = events[0][0]
    end = max(e[1] for e in events)
    last = origin
    union = 0
    total = 0
    for start, stop, stream, name in events:
        if stop < start:
            raise ValueError('negative activity duration')
        total += stop - start
        union += max(0, stop - max(start, last))
        last = max(last, stop)
    return {
        'span_us': (end - origin) / 1000,
        'union_us': union / 1000,
        'uncovered_us': (end - origin - union) / 1000,
        'sum_us': total / 1000,
        # 0.1us trace coordinates; aggregate spans/union above retain raw resolution.
        'coordinate_precision_us': 0.1,
        'events': [[round((a-origin)/1000, 1), round((b-origin)/1000, 1), s, n]
                   for a, b, s, n in events],
    }


def encode_report(report, limit=11000):
    # Intern full names once, retaining all events, streams and case coverage.
    names = sorted({e[3] for row in report['rows'] for e in row['trace']['events']})
    ids = {name: i for i, name in enumerate(names)}
    packed = json.loads(json.dumps(report))
    packed['names'] = names
    data = bytearray()
    for row in packed['rows']:
        events = row['trace'].pop('events')
        streams = sorted({e[2] for e in events})
        row['_streams'] = streams
        row['_events'] = len(events)
        previous = 0
        for start, stop, stream, name in events:
            # summarize() records 0.1us trace ticks, retained exactly in transport.
            now = round(start*10)
            _put_uint(data, now-previous)
            _put_uint(data, round(stop*10)-now)
            _put_uint(data, ids[name]*len(streams)+streams.index(stream))
            previous = now
    metadata = json.dumps(packed, separators=(',', ':')).encode()
    raw = struct.pack('<I', len(metadata)) + metadata + data
    blob = base64.b64encode(zlib.compress(raw, 9)).decode()
    if len(blob) > limit:
        raise ValueError(f'complete timeline report exceeds log budget: {len(blob)} > {limit}')
    return blob


def decode_report(blob):
    raw = zlib.decompress(base64.b64decode(blob))
    size, = struct.unpack_from('<I', raw)
    report = json.loads(raw[4:4+size])
    at = 4+size
    for row in report['rows']:
        events = []
        previous = 0
        for event_idx in range(row.pop('_events')):
            delta, at = _get_uint(raw, at)
            duration, at = _get_uint(raw, at)
            code, at = _get_uint(raw, at)
            name, stream = divmod(code, len(row['_streams']))
            start = previous + delta
            stop = start + duration
            previous = start
            events.append([start/10, stop/10, row['_streams'][stream], report['names'][name]])
        row['trace']['events'] = events
        del row['_streams']
    if at != len(raw):
        raise ValueError('trailing timeline bytes')
    del report['names']
    return report
