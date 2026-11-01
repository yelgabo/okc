"""Lossless transport of CUDA-event measurement floats; source is never encoded."""

def encode_report(report):
    import base64,json,struct,zlib
    pool=[];copy=dict(report);copy['rows']=[]
    for row in report['rows']:
        r=dict(row);r['samples_us']={}
        for tag,values in row['samples_us'].items():
            offset=len(pool)
            for value in values:
                ms=struct.unpack('<f',struct.pack('<f',value/1000))[0]
                assert ms*1000==value, ('non-lossless CUDA event float',value)
                pool.append(ms)
            r['samples_us'][tag]={'offset':offset,'count':len(values)}
        copy['rows'].append(r)
    header=json.dumps(copy,separators=(',',':'),sort_keys=True).encode()
    blob=struct.pack('<I',len(header))+header+struct.pack('<'+'f'*len(pool),*pool)
    packed=base64.b85encode(zlib.compress(blob,9)).decode()
    assert len(packed)<13000, ('report transport overflow',len(packed))
    return packed


def decode_report(encoded):
    import base64,json,struct,zlib
    blob=zlib.decompress(base64.b85decode(encoded))
    assert len(blob)>=4
    n=struct.unpack_from('<I',blob)[0]
    assert 0<n<=len(blob)-4
    result=json.loads(blob[4:4+n]);pool=blob[4+n:]
    assert len(pool)%4==0
    floats=struct.unpack('<'+'f'*(len(pool)//4),pool)
    used=[]
    for row in result['rows']:
        for tag,ref in row['samples_us'].items():
            a=ref['offset'];count=ref['count'];b=a+count
            assert type(a) is int and type(count) is int and 0<=a<b<=len(floats)
            assert count==result['rounds']==24
            row['samples_us'][tag]=[1000*value for value in floats[a:b]]
            used.extend(range(a,b))
    assert sorted(used)==list(range(len(floats))), 'missing/overlapping sample slices'
    assert result['format']=='PROJ_SK_XOR_V1'
    assert len(result['rows'])==90 and len(result['seeds'])==3
    assert len({(r['layer'],r['m'],r['seed']) for r in result['rows']})==90
    return result
